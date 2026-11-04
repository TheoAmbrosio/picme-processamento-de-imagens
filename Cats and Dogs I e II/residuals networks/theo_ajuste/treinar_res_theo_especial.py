"""Treina com CONFIG; --avaliar-teste CHECKPOINT acessa o teste separadamente."""
import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config_res_theo_especial import CONFIG, ConfigTreino
from .dados_res_theo_especial import DatasetManifesto, ROTULOS, carregar_divisao
from .res_theo_especial import criar_res_theo_especial


def selecionar_dispositivo() -> torch.device:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nome = torch.cuda.get_device_name(dispositivo) if dispositivo.type == "cuda" else "CPU"
    print(f"Dispositivo: {dispositivo} — {nome}", flush=True)
    return dispositivo


def preparar_execucao(config: ConfigTreino, dispositivo: torch.device) -> None:
    random.seed(config.semente)
    torch.manual_seed(config.semente)
    if dispositivo.type == "cuda":
        torch.cuda.manual_seed_all(config.semente)
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
    if config.cudnn_benchmark:
        print("cuDNN benchmark habilitado: escolhe algoritmos por medição e pode variar entre execuções.")
    print("Sementes registradas; não há garantia de determinismo entre máquinas/versões.")


def validar_config_dataset(config: ConfigTreino, manifesto: Path) -> dict:
    arquivo = manifesto.parent / "config_dataset.json"
    metadados = json.loads(arquivo.read_text(encoding="utf-8"))
    esperado = {"schema": 2, "rotulos": ROTULOS, "lado": 224,
                "normalizacao_salva": False, "semente": config.semente,
                "frac_validacao": config.frac_validacao, "frac_teste": config.frac_teste,
                "avaliacao": "Resize(256, bilinear) + CenterCrop(224)"}
    if config.divisao != "hash":
        esperado.update(divisao=config.divisao, aceitar_rotulos_duplicados=config.aceitar_rotulos_duplicados)
    if metadados != esperado:
        raise ValueError("Dataset/configuração incompatíveis; confira frações, semente e esquema.")
    metadados = dict(metadados)
    metadados["uso_res_theo_especial"] = {
        "treino": config.esquema_ampliacao,
        "epocas_iniciais_sem_ampliacao": config.epocas_iniciais_sem_ampliacao,
        "variantes_por_original_treino": 5,
        "validacao": "original esticada para RGB 224x224",
        "ruido_gaussiano_variancia": config.ruido_gaussiano_variancia,
        "pixels_brancos_fracao": config.pixels_brancos_fracao,
        "cutout_retangulos": config.cutout_retangulos,
        "cutout_lados_pixels": [config.cutout_lado_minimo, config.cutout_lado_maximo],
    }
    return metadados


def criar_loader(amostras, config: ConfigTreino, dispositivo: torch.device, treino: bool):
    return DataLoader(
        DatasetManifesto(
            amostras, usar_dados_ampliados=config.usar_dados_ampliados,
            esquema_ampliacao=config.esquema_ampliacao, semente=config.semente,
            ruido_gaussiano_variancia=config.ruido_gaussiano_variancia,
            pixels_brancos_fracao=config.pixels_brancos_fracao,
            cutout_retangulos=config.cutout_retangulos,
            cutout_lado_minimo=config.cutout_lado_minimo,
            cutout_lado_maximo=config.cutout_lado_maximo),
        batch_size=config.batch_size, shuffle=treino,
        num_workers=config.num_workers, pin_memory=dispositivo.type == "cuda",
        persistent_workers=config.num_workers > 0,
        generator=torch.Generator().manual_seed(config.semente),
        # Mantém inclusive um eventual último batch com apenas uma imagem.
        drop_last=False,
    )


def criar_otimizador(modelo: nn.Module, config: ConfigTreino):
    classe = getattr(torch.optim, config.otimizador, None)
    if not isinstance(classe, type) or not issubclass(classe, torch.optim.Optimizer):
        raise ValueError("otimizador deve ser uma classe válida de torch.optim.")
    if config.otimizador == "LBFGS":
        raise ValueError("Este loop usa step() sem closure; LBFGS exige outro loop explícito.")
    # Inclui o backbone desde o início: quando descongelado, ele já pertence ao
    # mesmo otimizador. Com requires_grad=False e grad=None, não é atualizado.
    taxa_inicial = (config.learning_rate / config.warmup_fator_inicial
                    if config.warmup_linear else config.learning_rate)
    otimizador = classe(modelo.parameters(), lr=taxa_inicial,
                       **config.argumentos_otimizador)
    print(f"Otimizador definido pelo participante: {config.otimizador}; "
          f"opções efetivas (incluindo defaults da biblioteca): {otimizador.defaults}")
    return otimizador


def taxa_aprendizado_epoca(config: ConfigTreino, epoca: int) -> float:
    if not config.warmup_linear or epoca >= config.warmup_epoca_final:
        return config.learning_rate
    inicial = config.learning_rate / config.warmup_fator_inicial
    progresso = (epoca - 1) / (config.warmup_epoca_final - 1)
    return inicial + (config.learning_rate - inicial) * progresso


def configurar_learning_rate_epoca(otimizador, config: ConfigTreino, epoca: int) -> float:
    taxa = taxa_aprendizado_epoca(config, epoca)
    for grupo in otimizador.param_groups:
        grupo["lr"] = taxa
    return taxa


def probabilidade_dropout_epoca(config: ConfigTreino, epoca: int) -> float:
    if not config.dropout_crescente:
        return config.dropout_probabilidade
    if epoca <= config.dropout_epoca_inicio:
        return 0.0
    if epoca < config.dropout_epoca_maximo:
        progresso = ((epoca - config.dropout_epoca_inicio)
                     / (config.dropout_epoca_maximo - config.dropout_epoca_inicio))
        return config.dropout_probabilidade * progresso
    inicio_reducao = config.dropout_epoca_inicio_reducao
    if inicio_reducao is None or epoca <= inicio_reducao:
        return config.dropout_probabilidade
    final = config.dropout_probabilidade_final
    fim_reducao = config.dropout_epoca_fim_reducao
    if epoca >= fim_reducao:
        return final
    progresso = (epoca - inicio_reducao) / (fim_reducao - inicio_reducao)
    return config.dropout_probabilidade + (final - config.dropout_probabilidade) * progresso


def usar_ampliacao_na_epoca(config: ConfigTreino, epoca: int) -> bool:
    return epoca > config.epocas_iniciais_sem_ampliacao


class ControleBackbone:
    """Contagem global de atualizações efetivas, atravessando as épocas."""
    def __init__(self, modelo, passos_congelado: int, congelar_sempre: bool = False,
                 epocas_congelado=None):
        self.modelo = modelo  # Módulo original, inclusive se o executor for compilado.
        self.passos_congelado = passos_congelado
        self.epocas_congelado = epocas_congelado
        self.epoca = 1
        self.congelar_sempre = congelar_sempre
        self.passos_concluidos = 0

    def preparar_batch(self) -> None:
        fase_inicial = (self.epoca <= self.epocas_congelado if self.epocas_congelado is not None
                        else self.passos_concluidos < self.passos_congelado)
        congelar = self.congelar_sempre or fase_inicial
        if self.modelo.backbone_congelado != congelar:
            self.modelo.congelar_backbone(congelar)
            if not congelar:
                print(f"Backbone liberado: época {self.epoca}; próximo passo = {self.passos_concluidos + 1}; "
                      "fc e camadas anteriores passam a treinar.", flush=True)

    def passo_concluido(self, otimizador, args, kwargs) -> None:
        # Hook de optimizer.step: GradScaler não o chama quando pula uma atualização
        # por gradientes não finitos. Não precisa consultar a escala na CPU por batch.
        self.passos_concluidos += 1


def formatar_duracao(segundos):
    segundos = max(0, round(segundos))
    horas, resto = divmod(segundos, 3600)
    minutos, segundos = divmod(resto, 60)
    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"


class ProgressoTreino:
    """ETA aproximada pela média dos batches concluídos; atualizada no log."""
    def __init__(self, total_batches, intervalo=5):
        self.total = total_batches
        self.intervalo = intervalo
        self.inicio = time.perf_counter()
        self.ultimo = self.inicio
        self.concluidos = 0

    def atualizar(self, fase, epoca, epocas, batch, batches, dispositivo):
        self.concluidos += 1
        agora = time.perf_counter()
        if batch != 1 and batch != batches and agora - self.ultimo < self.intervalo:
            return
        # Espera a GPU somente ao imprimir para medir trabalho efetivamente concluído.
        if dispositivo.type == "cuda":
            torch.cuda.synchronize(dispositivo)
        agora = time.perf_counter()
        self.ultimo = agora
        decorrido = agora - self.inicio
        restante = decorrido / self.concluidos * (self.total - self.concluidos)
        print(f"Época {epoca}/{epocas} | {fase}: batch {batch}/{batches} | "
              f"progresso total {100 * self.concluidos / self.total:.2f}% | "
              f"decorrido {formatar_duracao(decorrido)} | "
              f"falta aproximadamente {formatar_duracao(restante)}", flush=True)


def salvar_historico_e_graficos(historico, pasta, *, graficos=False):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "historico.csv"
    temporario = destino.with_suffix(".csv.tmp")
    with temporario.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(historico[0]))
        escritor.writeheader()
        escritor.writerows(historico)
    temporario.replace(destino)
    if not graficos:
        return
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figura = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figura)
    eixos = figura.subplots(3, 1, squeeze=False)
    for coluna, (chave_x, titulo_x) in enumerate((("epoca", "Época"),)):
        x = [h[chave_x] / (3600 if chave_x == "tempo_segundos" else 1) for h in historico]
        for linha, (metrica, titulo_y) in enumerate((("bce", "Perda BCE"),
                                                    ("acuracia", "Taxa de acerto (%)"),
                                                    ("f1", "F1-score (Dog=1)"))):
            eixo = eixos[linha, coluna]
            for split, nome in (("treino", "Treino"), ("validacao", "Validação")):
                y = [h[f"{metrica}_{split}"] * (100 if metrica == "acuracia" else 1)
                     for h in historico]
                eixo.plot(x, y, label=nome, marker="." if len(x) == 1 else None)
            eixo.set(xlabel=titulo_x, ylabel=titulo_y)
            if metrica == "acuracia":
                eixo.set_ylim(0, 100)
            elif metrica == "f1":
                eixo.set_ylim(0, 1)
            eixo.grid(alpha=0.3)
            eixo.legend()
    figura.suptitle("res_theo_especial — treinamento e validação")
    destino = pasta / "graficos_treinamento.png"
    temporario = destino.with_suffix(".png.tmp")
    figura.savefig(temporario, format="png", dpi=160)
    temporario.replace(destino)
    painel = pasta / "painel_treinamento.html"
    if not painel.exists():
        painel.write_text('<!doctype html>\n<html lang="pt-BR"><meta charset="utf-8"><title>res_theo_especial — progresso</title>\n<style>body{font-family:system-ui;margin:24px;background:#fafafa}img{max-width:100%;height:auto}</style>\n<h1>Treinamento e validação</h1>\n<p>O gráfico é salvo ao terminar cada época. Esta página verifica atualizações a cada 30 segundos.</p>\n<img id="grafico" src="graficos_treinamento.png" alt="Perda e taxa de acerto de treino e validação">\n<script>setInterval(()=>{document.getElementById(\'grafico\').src=\'graficos_treinamento.png?t=\'+Date.now()},30000)</script>\n</html>\n', encoding="utf-8")
    print(f"Painel: {painel}", flush=True)
    print(f"Histórico: {pasta / 'historico.csv'}; gráficos: {destino}", flush=True)


def executar_epoca(modelo, loader, dispositivo, *, otimizador=None, scaler=None,
                   usar_amp=False, channels_last=False, controle_backbone=None,
                   metricas=None, progresso=None, epoca=1, epocas=1) -> float:
    treino = otimizador is not None
    modelo.train(treino)
    criterio = nn.BCEWithLogitsLoss()
    soma_perdas = torch.zeros((), device=dispositivo, dtype=torch.float32)
    quantidade = 0
    acertos = torch.zeros((), device=dispositivo, dtype=torch.int64) if metricas is not None else None
    confusao = torch.zeros(3, device=dispositivo, dtype=torch.int64)
    with torch.set_grad_enabled(treino):
        for indice_batch, (imagens, rotulos) in enumerate(loader, 1):
            if treino and controle_backbone is not None:
                controle_backbone.epoca = epoca
                controle_backbone.preparar_batch()
            imagens = imagens.to(dispositivo, non_blocking=dispositivo.type == "cuda")
            rotulos = rotulos.to(dispositivo, dtype=torch.float32,
                                non_blocking=dispositivo.type == "cuda")
            if channels_last:
                imagens = imagens.contiguous(memory_format=torch.channels_last)
            if treino:
                otimizador.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dispositivo.type, dtype=torch.float16,
                                    enabled=usar_amp):
                logits = modelo(imagens)
                if logits.shape != rotulos.shape or logits.shape != (imagens.size(0), 1):
                    raise ValueError("Logits e rótulos devem ter shape [batch_size, 1].")
                perda = criterio(logits, rotulos)  # Sem sigmoid, arredondamento ou softmax.
            if treino:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(perda).backward()
                    scaler.step(otimizador)
                    scaler.update()
                else:
                    perda.backward()
                    otimizador.step()
            batch = imagens.size(0)
            soma_perdas += perda.detach().float() * batch
            quantidade += batch
            if acertos is not None:
                # Logit >= 0 equivale a sigmoid(logit) >= 0.5; não altera a perda.
                acertos += ((logits.detach() >= 0) == (rotulos >= 0.5)).sum()
                previstos, reais = logits.detach() >= 0, rotulos >= 0.5
                confusao += torch.stack(((previstos & reais).sum(),
                                        (previstos & ~reais).sum(),
                                        (~previstos & reais).sum()))
            if progresso is not None:
                progresso.atualizar("treino" if treino else "validação", epoca, epocas,
                                    indice_batch, len(loader), dispositivo)
    if quantidade == 0:
        raise ValueError("DataLoader vazio.")
    # Uma transferência escalar ao fim da época; sem loss.item() por batch.
    media = (soma_perdas / quantidade).item()
    if not torch.isfinite(torch.tensor(media)):
        raise FloatingPointError("BCE não finita; execução interrompida.")
    if metricas is not None:
        metricas.update(bce=media, acuracia=acertos.item() / quantidade, amostras=quantidade)
        tp, fp, fn = confusao.tolist()
        metricas["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return media


def salvar_checkpoint(modelo, otimizador, scaler, config, epoca, historico,
                      resumo_pesos, metadados, manifesto_sha256, passos_otimizador):
    destino = Path(config.checkpoint)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".pt.tmp")
    torch.save({
        "nome_modelo": "theo_ajuste", "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "scaler": scaler.state_dict(), "config": asdict(config), "epoca": epoca,
        "passos_otimizador": passos_otimizador,
        "backbone_congelado": modelo.backbone_congelado,
        "historico": historico, "rotulos": ROTULOS, "resumo_pesos": resumo_pesos,
        "dataset": metadados, "manifesto_sha256": manifesto_sha256,
        "torch_version": str(torch.__version__),
        "dropout": modelo.dropout_comum.estado(),
    }, temporario)
    temporario.replace(destino)


def treinar(config: ConfigTreino) -> None:
    config.validar()  # Falha antes de ler imagens ou baixar pesos se faltam escolhas.
    manifesto = Path(config.manifesto)
    metadados = validar_config_dataset(config, manifesto)
    conjuntos = carregar_divisao(
        manifesto,
        ("treino", "validacao") if config.frac_teste == 0 else ("treino", "validacao", "teste"),
        aceitar_rotulos_duplicados=config.aceitar_rotulos_duplicados,
        usar_dados_ampliados=config.usar_dados_ampliados,
        esquema_ampliacao=config.esquema_ampliacao,
    )
    dispositivo = selecionar_dispositivo()
    preparar_execucao(config, dispositivo)
    modo_dados = ("ampliação especial: cinco versões por original somente no treino; "
                  "validação com uma original esticada" if config.usar_dados_ampliados
                  else "originais: uma imagem por original, esticada para RGB 224x224")
    print(f"Dados: {modo_dados}.", flush=True)
    print(f"Rótulos: {ROTULOS}; amostras por conjunto: "
          + str({k: len(v) for k, v in conjuntos.items()}))
    loaders = {split: criar_loader(conjuntos[split], config, dispositivo, split == "treino")
               for split in ("treino", "validacao")}
    treino_original = [amostra for amostra in conjuntos["treino"] if amostra[2] == "original"]
    loader_treino_original = criar_loader(treino_original, config, dispositivo, True)
    modelo, resumo = criar_res_theo_especial(
        config.congelar_backbone or (config.epocas_backbone_congelado > 0
                                    if config.epocas_backbone_congelado is not None
                                    else config.passos_backbone_congelado > 0), config=config
    )
    origem = torch.load(config.checkpoint_origem, map_location="cpu", weights_only=False)
    if origem.get("manifesto_sha256") != hashlib.sha256(manifesto.read_bytes()).hexdigest():
        raise ValueError("O manifesto difere daquele do checkpoint de origem.")
    modelo.load_state_dict(origem["modelo"], strict=True)
    resumo["checkpoint_origem"] = config.checkpoint_origem
    resumo["epoca_origem"] = origem["epoca"]
    print(f"Fine-tuning: pesos carregados de {config.checkpoint_origem}; época {origem['epoca']}", flush=True)
    modelo.to(dispositivo)
    channels_last = config.channels_last and dispositivo.type == "cuda"
    if channels_last:
        modelo.to(memory_format=torch.channels_last)
    # O módulo original é usado no checkpoint; torch.compile fica apenas no executor.
    executor = torch.compile(modelo) if config.compilar_modelo else modelo
    otimizador = criar_otimizador(modelo, config)
    from registro_residuals import registrar_execucao
    registrar_execucao(config, "theo_ajuste", extras={"otimizador_opcoes_efetivas": otimizador.defaults})
    controle = ControleBackbone(modelo, config.passos_backbone_congelado,
                                config.congelar_backbone, config.epocas_backbone_congelado)
    registro_passos = otimizador.register_step_post_hook(controle.passo_concluido)
    if config.congelar_backbone:
        print("Backbone congelado durante todo o treinamento; apenas fc aprende.")
    elif config.epocas_backbone_congelado is not None:
        print(f"Apenas fc aprende nas primeiras {config.epocas_backbone_congelado} épocas; "
              f"backbone liberado no início da época {config.epocas_backbone_congelado + 1}.")
    else:
        print(f"Apenas fc aprende nas primeiras {config.passos_backbone_congelado} "
              "atualizações; depois, todas as camadas aprendem.")
    usar_amp = (config.precisao_mista and dispositivo.type == "cuda"
                and torch.amp.autocast_mode.is_autocast_available("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)
    print(f"AMP={usar_amp}; channels_last={channels_last}; compile={config.compilar_modelo}")
    historico = []
    assinatura_manifesto = hashlib.sha256(manifesto.read_bytes()).hexdigest()
    pasta_resultados = Path(config.checkpoint).parent
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    inicio_treino = time.perf_counter()
    total_batches = (
        config.epocas_iniciais_sem_ampliacao
        * (len(loader_treino_original) + len(loaders["validacao"]))
        + (config.epocas - config.epocas_iniciais_sem_ampliacao)
        * (len(loaders["treino"]) + len(loaders["validacao"])))
    progresso = ProgressoTreino(total_batches)
    print("ETA aproximada: recalculada a cada 5s e nas transições; pode variar ao liberar o backbone.", flush=True)
    for epoca in range(1, config.epocas + 1):
        learning_rate_efetivo = configurar_learning_rate_epoca(otimizador, config, epoca)
        print(f"Learning rate efetivo na época {epoca}: {learning_rate_efetivo:.10f}", flush=True)
        dropout_efetivo = probabilidade_dropout_epoca(config, epoca)
        modelo.dropout_comum.definir_probabilidade(dropout_efetivo)
        print(f"Dropout efetivo na época {epoca}: p={dropout_efetivo:.6f}", flush=True)
        inicio = time.perf_counter()
        metricas_treino, metricas_validacao = {}, {}
        fase_dados = ("ampliados_5_variantes" if usar_ampliacao_na_epoca(config, epoca)
                      else "originais")
        loader_treino = (loader_treino_original if fase_dados == "originais"
                         else loaders["treino"])
        print(f"Dados de treino na época {epoca}: {fase_dados}.", flush=True)
        treino = executar_epoca(executor, loader_treino, dispositivo,
                               otimizador=otimizador, scaler=scaler,
                               usar_amp=usar_amp, channels_last=channels_last,
                               controle_backbone=controle, metricas=metricas_treino,
                               progresso=progresso, epoca=epoca, epocas=config.epocas)
        validacao = executar_epoca(executor, loaders["validacao"], dispositivo,
                                  usar_amp=usar_amp, channels_last=channels_last,
                                  metricas=metricas_validacao, progresso=progresso,
                                  epoca=epoca, epocas=config.epocas)
        historico.append({"epoca": epoca, "bce_treino": treino, "bce_validacao": validacao,
                          "acuracia_treino": metricas_treino["acuracia"],
                          "acuracia_validacao": metricas_validacao["acuracia"],
                          "f1_treino": metricas_treino["f1"],
                          "f1_validacao": metricas_validacao["f1"],
                          "tempo_segundos": time.perf_counter() - inicio_treino,
                          "dados_treino": fase_dados,
                          "dropout_probabilidade_efetiva": dropout_efetivo,
                          "learning_rate_efetivo": learning_rate_efetivo,
                          "passos_otimizador": controle.passos_concluidos})
        print(f"Época {epoca}/{config.epocas}: BCE treino={treino:.6f}; "
              f"validação={validacao:.6f}; acerto treino={metricas_treino['acuracia']:.2%}; "
              f"acerto validação={metricas_validacao['acuracia']:.2%}; "
              f"F1 treino={metricas_treino['f1']:.4f}; F1 validação={metricas_validacao['f1']:.4f}; "
              f"{time.perf_counter()-inicio:.1f}s", flush=True)
        salvar_checkpoint(modelo, otimizador, scaler, config, epoca, historico,
                          resumo, metadados, assinatura_manifesto, controle.passos_concluidos)
        from checkpoint_validacao import preservar_melhor_checkpoint
        preservar_melhor_checkpoint(config.checkpoint, historico)
        salvar_historico_e_graficos(historico, pasta_resultados, graficos=True)
    registro_passos.remove()
    from registro_residuals import registrar_execucao
    registrar_execucao(config, "theo_ajuste", extras={"otimizador_opcoes_efetivas": otimizador.defaults,
                        "estado": "concluido", "ultima_epoca": config.epocas})
    print(f"Checkpoint da última época: {config.checkpoint}. Teste não foi avaliado.")


def avaliar_teste(checkpoint: Path, manifesto: Path) -> None:
    from .inferir_res_theo_especial import carregar_checkpoint
    dispositivo = selecionar_dispositivo()
    modelo, estado = carregar_checkpoint(checkpoint, dispositivo)
    config = ConfigTreino(**estado["config"])
    validar_config_dataset(config, manifesto)
    if hashlib.sha256(manifesto.read_bytes()).hexdigest() != estado["manifesto_sha256"]:
        raise ValueError("Manifesto diferente daquele usado no treinamento.")
    conjuntos = carregar_divisao(
        manifesto,
        ("treino", "validacao") if config.frac_teste == 0 else ("treino", "validacao", "teste"),
        aceitar_rotulos_duplicados=config.aceitar_rotulos_duplicados,
        usar_dados_ampliados=config.usar_dados_ampliados,
        esquema_ampliacao=config.esquema_ampliacao,
    )
    loader = criar_loader(conjuntos["teste"], config, dispositivo, False)
    print(f"BCE do teste final: {executar_epoca(modelo, loader, dispositivo):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resultados", type=Path, help="Pasta exclusiva desta execução")
    parser.add_argument("--avaliar-teste", type=Path)
    parser.add_argument("--manifesto", type=Path, default=Path(CONFIG.manifesto))
    args = parser.parse_args()
    if args.avaliar_teste:
        avaliar_teste(args.avaliar_teste, args.manifesto)
    else:
        CONFIG.manifesto = str(args.manifesto)
        CONFIG.validar()
        from registro_residuals import novo_diretorio, registrar_execucao
        pasta = args.resultados.resolve() if args.resultados else novo_diretorio(CONFIG, "theo_ajuste")
        pasta.mkdir(parents=True, exist_ok=True)
        CONFIG.checkpoint = str(pasta / "ultima_epoca.pt")
        if Path(CONFIG.checkpoint).exists() or (pasta / "historico.csv").exists():
            raise FileExistsError("Resultados existentes: escolha uma nova pasta para preservar a execução anterior.")
        registrar_execucao(CONFIG, "theo_ajuste")
        treinar(CONFIG)


if __name__ == "__main__":
    main()

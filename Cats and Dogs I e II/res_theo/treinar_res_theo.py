"""Treina com CONFIG; --avaliar-teste CHECKPOINT acessa o teste separadamente."""
import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config_res_theo import CONFIG, ConfigTreino
from .dados_res_theo import DatasetManifesto, ROTULOS, carregar_divisao
from .res_theo import criar_res_theo


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
    if metadados != esperado:
        raise ValueError("Dataset/configuração incompatíveis; confira frações, semente e esquema.")
    return metadados


def criar_loader(amostras, config: ConfigTreino, dispositivo: torch.device, treino: bool):
    return DataLoader(
        DatasetManifesto(amostras), batch_size=config.batch_size, shuffle=treino,
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
    otimizador = classe(modelo.parameters(), lr=config.learning_rate,
                       **config.argumentos_otimizador)
    print(f"Otimizador definido pelo participante: {config.otimizador}; "
          f"opções efetivas (incluindo defaults da biblioteca): {otimizador.defaults}")
    return otimizador


class ControleBackbone:
    """Contagem global de atualizações efetivas, atravessando as épocas."""
    def __init__(self, modelo, passos_congelado: int, congelar_sempre: bool = False):
        self.modelo = modelo  # Módulo original, inclusive se o executor for compilado.
        self.passos_congelado = passos_congelado
        self.congelar_sempre = congelar_sempre
        self.passos_concluidos = 0

    def preparar_batch(self) -> None:
        congelar = self.congelar_sempre or self.passos_concluidos < self.passos_congelado
        if self.modelo.backbone_congelado != congelar:
            self.modelo.congelar_backbone(congelar)
            if not congelar:
                print(f"Backbone liberado: próximo passo = {self.passos_concluidos + 1}; "
                      "fc e camadas anteriores passam a treinar.", flush=True)

    def passo_concluido(self, otimizador, args, kwargs) -> None:
        # Hook de optimizer.step: GradScaler não o chama quando pula uma atualização
        # por gradientes não finitos. Não precisa consultar a escala na CPU por batch.
        self.passos_concluidos += 1


def executar_epoca(modelo, loader, dispositivo, *, otimizador=None, scaler=None,
                   usar_amp=False, channels_last=False, controle_backbone=None) -> float:
    treino = otimizador is not None
    modelo.train(treino)
    criterio = nn.BCEWithLogitsLoss()
    soma_perdas = torch.zeros((), device=dispositivo, dtype=torch.float32)
    quantidade = 0
    with torch.set_grad_enabled(treino):
        for imagens, rotulos in loader:
            if treino and controle_backbone is not None:
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
    if quantidade == 0:
        raise ValueError("DataLoader vazio.")
    # Uma transferência escalar ao fim da época; sem loss.item() por batch.
    media = (soma_perdas / quantidade).item()
    if not torch.isfinite(torch.tensor(media)):
        raise FloatingPointError("BCE não finita; execução interrompida.")
    return media


def salvar_checkpoint(modelo, otimizador, scaler, config, epoca, historico,
                      resumo_pesos, metadados, manifesto_sha256, passos_otimizador):
    destino = Path(config.checkpoint)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".pt.tmp")
    torch.save({
        "nome_modelo": "res_theo", "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "scaler": scaler.state_dict(), "config": asdict(config), "epoca": epoca,
        "passos_otimizador": passos_otimizador,
        "backbone_congelado": modelo.backbone_congelado,
        "historico": historico, "rotulos": ROTULOS, "resumo_pesos": resumo_pesos,
        "dataset": metadados, "manifesto_sha256": manifesto_sha256,
        "torch_version": str(torch.__version__),
    }, temporario)
    temporario.replace(destino)


def treinar(config: ConfigTreino) -> None:
    config.validar()  # Falha antes de ler imagens ou baixar pesos se faltam escolhas.
    manifesto = Path(config.manifesto)
    metadados = validar_config_dataset(config, manifesto)
    conjuntos = carregar_divisao(manifesto)
    dispositivo = selecionar_dispositivo()
    preparar_execucao(config, dispositivo)
    print(f"Rótulos: {ROTULOS}; variantes por conjunto: "
          + str({k: len(v) for k, v in conjuntos.items()}))
    loaders = {split: criar_loader(conjuntos[split], config, dispositivo, split == "treino")
               for split in ("treino", "validacao")}
    modelo, resumo = criar_res_theo(
        config.congelar_backbone or config.passos_backbone_congelado > 0
    )
    modelo.to(dispositivo)
    channels_last = config.channels_last and dispositivo.type == "cuda"
    if channels_last:
        modelo.to(memory_format=torch.channels_last)
    # O módulo original é usado no checkpoint; torch.compile fica apenas no executor.
    executor = torch.compile(modelo) if config.compilar_modelo else modelo
    otimizador = criar_otimizador(modelo, config)
    controle = ControleBackbone(modelo, config.passos_backbone_congelado,
                                config.congelar_backbone)
    registro_passos = otimizador.register_step_post_hook(controle.passo_concluido)
    if config.congelar_backbone:
        print("Backbone congelado durante todo o treinamento; apenas fc aprende.")
    else:
        print(f"Apenas fc aprende nas primeiras {config.passos_backbone_congelado} "
              "atualizações; depois, todas as camadas aprendem.")
    usar_amp = (config.precisao_mista and dispositivo.type == "cuda"
                and torch.amp.autocast_mode.is_autocast_available("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)
    print(f"AMP={usar_amp}; channels_last={channels_last}; compile={config.compilar_modelo}")
    historico = []
    assinatura_manifesto = hashlib.sha256(manifesto.read_bytes()).hexdigest()
    for epoca in range(1, config.epocas + 1):
        inicio = time.perf_counter()
        treino = executar_epoca(executor, loaders["treino"], dispositivo,
                               otimizador=otimizador, scaler=scaler,
                               usar_amp=usar_amp, channels_last=channels_last,
                               controle_backbone=controle)
        validacao = executar_epoca(executor, loaders["validacao"], dispositivo,
                                  usar_amp=usar_amp, channels_last=channels_last)
        historico.append({"epoca": epoca, "bce_treino": treino, "bce_validacao": validacao,
                          "passos_otimizador": controle.passos_concluidos})
        print(f"Época {epoca}/{config.epocas}: BCE treino={treino:.6f}; "
              f"validação={validacao:.6f}; {time.perf_counter()-inicio:.1f}s", flush=True)
        salvar_checkpoint(modelo, otimizador, scaler, config, epoca, historico,
                          resumo, metadados, assinatura_manifesto, controle.passos_concluidos)
    registro_passos.remove()
    print(f"Checkpoint da última época: {config.checkpoint}. Teste não foi avaliado.")


def avaliar_teste(checkpoint: Path, manifesto: Path) -> None:
    from .inferir_res_theo import carregar_checkpoint
    dispositivo = selecionar_dispositivo()
    modelo, estado = carregar_checkpoint(checkpoint, dispositivo)
    config = ConfigTreino(**estado["config"])
    validar_config_dataset(config, manifesto)
    if hashlib.sha256(manifesto.read_bytes()).hexdigest() != estado["manifesto_sha256"]:
        raise ValueError("Manifesto diferente daquele usado no treinamento.")
    conjuntos = carregar_divisao(manifesto)
    loader = criar_loader(conjuntos["teste"], config, dispositivo, False)
    print(f"BCE do teste final: {executar_epoca(modelo, loader, dispositivo):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avaliar-teste", type=Path)
    parser.add_argument("--manifesto", type=Path, default=Path(CONFIG.manifesto))
    args = parser.parse_args()
    if args.avaliar_teste:
        avaliar_teste(args.avaliar_teste, args.manifesto)
    else:
        CONFIG.manifesto = str(args.manifesto)
        treinar(CONFIG)


if __name__ == "__main__":
    main()

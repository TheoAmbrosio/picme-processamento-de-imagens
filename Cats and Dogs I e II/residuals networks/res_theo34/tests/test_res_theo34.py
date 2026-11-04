"""Testes pequenos: dados artificiais; nenhum treino ou dataset real."""
import csv
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.models import ResNet34_Weights, resnet34

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from res_theo34.res_theo34 import ResTheo, conferir_transferencia
from res_theo34.dados_res_theo34 import DatasetManifesto, carregar_divisao
from criar_dataset_ampliado import aplicar_avaliacao, split_de, variantes


@pytest.fixture(scope="module", autouse=True)
def limitar_cpu():
    anterior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(anterior)


def test_shapes_probabilidades_backward_e_batch_um():
    modelo = ResTheo(resnet34(weights=None))
    assert not hasattr(modelo, "layer4")
    assert [len(modelo.layer1), len(modelo.layer2), len(modelo.layer3)] == [3, 4, 6]
    esperados = {"conv1": (64, 112, 112), "maxpool": (64, 56, 56),
                 "layer1": (64, 56, 56), "layer2": (128, 28, 28),
                 "layer3": (256, 14, 14), "flatten": (256,)}
    observados = {}
    hooks = [getattr(modelo, nome).register_forward_hook(
        lambda modulo, entrada, saida, nome=nome: observados.update({nome: tuple(saida.shape[1:])})
    ) for nome in esperados]
    for batch in (1, 2):
        modelo.train()
        modelo.zero_grad(set_to_none=True)
        x = torch.randn(batch, 3, 224, 224)
        logits = modelo(x)
        assert logits.shape == (batch, 1)
        assert observados == esperados
        perda = torch.nn.BCEWithLogitsLoss()(logits, torch.zeros(batch, 1))
        assert torch.isfinite(perda)
        perda.backward()
        assert modelo.conv1.weight.grad is not None
        assert torch.isfinite(modelo.fc.weight.grad).all()
        modelo.eval()
        p = modelo.predict_proba(x)
        assert p.shape == (batch, 1)
        assert ((0 <= p) & (p <= 1)).all()
    for hook in hooks:
        hook.remove()


def test_pesos_oficiais_e_buffers_preservados():
    origem = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    estado_antes = {k: v.clone() for k, v in origem.state_dict().items()
                    if not k.startswith(("layer4.", "fc."))}
    modelo = ResTheo(origem)
    conferir_transferencia(modelo, origem)
    for chave, valor in estado_antes.items():
        assert torch.equal(modelo.state_dict()[chave], valor), chave
    assert torch.isfinite(modelo.fc.bias).all()
    # Uma incompatibilidade inesperada deve falhar explicitamente.
    modelo.layer3[0].bn1.running_mean.add_(1)
    origem_independente = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    with pytest.raises(ValueError, match="Divergência"):
        conferir_transferencia(modelo, origem_independente)


def test_normal_apenas_nova_camada(monkeypatch):
    chamadas = []
    original = torch.nn.init.normal_
    def registrar(tensor, mean=0.0, std=1.0, **kwargs):
        chamadas.append((tensor, mean, std))
        return original(tensor, mean=mean, std=std, **kwargs)
    origem = resnet34(weights=None)
    monkeypatch.setattr(torch.nn.init, "normal_", registrar)
    modelo = ResTheo(origem)
    assert len(chamadas) == 2
    assert chamadas[0][0] is modelo.fc.weight
    assert chamadas[1][0] is modelo.fc.bias
    assert all((media, desvio) == (0.0, 1.0) for _, media, desvio in chamadas)


def test_congelamento_inclui_batchnorm():
    modelo = ResTheo(resnet34(weights=None), congelar_backbone=True)
    modelo.train()
    antes = modelo.bn1.running_mean.clone()
    modelo(torch.randn(1, 3, 224, 224)).sum().backward()
    assert torch.equal(antes, modelo.bn1.running_mean)
    assert modelo.fc.weight.grad is not None
    assert all(not p.requires_grad for n, p in modelo.named_parameters() if not n.startswith("fc."))
    modelo.congelar_backbone(False)
    modelo.train()
    assert all(p.requires_grad for p in modelo.parameters())
    assert modelo.bn1.training


def test_avaliacao_igual_preset_oficial():
    imagem = Image.frombytes("RGB", (337, 249),
                            bytes((i * 13) % 256 for i in range(337 * 249 * 3)))
    from res_theo34.dados_res_theo34 import normalizar_imagem
    assert torch.equal(normalizar_imagem(aplicar_avaliacao(imagem)),
                       ResNet34_Weights.IMAGENET1K_V1.transforms()(imagem))


def test_divisao_antes_ampliacao_e_rotulos(tmp_path):
    rows = []
    for split, classe, sha in (("treino", "Cat", "a" * 64),
                              ("validacao", "Dog", "b" * 64),
                              ("teste", "Cat", "c" * 64)):
        original = f"{split}.jpg"
        for nome, geo, espelho, rotacao in variantes(classe, original, split):
            caminho = f"PetImages/{classe}/{nome}"
            destino = tmp_path / caminho
            destino.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (224, 224)).save(destino)
            rows.append(dict(split=split, classe=classe, rotulo=int(classe == "Dog"),
                             caminho=caminho, arquivo_origem=original, sha256=sha,
                             geometria=geo, espelho=espelho, rotacao=rotacao))
    manifesto = tmp_path / "manifesto_variantes.csv"
    with manifesto.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    conjuntos = carregar_divisao(manifesto)
    assert [len(conjuntos[c]) for c in ("treino", "validacao", "teste")] == [16, 1, 1]
    x, y = DatasetManifesto(conjuntos["validacao"])[0]
    assert x.shape == (3, 224, 224) and y.tolist() == [1.0]
    assert y.dtype == torch.float32
    rows[-1]["sha256"] = rows[0]["sha256"]
    with manifesto.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="Vazamento"):
        carregar_divisao(manifesto)
    assert split_de("a" * 64, 42, 0.2, 0.1) == split_de("a" * 64, 42, 0.2, 0.1)


def test_perda_loop_cpu_com_batch_um():
    from res_theo34.treinar_res_theo34 import executar_epoca
    from torch.utils.data import DataLoader, TensorDataset
    modelo = torch.nn.Linear(3, 1)
    loader = DataLoader(TensorDataset(torch.randn(3, 3), torch.tensor([[0.], [1.], [0.]])),
                        batch_size=1)
    otimizador = torch.optim.SGD(modelo.parameters(), lr=0.001)  # apenas teste sintético
    antes = modelo.weight.detach().clone()
    perda = executar_epoca(modelo, loader, torch.device("cpu"), otimizador=otimizador)
    assert perda >= 0 and not torch.equal(antes, modelo.weight)
    assert executar_epoca(modelo, loader, torch.device("cpu")) >= 0


def test_backbone_preservado_por_n_passos_e_depois_liberado():
    from torch.utils.data import DataLoader, TensorDataset
    from res_theo34.config_res_theo34 import ConfigTreino
    from res_theo34.treinar_res_theo34 import ControleBackbone, criar_otimizador, executar_epoca

    modelo = ResTheo(resnet34(weights=None), congelar_backbone=True)
    config = ConfigTreino(otimizador="SGD", learning_rate=0.001,
                         argumentos_otimizador={"momentum": 0.9, "weight_decay": 0.01})
    otimizador = criar_otimizador(modelo, config)
    controle = ControleBackbone(modelo, passos_congelado=2)
    hook = otimizador.register_step_post_hook(controle.passo_concluido)
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 224, 224),
                                     torch.tensor([[0.], [1.]])), batch_size=2)
    antes = {k: v.clone() for k, v in modelo.state_dict().items() if not k.startswith("fc.")}
    fc_antes = modelo.fc.weight.detach().clone()
    # Duas chamadas de época com um batch cada: a contagem não reinicia por época.
    for passo in (1, 2):
        executar_epoca(modelo, loader, torch.device("cpu"), otimizador=otimizador,
                       controle_backbone=controle)
        assert controle.passos_concluidos == passo
        assert modelo.backbone_congelado
        assert all(torch.equal(v, modelo.state_dict()[k]) for k, v in antes.items())
    assert not torch.equal(fc_antes, modelo.fc.weight)
    executar_epoca(modelo, loader, torch.device("cpu"), controle_backbone=controle)
    assert controle.passos_concluidos == 2  # Validação não conta como passo.
    executar_epoca(modelo, loader, torch.device("cpu"), otimizador=otimizador,
                   controle_backbone=controle)
    assert controle.passos_concluidos == 3
    assert not modelo.backbone_congelado and modelo.bn1.training
    assert modelo.conv1.weight.grad is not None
    assert not torch.equal(antes["conv1.weight"], modelo.conv1.weight)
    assert not torch.equal(antes["bn1.running_mean"], modelo.bn1.running_mean)
    hook.remove()


@pytest.mark.parametrize("passos,permanente", [(0, False), (50, False), (50, True)])
def test_limite_50_zero_e_congelamento_permanente(passos, permanente):
    from res_theo34.treinar_res_theo34 import ControleBackbone
    modelo = ResTheo(resnet34(weights=None))
    controle = ControleBackbone(modelo, passos, permanente)
    controle.preparar_batch()
    assert modelo.backbone_congelado == (permanente or passos > 0)
    controle.passos_concluidos = 49
    controle.preparar_batch()
    assert modelo.backbone_congelado == (permanente or passos == 50)
    controle.passo_concluido(None, (), {})
    controle.preparar_batch()
    assert modelo.backbone_congelado == permanente


@pytest.mark.parametrize("split_alvo,quantidade", [("treino", 16), ("validacao", 1), ("teste", 1)])
def test_gerador_separa_antes_e_salva_rgb(tmp_path, split_alvo, quantidade):
    import hashlib
    from criar_dataset_ampliado import processar
    origem = tmp_path / "entrada.jpg"
    Image.new("RGB", (337, 249), (10, 20, 30)).save(origem)
    sha = hashlib.sha256(origem.read_bytes()).hexdigest()
    semente = next(i for i in range(1000) if split_de(sha, i, 0.2, 0.1) == split_alvo)
    saida = tmp_path / "saida"
    saida.mkdir()
    stat = origem.stat()
    linha = processar(("Dog", str(origem), origem.name, stat.st_size, stat.st_mtime_ns,
                       str(saida), False, 0.1, 0.2, semente))
    assert linha["status"] == "ok" and linha["split"] == split_alvo
    assert linha["rotulo"] == 1
    arquivos = list(saida.glob("*.jpg"))
    assert len(arquivos) == quantidade
    for arquivo in arquivos:
        with Image.open(arquivo) as im:
            assert im.mode == "RGB" and im.size == (224, 224)


def test_checkpoint_e_inferencia_sem_novo_download(tmp_path):
    from res_theo34.dados_res_theo34 import ROTULOS
    from res_theo34.inferir_res_theo34 import carregar_checkpoint, probabilidade_cachorro
    modelo = ResTheo(resnet34(weights=None)).eval()
    checkpoint = tmp_path / "res_theo34.pt"
    torch.save({"nome_modelo": "res_theo", "rotulos": ROTULOS,
                "dataset": {"lado": 224}, "modelo": modelo.state_dict()}, checkpoint)
    carregado, _ = carregar_checkpoint(checkpoint, torch.device("cpu"))
    for chave, valor in modelo.state_dict().items():
        assert torch.equal(valor, carregado.state_dict()[chave])
    imagem = tmp_path / "teste.png"
    Image.new("RGB", (80, 100), (90, 120, 20)).save(imagem)
    assert 0 <= probabilidade_cachorro(carregado, imagem) <= 1


def test_config_sem_escolhas_nao_inicia_treinamento():
    from res_theo34.config_res_theo34 import ConfigTreino
    with pytest.raises(ValueError, match="Defina em res_theo34/config_res_theo34"):
        ConfigTreino().validar()


def test_bloqueio_de_splits_diferentes_e_dataset_antigo(tmp_path, monkeypatch):
    import argparse
    import criar_dataset_ampliado as gerador
    nomes = ("SAIDA_BASE", "SAIDA", "MANIFESTO_ORIGINAIS", "MANIFESTO_VARIANTES",
             "LOG_ERROS", "RELATORIO")
    for nome in nomes:
        monkeypatch.setattr(gerador, nome, getattr(gerador, nome))
    args = argparse.Namespace(saida=tmp_path / "novo", frac_validacao=0.1,
                              frac_teste=0.2, semente=42, apenas_verificar=False)
    gerador.configurar_dataset(args)
    args.frac_teste = 0.3
    with pytest.raises(ValueError, match="diverge"):
        gerador.configurar_dataset(args)
    args.saida = tmp_path / "antigo"
    args.saida.mkdir()
    (args.saida / "manifesto_originais.csv").write_text("antigo", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset antigo"):
        gerador.configurar_dataset(args)


def test_gerador_manifesto_relatorio_e_retomada_artificiais(tmp_path, monkeypatch):
    import argparse
    import hashlib
    import criar_dataset_ampliado as gerador
    origem = tmp_path / "originais"
    hashes = []
    for i, classe in enumerate(("Cat", "Dog", "Cat")):
        pasta = origem / classe
        pasta.mkdir(parents=True, exist_ok=True)
        arquivo = pasta / f"{i}.jpg"
        Image.new("RGB", (30 + i, 40 + i), (i * 80, 90, 10)).save(arquivo)
        hashes.append(hashlib.sha256(arquivo.read_bytes()).hexdigest())
    semente = next(s for s in range(10000)
                   if len({split_de(h, s, 0.2, 0.2) for h in hashes}) == 3)
    args = argparse.Namespace(saida=tmp_path / "ampliado", frac_validacao=0.2,
                              frac_teste=0.2, semente=semente, apenas_verificar=False, workers=1)
    for nome in ("SAIDA_BASE", "SAIDA", "MANIFESTO_ORIGINAIS", "MANIFESTO_VARIANTES",
                 "LOG_ERROS", "RELATORIO"):
        monkeypatch.setattr(gerador, nome, getattr(gerador, nome))
    monkeypatch.setattr(gerador, "ORIGEM", origem)
    monkeypatch.setattr(gerador, "garantir_dataset", lambda: None)
    monkeypatch.setattr(gerador, "ler_argumentos", lambda: args)
    assert gerador.main() == 0
    conjuntos = carregar_divisao(args.saida / "manifesto_variantes.csv")
    assert [len(conjuntos[s]) for s in ("treino", "validacao", "teste")] == [16, 1, 1]
    arquivos = {p: p.stat().st_mtime_ns for p in args.saida.rglob("*.jpg")}
    assert gerador.main() == 0
    assert arquivos == {p: p.stat().st_mtime_ns for p in arquivos}


def test_divisao_balanceada_preserva_grupos_e_cotas():
    from collections import Counter
    from criar_dataset_ampliado import sortear_grupos_balanceados
    grupos = [[("Cat", "duplicado.jpg"), ("Dog", "duplicado.jpg")]]
    grupos += [[(classe, f"{i}.jpg")] for classe in ("Cat", "Dog") for i in range(99)]
    plano = sortear_grupos_balanceados(grupos, 42, 0.2)
    assert plano == sortear_grupos_balanceados(grupos, 42, 0.2)
    assert plano != sortear_grupos_balanceados(grupos, 43, 0.2)
    assert plano[("Cat", "duplicado.jpg")] == plano[("Dog", "duplicado.jpg")]
    assert Counter((c, s) for (c, _), s in plano.items()) == {
        ("Cat", "treino"): 80, ("Dog", "treino"): 80,
        ("Cat", "validacao"): 20, ("Dog", "validacao"): 20,
    }


def test_config_participante_sem_teste():
    from res_theo34.config_res_theo34 import ConfigTreino
    ConfigTreino(otimizador="Adam", learning_rate=0.001, batch_size=1000,
                 epocas=1000, frac_validacao=0.2, frac_teste=0.0, dropout_probabilidade=0).validar()


def test_metricas_acuracia_e_bce_com_ultimo_batch_menor():
    from torch.utils.data import DataLoader, TensorDataset
    from res_theo34.treinar_res_theo34 import executar_epoca
    x = torch.tensor([[-2.], [0.], [3.]])
    y = torch.tensor([[0.], [1.], [0.]])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    metricas = {}
    perda = executar_epoca(torch.nn.Identity(), loader, torch.device('cpu'), metricas=metricas)
    assert perda == pytest.approx(torch.nn.functional.binary_cross_entropy_with_logits(x, y).item())
    assert metricas['acuracia'] == pytest.approx(2 / 3)
    assert metricas['amostras'] == 3


def test_eta_intervalo_e_conclusao(monkeypatch, capsys):
    import res_theo34.treinar_res_theo34 as treino
    agora = [0.0]
    monkeypatch.setattr(treino.time, 'perf_counter', lambda: agora[0])
    progresso = treino.ProgressoTreino(4, intervalo=15)
    agora[0] = 10
    progresso.atualizar('treino', 1, 1, 1, 4, torch.device('cpu'))
    assert 'falta aproximadamente 00:00:30' in capsys.readouterr().out
    agora[0] = 11
    progresso.atualizar('treino', 1, 1, 2, 4, torch.device('cpu'))
    assert capsys.readouterr().out == ''
    agora[0] = 30
    progresso.atualizar('treino', 1, 1, 3, 4, torch.device('cpu'))
    assert 'falta aproximadamente 00:00:10' in capsys.readouterr().out
    agora[0] = 40
    progresso.atualizar('treino', 1, 1, 4, 4, torch.device('cpu'))
    saida = capsys.readouterr().out
    assert '100.00%' in saida and 'falta aproximadamente 00:00:00' in saida


def test_historico_e_graficos_sem_janela(tmp_path):
    from res_theo34.treinar_res_theo34 import salvar_historico_e_graficos
    historico = [dict(epoca=1, bce_treino=0.6, bce_validacao=0.7,
                      acuracia_treino=0.7, acuracia_validacao=0.6,
                      tempo_segundos=60, passos_otimizador=2)]
    pasta = tmp_path / 'resultados'
    salvar_historico_e_graficos(historico, pasta)
    assert not (pasta / 'graficos_treinamento.png').exists()
    salvar_historico_e_graficos(historico, pasta, graficos=True)
    with (pasta / 'historico.csv').open(newline='') as f:
        linhas = list(csv.DictReader(f))
    assert len(linhas) == 1 and float(linhas[0]['acuracia_treino']) == 0.7
    with Image.open(pasta / 'graficos_treinamento.png') as imagem:
        assert imagem.width > 0 and imagem.height > 0


def test_backbone_congelado_dez_epocas_independente_dos_passos():
    from res_theo34.treinar_res_theo34 import ControleBackbone
    modelo = ResTheo(resnet34(weights=None))
    controle = ControleBackbone(modelo, 50, epocas_congelado=10)
    for epoca in range(1, 11):
        controle.epoca = epoca
        controle.passos_concluidos = epoca * 1000
        modelo.train()
        controle.preparar_batch()
        assert modelo.backbone_congelado
        assert not modelo.bn1.training
        assert modelo.fc.weight.requires_grad
        assert not modelo.conv1.weight.requires_grad
    controle.epoca = 11
    controle.preparar_batch()
    assert not modelo.backbone_congelado
    assert modelo.conv1.weight.requires_grad and modelo.bn1.training
    permanente = ControleBackbone(modelo, 50, True, epocas_congelado=10)
    permanente.epoca = 11
    permanente.preparar_batch()
    assert modelo.backbone_congelado

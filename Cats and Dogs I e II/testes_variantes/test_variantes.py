import importlib
import sys
from pathlib import Path

import pytest
import torch
from torchvision.models import resnet18, resnet34

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residuals networks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residuals networks" / "arquivados"))


@pytest.fixture(scope='module', autouse=True)
def threads():
    anterior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(anterior)


@pytest.mark.parametrize('nome,factory,blocos', [
    ('res_theo18', resnet18, [2,2,2]),
    ('res_theo34', resnet34, [3,4,6]),
])
def test_variante_transferencia_dropout_e_checkpoint(tmp_path, nome, factory, blocos):
    modulo = importlib.import_module(f'{nome}.{nome}')
    cfg = importlib.import_module(f'{nome}.config_{nome}')
    dropout = importlib.import_module(f'{nome}.dropout_comum')
    inferir = importlib.import_module(f'{nome}.inferir_{nome}')
    cfg.CONFIG.validar()
    origem = factory(weights=None)
    modelo = modulo.ResTheo(origem)
    assert not hasattr(modelo, 'layer4')
    assert [len(modelo.layer1),len(modelo.layer2),len(modelo.layer3)] == blocos
    modulo.conferir_transferencia(modelo, origem)
    # Configuração exclusivamente sintética de teste; não preenche os campos do participante.
    config = cfg.ConfigTreino(otimizador='Adam',learning_rate=0.001,batch_size=2,epocas=1,
                             frac_validacao=0.2,frac_teste=0,
                             dropout_posicoes=('conv1','layer1.0.conv1'), dropout_probabilidade=0.3)
    config.validar()
    x = torch.randn(2,3,224,224)
    modelo.eval()
    with torch.no_grad():
        esperado = modelo(x)
    antes = {k:v.clone() for k,v in modelo.state_dict().items()}
    controle = dropout.instalar_dropout(modelo, config)
    assert all(torch.equal(antes[k],v) for k,v in modelo.state_dict().items())
    modelo.eval()
    with torch.no_grad():
        assert torch.equal(modelo(x), esperado)
        assert torch.equal(modelo(x), modelo(x))
    modelo.train()
    perda = modelo(x).sum()
    perda.backward()
    assert controle.posicoes == config.dropout_posicoes
    assert modelo.fc.weight.grad is not None
    modulo.conferir_transferencia(modelo, origem)
    modelo.eval()
    caminho = tmp_path / 'checkpoint.pt'
    from dataclasses import asdict
    torch.save(dict(nome_modelo=nome,rotulos={'Cat':0,'Dog':1},dataset={'lado':224},
                    config=asdict(config),modelo=modelo.state_dict(),dropout=controle.estado()),caminho)
    carregado,_=inferir.carregar_checkpoint(caminho,torch.device('cpu'))
    with torch.no_grad():
        assert torch.equal(modelo(x),carregado(x))
    controle.train()
    z=controle.camadas[0](torch.ones(10000))
    assert 0.27 < (z==0).float().mean().item() < 0.33


@pytest.mark.parametrize('nome', ['res_theo18','res_theo34'])
def test_rejeita_bloco_residual_e_factory_sem_decisoes(nome, monkeypatch):
    modelo_mod = importlib.import_module(f'{nome}.{nome}')
    cfg=importlib.import_module(f'{nome}.config_{nome}')
    dropout=importlib.import_module(f'{nome}.dropout_comum')
    factory=resnet18 if nome=='res_theo18' else resnet34
    modelo=modelo_mod.ResTheo(factory(weights=None))
    config=cfg.ConfigTreino(dropout_posicoes=('layer1.0',))
    with pytest.raises(ValueError,match='residual'):
        dropout.instalar_dropout(modelo,config)
    def nao_baixar(**kwargs):
        raise AssertionError('Não deveria baixar pesos antes de validar dropout.')
    monkeypatch.setattr(modelo_mod, 'resnet18' if nome=='res_theo18' else 'resnet34', nao_baixar)
    with pytest.raises(ValueError, match='ainda não liberada'):
        getattr(modelo_mod,f'criar_{nome}')(config=cfg.ConfigTreino())


def test_dropout_comum_mascaras_variaveis_e_avaliacao():
    from res_theo18.dropout_comum import DropoutComum
    controle = DropoutComum(('ponto_teste',))
    x = torch.ones(10000)
    controle.train()
    assert not torch.equal(controle.camadas[0](x), controle.camadas[0](x))
    controle.eval()
    assert torch.equal(controle.camadas[0](x), x)
    assert controle.estado() == {"tipo": "comum", "posicoes": ('ponto_teste',),
                                 "probabilidade": 0.3, "probabilidade_maxima": 0.3}


@pytest.mark.parametrize('nome,ativado', [
    ('res_theo18', False), ('res_theo34', False),
    ('res_theo_especial', True), ('res_theo_especial34', True),
    ('res_theo_especial50', True),
    ('alex_theo', True),
])
def test_dropout_crescente_linear_e_configuracao(nome, ativado):
    from dataclasses import replace
    cfg = importlib.import_module(f'{nome}.config_{nome}')
    treino = importlib.import_module(f'{nome}.treinar_{nome}')
    assert cfg.CONFIG.dropout_crescente is ativado
    assert treino.probabilidade_dropout_epoca(cfg.CONFIG, 1) == (0 if ativado else pytest.approx(0.1))
    alteracoes = dict(dropout_crescente=True, dropout_probabilidade=0.1,
                      dropout_epoca_inicio=20, dropout_epoca_maximo=40)
    if hasattr(cfg.CONFIG, 'dropout_epoca_inicio_reducao'):
        alteracoes.update(dropout_epoca_inicio_reducao=None,
                          dropout_probabilidade_final=None)
    config = replace(cfg.CONFIG, **alteracoes)
    assert treino.probabilidade_dropout_epoca(config, 1) == 0
    assert treino.probabilidade_dropout_epoca(config, 20) == 0
    assert treino.probabilidade_dropout_epoca(config, 30) == pytest.approx(0.05)
    assert treino.probabilidade_dropout_epoca(config, 40) == pytest.approx(0.1)
    assert treino.probabilidade_dropout_epoca(config, 50) == pytest.approx(0.1)


@pytest.mark.parametrize('nome,factory', [('res_theo18',resnet18),('res_theo34',resnet34)])
def test_todos_pontos_forward_backward_e_avaliacao(nome,factory):
    modulo=importlib.import_module(f'{nome}.{nome}')
    cfg=importlib.import_module(f'{nome}.config_{nome}')
    dropout=importlib.import_module(f'{nome}.dropout_comum')
    cfg.CONFIG.validar()
    modelo=modulo.ResTheo(factory(weights=None))
    tipos=(torch.nn.Conv2d,torch.nn.BatchNorm2d,torch.nn.ReLU,torch.nn.MaxPool2d,
           torch.nn.AdaptiveAvgPool2d,torch.nn.Flatten,torch.nn.Linear)
    esperado=tuple(n for n,m in modelo.named_modules() if isinstance(m,tipos) and n not in ("conv1", "fc"))
    assert cfg.CONFIG.dropout_posicoes == esperado
    assert cfg.CONFIG.dropout_probabilidade == 0.1
    assert "conv1" not in esperado and "fc" not in esperado
    modelo.eval()
    x=torch.randn(2,3,224,224)
    with torch.no_grad(): referencia=modelo(x)
    controle=dropout.instalar_dropout(modelo,cfg.CONFIG)
    modelo.eval()
    with torch.no_grad(): assert torch.equal(modelo(x),referencia)
    modelo.train()
    perda=torch.nn.functional.binary_cross_entropy_with_logits(modelo(x),torch.zeros(2,1))
    perda.backward()
    assert torch.isfinite(perda) and torch.isfinite(modelo.conv1.weight.grad).all()
    assert len(controle.handles)==len(esperado)


@pytest.mark.parametrize('nome', ['res_theo18','res_theo34'])
def test_drop_zero_e_caminhos_externos(nome):
    from dataclasses import replace
    cfg=importlib.import_module(f'{nome}.config_{nome}')
    config=replace(cfg.CONFIG, dropout_probabilidade=0, dropout_posicoes=None)
    config.validar()
    assert Path(config.manifesto).exists()
    assert 'resultados_residuals_networks' in config.checkpoint
    assert 'residuals networks' not in str(Path(config.checkpoint).parent)
    assert config.usar_dados_ampliados is False


@pytest.mark.parametrize('nome', ['res_theo18', 'res_theo34'])
def test_configuracao_do_zero_he_normal_fan_out(nome):
    cfg = importlib.import_module(f'{nome}.config_{nome}')
    modulo = importlib.import_module(f'{nome}.{nome}')
    cfg.CONFIG.validar()
    assert cfg.CONFIG.usar_pesos_pre_treinados is False
    assert cfg.CONFIG.epocas_backbone_congelado == 0
    modelo, resumo = getattr(modulo, f'criar_{nome}')(False, cfg.CONFIG)
    assert resumo['pesos'] is None and resumo['inicializacao'] == 'he_normal_fan_out'
    assert abs(modelo.conv1.weight.mean().item()) < 0.05
    _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(modelo.conv1.weight)
    desvio_esperado = (2 / fan_out) ** 0.5
    assert 0.95 < modelo.conv1.weight.std().item() / desvio_esperado < 1.05
    assert torch.equal(modelo.bn1.weight, torch.ones_like(modelo.bn1.weight))
    assert torch.equal(modelo.bn1.bias, torch.zeros_like(modelo.bn1.bias))
    assert modelo.backbone_congelado is False


def test_res_theo_especial_canais_inicializacao_dropout_e_forward():
    from res_theo_especial.config_res_theo_especial import CONFIG
    from res_theo_especial.res_theo_especial import criar_res_theo_especial
    from res_theo_especial.treinar_res_theo_especial import (
        configurar_learning_rate_epoca, criar_otimizador,
        probabilidade_dropout_epoca, taxa_aprendizado_epoca,
        usar_ampliacao_na_epoca)
    CONFIG.validar()
    modelo, resumo = criar_res_theo_especial(False, CONFIG)
    assert resumo['canais'] == [32, 32, 64, 128]
    assert modelo.conv1.out_channels == 32
    assert modelo.layer1[0].conv1.out_channels == 32
    assert modelo.layer2[0].conv1.out_channels == 64
    assert modelo.layer3[0].conv1.out_channels == 128
    assert not hasattr(modelo, 'layer4')
    assert modelo.fc.in_features == 128 and modelo.fc.out_features == 1
    assert modelo.dropout_comum.probabilidade == 0.2
    assert CONFIG.usar_pesos_pre_treinados is False
    assert CONFIG.epocas == 110 and CONFIG.dropout_crescente is True
    assert CONFIG.otimizador == "AdamW"
    assert CONFIG.argumentos_otimizador == {"weight_decay": 0.01}
    otimizador = criar_otimizador(modelo, CONFIG)
    assert isinstance(otimizador, torch.optim.AdamW)
    assert otimizador.defaults["lr"] == pytest.approx(0.000001)
    assert otimizador.defaults["weight_decay"] == pytest.approx(0.01)
    assert usar_ampliacao_na_epoca(CONFIG, 1) is False
    assert usar_ampliacao_na_epoca(CONFIG, 20) is False
    assert usar_ampliacao_na_epoca(CONFIG, 21) is True
    assert usar_ampliacao_na_epoca(CONFIG, 110) is True
    assert taxa_aprendizado_epoca(CONFIG, 1) == pytest.approx(0.000001)
    assert taxa_aprendizado_epoca(CONFIG, 30) == pytest.approx(0.001)
    assert taxa_aprendizado_epoca(CONFIG, 31) == pytest.approx(0.001)
    meio = taxa_aprendizado_epoca(CONFIG, 15)
    assert 0.000001 < meio < 0.001
    assert configurar_learning_rate_epoca(otimizador, CONFIG, 15) == pytest.approx(meio)
    assert otimizador.param_groups[0]["lr"] == pytest.approx(meio)
    assert CONFIG.dropout_epoca_inicio_reducao == 80
    assert CONFIG.dropout_probabilidade_final == pytest.approx(0.05)
    assert probabilidade_dropout_epoca(CONFIG, 50) == pytest.approx(0.2)
    assert probabilidade_dropout_epoca(CONFIG, 80) == pytest.approx(0.2)
    assert probabilidade_dropout_epoca(CONFIG, 90) == pytest.approx(0.125)
    assert probabilidade_dropout_epoca(CONFIG, 100) == pytest.approx(0.05)
    assert probabilidade_dropout_epoca(CONFIG, 110) == pytest.approx(0.05)
    modelo.eval()
    with torch.no_grad():
        assert modelo(torch.randn(2, 3, 224, 224)).shape == (2, 1)


def test_res_theo_especial_cinco_variantes_deterministicas(tmp_path):
    from PIL import Image
    from res_theo_especial.config_res_theo_especial import CONFIG
    from res_theo_especial.dados_res_theo_especial import DatasetManifesto, VARIANTES_ESPECIAIS

    assert CONFIG.ruido_gaussiano_variancia == pytest.approx(0.1)
    assert CONFIG.cutout_retangulos == 8
    assert (CONFIG.cutout_lado_minimo, CONFIG.cutout_lado_maximo) == (20, 40)

    pixels = torch.zeros(3, 60, 80, dtype=torch.uint8)
    pixels[0] = torch.arange(80, dtype=torch.uint8).repeat(60, 1)
    pixels[1] = torch.arange(60, dtype=torch.uint8).repeat(80, 1).T
    pixels[2] = 127
    caminho = tmp_path / "assimetrica.png"
    Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(caminho)
    amostras = [(caminho, 0, variante) for variante in VARIANTES_ESPECIAIS]
    dataset = DatasetManifesto(
        amostras, esquema_ampliacao=CONFIG.esquema_ampliacao,
        semente=CONFIG.semente,
        ruido_gaussiano_variancia=CONFIG.ruido_gaussiano_variancia,
        pixels_brancos_fracao=CONFIG.pixels_brancos_fracao,
        cutout_retangulos=CONFIG.cutout_retangulos,
        cutout_lado_minimo=CONFIG.cutout_lado_minimo,
        cutout_lado_maximo=CONFIG.cutout_lado_maximo)
    saidas = [dataset[i][0] for i in range(5)]
    assert all(x.shape == (3, 224, 224) and torch.isfinite(x).all() for x in saidas)
    assert torch.equal(saidas[3], dataset[3][0])
    assert torch.equal(saidas[4], dataset[4][0])
    media = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    desvio = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    cinza = saidas[2] * desvio + media
    assert torch.allclose(cinza[0], cinza[1], atol=1e-6)
    assert torch.allclose(cinza[1], cinza[2], atol=1e-6)
    assert not torch.equal(saidas[0], saidas[1])
    assert not torch.equal(saidas[0], saidas[3])
    assert not torch.equal(saidas[0], saidas[4])


@pytest.mark.parametrize('nome,blocos,fc,canais', [
    ('res_theo_especial34', [3, 4, 6, 3], 256, [32, 64, 128, 256]),
    ('res_theo_especial50', [3, 4, 6, 3], 512, [16, 32, 64, 128]),
])
def test_especiais_34_50_arquitetura_dropout_e_conv1(nome, blocos, fc, canais, monkeypatch):
    from types import SimpleNamespace
    cfg = importlib.import_module(f'{nome}.config_{nome}')
    modulo = importlib.import_module(f'{nome}.{nome}')
    treino = importlib.import_module(f'{nome}.treinar_{nome}')
    cfg.CONFIG.validar()
    assert cfg.CONFIG.dropout_crescente is True
    assert cfg.CONFIG.usar_pesos_pre_treinados_conv1 is True
    assert cfg.CONFIG.epocas_conv1_congelada == 5
    referencia = torch.full((64, 3, 7, 7), 0.012345)
    monkeypatch.setattr(modulo, 'resnet34' if nome.endswith('34') else 'resnet50',
                        lambda weights: SimpleNamespace(conv1=SimpleNamespace(weight=referencia)))
    modelo, resumo = getattr(modulo, f'criar_{nome}')(False, cfg.CONFIG)
    assert [len(modelo.layer1), len(modelo.layer2), len(modelo.layer3), len(modelo.layer4)] == blocos
    assert modelo.conv1.out_channels == 64
    assert [modelo.layer1[0].conv1.out_channels, modelo.layer2[0].conv1.out_channels,
            modelo.layer3[0].conv1.out_channels, modelo.layer4[0].conv1.out_channels] == canais
    assert modelo.fc.in_features == fc
    assert torch.equal(modelo.conv1.weight, referencia)
    assert resumo['conv1_pesos'] is not None
    assert modelo.dropout_comum.probabilidade_maxima == 0.1
    treino.configurar_conv1_epoca(modelo, cfg.CONFIG, 1)
    assert modelo.conv1.weight.requires_grad is False
    treino.configurar_conv1_epoca(modelo, cfg.CONFIG, 5)
    assert modelo.conv1.weight.requires_grad is False
    treino.configurar_conv1_epoca(modelo, cfg.CONFIG, 6)
    assert modelo.conv1.weight.requires_grad is True
    modelo.eval()
    with torch.no_grad():
        assert modelo(torch.randn(2, 3, 224, 224)).shape == (2, 1)


def test_alex_theo_preserva_estrutura_e_troca_somente_saida():
    from torchvision.models import alexnet
    from alex_theo.alex_theo import AlexTheo
    from alex_theo.config_alex_theo import CONFIG
    from alex_theo.treinar_alex_theo import configurar_conv1_epoca
    CONFIG.validar()
    origem = alexnet(weights=None)
    preservado = {k: v.clone() for k, v in origem.state_dict().items()
                  if not k.startswith('classifier.6.')}
    modelo = AlexTheo(origem, dropout_probabilidade=CONFIG.dropout_probabilidade)
    assert modelo.classifier[6].in_features == 4096
    assert modelo.classifier[6].out_features == 1
    assert all(torch.equal(modelo.state_dict()[k], v) for k, v in preservado.items())
    assert modelo.conv1.kernel_size == (11, 11)
    assert modelo.dropout_comum.posicoes == ('classifier.0', 'classifier.3')
    modelo.dropout_comum.definir_probabilidade(0.05)
    assert modelo.classifier[0].p == modelo.classifier[3].p == 0.05
    configurar_conv1_epoca(modelo, CONFIG, 1)
    assert modelo.conv1.weight.requires_grad is False
    configurar_conv1_epoca(modelo, CONFIG, 6)
    assert modelo.conv1.weight.requires_grad is True
    modelo.eval()
    with torch.no_grad():
        assert modelo(torch.randn(2, 3, 224, 224)).shape == (2, 1)


def test_registro_execucao_preserva_config_e_fontes(tmp_path, monkeypatch):
    from dataclasses import replace
    import json
    import registro_residuals as registro
    from res_theo18.config_res_theo18 import CONFIG
    monkeypatch.setattr(registro, 'RESULTADOS', tmp_path)
    a=registro.novo_diretorio(CONFIG, 'res_theo18')
    b=registro.novo_diretorio(CONFIG, 'res_theo18')
    assert a!=b and f'bs{CONFIG.batch_size}_ep{CONFIG.epocas}_cong{CONFIG.epocas_backbone_congelado}ep_do{int(CONFIG.dropout_probabilidade*100)}' in a.name
    config=replace(CONFIG,checkpoint=str(a/'ultima_epoca.pt'))
    registro.registrar_execucao(config,'res_theo18')
    dados=json.loads((a/'parametros.json').read_text())
    assert dados['config']['batch_size']==CONFIG.batch_size
    assert dados['config']['dropout_probabilidade']==CONFIG.dropout_probabilidade
    assert dados['config']['usar_pesos_pre_treinados'] is False
    assert (a/'codigo/res_theo18.py').exists()
    assert (a/'IDENTIFICACAO.md').exists()


@pytest.mark.parametrize('nome,factory',[('res_theo18',resnet18),('res_theo34',resnet34)])
def test_treino_artificial_registra_parametros_e_resultados(tmp_path,monkeypatch,nome,factory):
    from dataclasses import replace
    from torch.utils.data import DataLoader,TensorDataset
    import json
    treino=importlib.import_module(f'{nome}.treinar_{nome}')
    modulo=importlib.import_module(f'{nome}.{nome}')
    dropout=importlib.import_module(f'{nome}.dropout_comum')
    cfg=importlib.import_module(f'{nome}.config_{nome}')
    manifesto=tmp_path/'manifesto.csv';manifesto.write_text('artificial')
    config=replace(cfg.CONFIG,epocas=1,batch_size=2,num_workers=0,
                   checkpoint=str(tmp_path/'execucao/ultima_epoca.pt'),manifesto=str(manifesto))
    monkeypatch.setattr(treino,'validar_config_dataset',lambda *a:{'lado':224})
    monkeypatch.setattr(treino,'carregar_divisao',lambda *a,**kw:{'treino':[0,1],'validacao':[0,1],'teste':[]})
    monkeypatch.setattr(treino,'selecionar_dispositivo',lambda:torch.device('cpu'))
    loader=DataLoader(TensorDataset(torch.randn(2,3,224,224),torch.tensor([[0.],[1.]])),batch_size=2)
    monkeypatch.setattr(treino,'criar_loader',lambda *a,**kw:loader)
    def criar(congelar,config):
        modelo=modulo.ResTheo(factory(weights=None),congelar)
        dropout.instalar_dropout(modelo,config)
        return modelo,{}
    monkeypatch.setattr(treino,f'criar_{nome}',criar)
    treino.treinar(config)
    pasta=Path(config.checkpoint).parent
    assert (pasta/'historico.csv').exists() and (pasta/'graficos_treinamento.png').exists()
    dados=json.loads((pasta/'parametros.json').read_text())
    assert dados['modelo']==nome and dados['estado']=='concluido'
    assert dados['config']['epocas']==1
    assert torch.load(config.checkpoint,weights_only=True)['nome_modelo']==nome

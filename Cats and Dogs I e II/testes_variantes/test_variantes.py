import importlib
import sys
from pathlib import Path

import pytest
import torch
from torchvision.models import resnet18, resnet34

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residuals networks"))


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
                             dropout_posicoes=('conv1','layer1.0.conv1'))
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
    assert controle.estado() == {"tipo": "comum", "posicoes": ('ponto_teste',), "probabilidade": 0.3}


@pytest.mark.parametrize('nome,factory', [('res_theo18',resnet18),('res_theo34',resnet34)])
def test_todos_pontos_forward_backward_e_avaliacao(nome,factory):
    modulo=importlib.import_module(f'{nome}.{nome}')
    cfg=importlib.import_module(f'{nome}.config_{nome}')
    dropout=importlib.import_module(f'{nome}.dropout_comum')
    cfg.CONFIG.validar()
    modelo=modulo.ResTheo(factory(weights=None))
    tipos=(torch.nn.Conv2d,torch.nn.BatchNorm2d,torch.nn.ReLU,torch.nn.MaxPool2d,
           torch.nn.AdaptiveAvgPool2d,torch.nn.Flatten,torch.nn.Linear)
    esperado=tuple(n for n,m in modelo.named_modules() if isinstance(m,tipos))
    assert cfg.CONFIG.dropout_posicoes == esperado
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


def test_registro_execucao_preserva_config_e_fontes(tmp_path, monkeypatch):
    from dataclasses import replace
    import json
    import registro_residuals as registro
    from res_theo18.config_res_theo18 import CONFIG
    monkeypatch.setattr(registro, 'RESULTADOS', tmp_path)
    a=registro.novo_diretorio(CONFIG, 'res_theo18')
    b=registro.novo_diretorio(CONFIG, 'res_theo18')
    assert a!=b and 'bs128_ep25_cong10ep_do30' in a.name
    config=replace(CONFIG,checkpoint=str(a/'ultima_epoca.pt'))
    registro.registrar_execucao(config,'res_theo18')
    dados=json.loads((a/'parametros.json').read_text())
    assert dados['config']['batch_size']==128
    assert dados['config']['dropout_probabilidade']==0.3
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

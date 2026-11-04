import importlib
import pytest
import torch


@pytest.mark.parametrize('nome', ['theo_curto','theo_48_88','theo_28_42_64'])
def test_agenda_e_adamw(nome):
    config = importlib.import_module(f'{nome}.config_{nome}').CONFIG
    treino = importlib.import_module(f'{nome}.treinar_{nome}')
    config.validar()
    assert config.epocas == 80
    assert config.epocas_iniciais_sem_ampliacao == 50
    assert config.checkpoint_origem == ''
    assert config.ruido_gaussiano_variancia == .08
    assert [treino.usar_ampliacao_na_epoca(config,e) for e in (1,50,51,80)] == [False,False,True,True]
    assert [treino.taxa_aprendizado_epoca(config,e) for e in (1,50,51,75,80)] == pytest.approx([.001,.001,.000001,.001,.001])
    assert [treino.probabilidade_dropout_epoca(config,e) for e in (1,25,50,75,80)] == pytest.approx([0,.2,.1,.05,.05])
    modelo = torch.nn.Linear(1,1)
    opt = treino.criar_otimizador(modelo, config)
    assert isinstance(opt,torch.optim.AdamW)
    assert opt.param_groups[0]['lr'] == .001
    assert opt.param_groups[0]['weight_decay'] == .01
    identidade = id(opt)
    treino.configurar_learning_rate_epoca(opt,config,51)
    assert id(opt) == identidade
    assert opt.param_groups[0]['lr'] == pytest.approx(.000001)


def test_theo_28_42_64_canais_e_gradientes():
    from theo_28_42_64.theo_28_42_64 import criar_modelo
    modelo = criar_modelo()
    assert modelo.conv1.out_channels == modelo.bn1.num_features == 28
    assert [modelo.layer1[0].conv1.out_channels, modelo.layer2[0].conv1.out_channels,
            modelo.layer3[0].conv1.out_channels] == [28,42,64]
    assert modelo.layer2[0].downsample[0].in_channels == 28
    assert modelo.layer3[0].downsample[0].in_channels == 42
    assert modelo.fc.in_features == 64
    assert not hasattr(modelo,'layer4')
    saida = modelo(torch.randn(2,3,224,224))
    assert saida.shape == (2,1)
    saida.sum().backward()
    assert torch.isfinite(modelo.conv1.weight.grad).all()

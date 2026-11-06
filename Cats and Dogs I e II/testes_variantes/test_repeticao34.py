import torch
from torchvision.models import resnet34


def test_repeticao_inicializacao_e_congelamento():
    from res_theo34_repeticao.res_theo34 import ResTheo
    from res_theo34_repeticao.config_res_theo34 import CONFIG
    from res_theo34_repeticao.treinar_res_theo34 import ControleBackbone
    CONFIG.validar()
    origem = resnet34(weights=None)
    torch.manual_seed(42)
    modelo = ResTheo(origem, True)
    assert modelo.conv1 is origem.conv1
    assert not hasattr(modelo,'layer4')
    assert .8 < modelo.fc.weight.std().item() < 1.2
    assert modelo.fc.bias.item() != 0
    controle = ControleBackbone(modelo,50,False,5)
    controle.epoca=5;controle.preparar_batch()
    assert modelo.conv1.weight.requires_grad is False
    controle.epoca=6;controle.preparar_batch()
    assert modelo.conv1.weight.requires_grad is True
    assert CONFIG.usar_dados_ampliados and CONFIG.usar_pesos_pre_treinados
    assert CONFIG.dropout_probabilidade == 0

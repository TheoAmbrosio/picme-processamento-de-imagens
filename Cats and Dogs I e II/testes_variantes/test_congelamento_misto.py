import torch
from torch import nn
from res_theo34_misto.treinar_res_theo34 import aplicar_congelamento_final


def test_congelamento_progressivo():
    modelo = nn.Module()
    for nome in ('conv1','bn1','layer1','layer2','layer3','fc'):
        setattr(modelo,nome,nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2)))
    for epoca, congelados in ((70,()),(71,('conv1','bn1','layer1')),
            (81,('conv1','bn1','layer1','layer2')),
            (91,('conv1','bn1','layer1','layer2','layer3'))):
        modelo.train()
        assert aplicar_congelamento_final(modelo,epoca)==congelados
        for nome,modulo in modelo.named_children():
            assert all(p.requires_grad == (nome not in congelados) for p in modulo.parameters())
            assert modulo.training == (nome not in congelados)

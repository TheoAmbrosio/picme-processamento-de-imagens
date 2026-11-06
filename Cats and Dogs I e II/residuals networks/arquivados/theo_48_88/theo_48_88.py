"""BasicBlocks [2,2,2], canais 32/32/48/88, layer4 removida."""
from torch import nn
from res_theo_especial.res_theo_especial import ResTheo, inicializar_modelo_do_zero
from res_theo_especial.dropout_comum import instalar_dropout
from .config_theo_48_88 import CONFIG


class Theo4888(ResTheo):
    def __init__(self):
        super().__init__(False)
        self.inplanes = 32
        self.layer2 = self._make_layer(48, 2, stride=2)
        self.layer3 = self._make_layer(88, 2, stride=2)
        self.fc = nn.Linear(88, 1)
        inicializar_modelo_do_zero(self)


def criar_modelo(config=CONFIG):
    modelo = Theo4888()
    instalar_dropout(modelo, config)
    return modelo


def criar_para_treino(congelar_backbone=False, config=CONFIG):
    modelo = criar_modelo(config)
    modelo.congelar_backbone(congelar_backbone)
    return modelo, {"pesos": None, "inicializacao": "he_normal_fan_out",
                    "canais": [32, 32, 48, 88]}

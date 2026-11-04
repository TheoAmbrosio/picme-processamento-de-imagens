"""BasicBlocks [2,2,2], canais 28/28/42/64, layer4 removida."""
from torch import nn
from res_theo_especial.res_theo_especial import ResTheo, inicializar_modelo_do_zero
from res_theo_especial.dropout_comum import instalar_dropout
from .config_theo_28_42_64 import CONFIG


class Theo284264(ResTheo):
    def __init__(self):
        super().__init__(False)
        self.conv1 = nn.Conv2d(3, 28, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(28)
        self.inplanes = 28
        self.layer1 = self._make_layer(28, 2)
        self.layer2 = self._make_layer(42, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.fc = nn.Linear(64, 1)
        inicializar_modelo_do_zero(self)


def criar_modelo(config=CONFIG):
    modelo = Theo284264()
    instalar_dropout(modelo, config)
    return modelo


def criar_para_treino(congelar_backbone=False, config=CONFIG):
    modelo = criar_modelo(config)
    modelo.congelar_backbone(congelar_backbone)
    return modelo, {"pesos": None, "inicializacao": "he_normal_fan_out",
                    "canais": [28, 28, 42, 64]}

"""Dois estágios residuais: conv1=32, layer1=32, layer2=64; saída única."""
from torch import nn
from res_theo_especial.res_theo_especial import ResTheo, inicializar_modelo_do_zero
from res_theo_especial.dropout_comum import instalar_dropout
from .config_theo_curto import CONFIG


class TheoCurto(ResTheo):
    def __init__(self):
        super().__init__(False)
        del self.layer3
        self.fc = nn.Linear(64, 1)
        inicializar_modelo_do_zero(self)

    def congelar_backbone(self, congelar=True):
        self.backbone_congelado = congelar
        for nome in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2"):
            modulo = getattr(self, nome)
            modulo.requires_grad_(not congelar)
            modulo.train(self.training and not congelar)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if self.backbone_congelado:
            self.congelar_backbone(True)
        return self

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        return self.fc(self.flatten(self.avgpool(x)))


def criar_modelo(config=CONFIG):
    modelo = TheoCurto()
    instalar_dropout(modelo, config)
    return modelo


def criar_para_treino(congelar_backbone=False, config=CONFIG):
    modelo = criar_modelo(config)
    modelo.congelar_backbone(congelar_backbone)
    return modelo, {"pesos": None, "inicializacao": "he_normal_fan_out",
                    "canais": [32, 32, 64], "layer3_removida": True}

"""CNN sequencial: uma Conv-BN-ReLU por estágio, sem somas ou projeções."""
import torch
from torch import nn
from res_theo_especial.res_theo_especial import inicializar_modelo_do_zero
from res_theo_especial.dropout_comum import instalar_dropout


def posicoes_dropout(canais):
    return ('bn1', 'relu', 'maxpool',
            *(f'layer{i}.{j}' for i in range(1, len(canais) + 1) for j in range(3)),
            'avgpool', 'flatten')


class TheoSemRes(nn.Module):
    def __init__(self, canais):
        super().__init__()
        self.canais = tuple(canais)
        self.conv1 = nn.Conv2d(3, 28, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(28)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        entrada = 28
        for i, saida in enumerate(canais, 1):
            setattr(self, f'layer{i}', nn.Sequential(
                nn.Conv2d(entrada, saida, 3, stride=1 if i == 1 else 2,
                          padding=1, bias=False),
                nn.BatchNorm2d(saida), nn.ReLU(inplace=True)))
            entrada = saida
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(canais[-1], 1)
        inicializar_modelo_do_zero(self)
        self.congelar_backbone(False)

    def congelar_backbone(self, congelar=True):
        self.backbone_congelado = congelar
        for nome, modulo in self.named_children():
            if nome not in ('fc', 'dropout_comum'):
                modulo.requires_grad_(not congelar)
                modulo.train(self.training and not congelar)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_congelado:
            self.congelar_backbone(True)
        return self

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for i in range(1, len(self.canais) + 1):
            x = getattr(self, f'layer{i}')(x)
        return self.fc(self.flatten(self.avgpool(x)))

    @torch.inference_mode()
    def predict_proba(self, x):
        if self.training:
            raise RuntimeError('Chame eval() antes da inferência.')
        return self(x).sigmoid()


def criar(canais, config, congelar=False):
    modelo = TheoSemRes(canais)
    instalar_dropout(modelo, config)
    modelo.congelar_backbone(congelar)
    return modelo, dict(pesos=None, inicializacao='he_normal_fan_out',
                        canais=[28, *canais], residuos=False,
                        convolucoes_por_estagio=1)

"""ResNet-34 completa e estreita, com conv1 de 64 canais."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet34
from torchvision.models.resnet import BasicBlock, conv1x1

PESOS = ResNet18_Weights.IMAGENET1K_V1
PESOS_CONV1 = ResNet34_Weights.IMAGENET1K_V1
ETAPAS_BACKBONE = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")


class ResTheo(nn.Module):
    def __init__(self, congelar_backbone=False):
        super().__init__()
        self.inplanes, self.dilation, self.groups, self.base_width = 64, 1, 1, 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 3)
        self.layer2 = self._make_layer(64, 4, stride=2)
        self.layer3 = self._make_layer(128, 6, stride=2)
        self.layer4 = self._make_layer(256, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(256, 1)
        inicializar_modelo_do_zero(self)
        self.congelar_backbone(congelar_backbone)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes))
        camadas = [BasicBlock(self.inplanes, planes, stride, downsample,
                              self.groups, self.base_width, self.dilation, nn.BatchNorm2d)]
        self.inplanes = planes
        camadas.extend(BasicBlock(self.inplanes, planes, groups=self.groups,
                                  base_width=self.base_width, dilation=self.dilation,
                                  norm_layer=nn.BatchNorm2d) for _ in range(1, blocks))
        return nn.Sequential(*camadas)

    def congelar_backbone(self, congelar=True):
        self.backbone_congelado = congelar
        for nome in ETAPAS_BACKBONE:
            modulo = getattr(self, nome)
            modulo.requires_grad_(not congelar)
            modulo.train(self.training and not congelar)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_congelado:
            for nome in ETAPAS_BACKBONE:
                getattr(self, nome).eval()
        return self

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.flatten(self.avgpool(x)))

    @torch.inference_mode()
    def predict_proba(self, x):
        if self.training:
            raise RuntimeError("Chame modelo.eval() antes de predict_proba().")
        return torch.sigmoid(self(x))


def inicializar_modelo_do_zero(modelo):
    for modulo in modelo.modules():
        if isinstance(modulo, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(modulo.weight, mode="fan_out", nonlinearity="relu")
            if modulo.bias is not None:
                nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(modulo.weight)
            nn.init.zeros_(modulo.bias)
            modulo.reset_running_stats()


def criar_res_theo_especial34(congelar_backbone=False, config=None):
    from .config_res_theo_especial34 import CONFIG, validar_dropout
    config = CONFIG if config is None else config
    validar_dropout(config)
    modelo = ResTheo(congelar_backbone)
    origem_conv1 = None
    if config.usar_pesos_pre_treinados_conv1:
        origem_conv1 = resnet34(weights=PESOS_CONV1).conv1.weight.detach()
        with torch.no_grad():
            modelo.conv1.weight.copy_(origem_conv1)
    resumo = {"pesos": None, "inicializacao": config.inicializacao_do_zero,
              "canais": [64, 32, 64, 128, 256], "blocos": [3, 4, 6, 3],
              "conv1_pesos": PESOS_CONV1.name if origem_conv1 is not None else None,
              "conv1_epocas_congelada": config.epocas_conv1_congelada}
    from .dropout_comum import instalar_dropout
    instalar_dropout(modelo, config)
    return modelo, resumo

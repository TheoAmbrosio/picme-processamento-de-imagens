"""ResNet-18 reduzida até layer3: canais 32, 32, 64 e 128."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import BasicBlock, conv1x1

PESOS = ResNet18_Weights.IMAGENET1K_V1  # Apenas define a normalização das entradas.
ETAPAS_BACKBONE = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3")


class ResTheo(nn.Module):
    def __init__(self, congelar_backbone: bool = False):
        super().__init__()
        self.inplanes = 32
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(128, 1)
        inicializar_modelo_do_zero(self)
        self.congelar_backbone(congelar_backbone)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        camadas = [BasicBlock(self.inplanes, planes, stride, downsample,
                              self.groups, self.base_width, self.dilation,
                              nn.BatchNorm2d)]
        self.inplanes = planes
        camadas.extend(BasicBlock(self.inplanes, planes, groups=self.groups,
                                  base_width=self.base_width, dilation=self.dilation,
                                  norm_layer=nn.BatchNorm2d)
                        for _ in range(1, blocks))
        return nn.Sequential(*camadas)

    def congelar_backbone(self, congelar: bool = True) -> None:
        self.backbone_congelado = congelar
        for nome in ETAPAS_BACKBONE:
            modulo = getattr(self, nome)
            modulo.requires_grad_(not congelar)
            modulo.train(self.training and not congelar)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_congelado:
            for nome in ETAPAS_BACKBONE:
                getattr(self, nome).eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                  # [B, 32, 112, 112]
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)                # [B, 32, 56, 56]
        x = self.layer1(x)                 # [B, 32, 56, 56]
        x = self.layer2(x)                 # [B, 64, 28, 28]
        x = self.layer3(x)                 # [B, 128, 14, 14]
        return self.fc(self.flatten(self.avgpool(x)))

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("Chame modelo.eval() antes de predict_proba().")
        return torch.sigmoid(self(x))


def inicializar_modelo_do_zero(modelo: nn.Module) -> None:
    for modulo in modelo.modules():
        if isinstance(modulo, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(modulo.weight, mode="fan_out", nonlinearity="relu")
            if modulo.bias is not None:
                nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if modulo.weight is not None:
                nn.init.ones_(modulo.weight)
            if modulo.bias is not None:
                nn.init.zeros_(modulo.bias)
            modulo.reset_running_stats()


def criar_res_theo_especial(congelar_backbone: bool = False, config=None):
    from .config_res_theo_especial import CONFIG, validar_dropout
    config = CONFIG if config is None else config
    validar_dropout(config)
    if config.usar_pesos_pre_treinados:
        raise ValueError("res_theo_especial não aceita pesos pré-treinados: os canais foram reduzidos.")
    modelo = ResTheo(congelar_backbone)
    resumo = {"pesos": None, "inicializacao": config.inicializacao_do_zero,
              "canais": [32, 32, 64, 128],
              "inicializados": [nome for nome, _ in modelo.named_parameters()]}
    print("ResTheo especial do zero: canais 32/32/64/128; He normal (variância 2/fan_out), vieses em 0; "
          "BatchNorm com escala 1 e deslocamento 0.")
    from .dropout_comum import instalar_dropout
    instalar_dropout(modelo, config)
    return modelo, resumo

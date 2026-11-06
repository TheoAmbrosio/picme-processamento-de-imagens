"""res_theo34: ResNet-34 oficial até layer3, com uma única saída binária."""
import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.models.resnet import ResNet

PESOS = ResNet34_Weights.IMAGENET1K_V1
ETAPAS_BACKBONE = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3")


class ResTheo(nn.Module):
    def __init__(self, origem: ResNet, congelar_backbone: bool = False):
        super().__init__()
        # Reutilização dos módulos: parâmetros E buffers de BatchNorm são preservados.
        for nome in ETAPAS_BACKBONE:
            setattr(self, nome, getattr(origem, nome))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(256, 2)
        inicializar_nova_camada(self.fc)
        self.congelar_backbone(congelar_backbone)

    def congelar_backbone(self, congelar: bool = True) -> None:
        self.backbone_congelado = congelar
        for nome in ETAPAS_BACKBONE:
            modulo = getattr(self, nome)
            modulo.requires_grad_(not congelar)
            modulo.train(self.training and not congelar)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_congelado:
            # Congelar também as estatísticas de BN durante o treino do classificador.
            for nome in ETAPAS_BACKBONE:
                getattr(self, nome).eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                  # [B, 64, 112, 112]
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)                # [B, 64, 56, 56]
        x = self.layer1(x)                 # [B, 64, 56, 56]: 3 BasicBlocks
        x = self.layer2(x)                 # [B, 128, 28, 28]: 4 BasicBlocks
        x = self.layer3(x)                 # [B, 256, 14, 14]: 6 BasicBlocks
        x = self.flatten(self.avgpool(x))  # [B, 256], inclusive B=1
        return self.fc(x)                  # [B, 1], logits sem sigmoid

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """P(cachorro), shape [B, 1]. Chame modelo.eval() antes da inferência."""
        if self.training:
            raise RuntimeError("Chame modelo.eval() antes de predict_proba().")
        return torch.softmax(self(x),dim=1)[:,1:2]


def converter_pesos_binarios(estado):
    pesos=dict(estado)
    pesos['fc.weight']=torch.cat((torch.zeros_like(estado['fc.weight']),estado['fc.weight']),dim=0)
    pesos['fc.bias']=torch.cat((torch.zeros_like(estado['fc.bias']),estado['fc.bias']),dim=0)
    return pesos


def inicializar_nova_camada(camada: nn.Linear) -> None:
    nn.init.normal_(camada.weight, mean=0.0, std=1.0)
    nn.init.normal_(camada.bias, mean=0.0, std=1.0)


def inicializar_modelo_do_zero(modelo: nn.Module) -> None:
    """He normal para ReLU (variância 2/fan_out); vieses 0; BatchNorm 1/0."""
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


def conferir_transferencia(modelo: ResTheo, origem: ResNet) -> dict:
    """Não tolera chaves, shapes ou valores inesperados, incluindo buffers."""
    esperado = {k: v for k, v in origem.state_dict().items()
                if not k.startswith(("layer4.", "fc."))}
    recebido = {k: v for k, v in modelo.state_dict().items() if not k.startswith("fc.")}
    if esperado.keys() != recebido.keys():
        raise ValueError(f"Chaves incompatíveis: {esperado.keys() ^ recebido.keys()}")
    for nome, valor in esperado.items():
        if recebido[nome].shape != valor.shape or not torch.equal(recebido[nome], valor):
            raise ValueError(f"Divergência na transferência: {nome}")
    if {k for k in modelo.state_dict() if k not in recebido} != {"fc.weight", "fc.bias"}:
        raise ValueError("Parâmetros novos inesperados no classificador.")
    return {
        "pesos": PESOS.name,
        "tensores_reaproveitados": len(esperado),
        "buffers_reaproveitados": sum(k in esperado for k, _ in origem.named_buffers()),
        "descartados": [k for k in origem.state_dict() if k.startswith(("layer4.", "fc."))],
        "inicializados": ["fc.weight", "fc.bias"],
    }


def criar_res_theo34(congelar_backbone: bool = False, config=None):
    from .config_res_theo34 import CONFIG, validar_dropout
    config = CONFIG if config is None else config
    validar_dropout(config)
    pesos = PESOS if config.usar_pesos_pre_treinados else None
    origem = resnet34(weights=pesos)
    modelo = ResTheo(origem, congelar_backbone)
    if config.usar_pesos_pre_treinados:
        resumo = conferir_transferencia(modelo, origem)
        print(f"ImageNet: {resumo['tensores_reaproveitados']} tensores reaproveitados "
              f"({resumo['buffers_reaproveitados']} buffers de BatchNorm incluídos).")
        print(f"Descartados: layer4.* e fc.* originais ({len(resumo['descartados'])} tensores).")
    else:
        inicializar_modelo_do_zero(modelo)
        resumo = {"pesos": None, "inicializacao": config.inicializacao_do_zero,
                  "inicializados": [nome for nome, _ in modelo.named_parameters()]}
        print("Sem pesos pré-treinados: convoluções e fc com He normal (variância 2/fan_out), vieses em 0; "
              "BatchNorm com escala 1 e deslocamento 0.")
    from .dropout_comum import instalar_dropout
    instalar_dropout(modelo, config)
    return modelo, resumo

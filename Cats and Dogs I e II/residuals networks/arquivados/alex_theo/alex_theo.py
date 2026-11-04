"""AlexNet oficial ImageNet, substituindo somente a saída por um logit binário."""
import torch
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet

PESOS = AlexNet_Weights.IMAGENET1K_V1


class ControleDropoutAlexNet:
    def __init__(self, camadas, probabilidade):
        self.camadas = list(camadas)
        self.posicoes = ("classifier.0", "classifier.3")
        self.probabilidade = float(probabilidade)
        self.probabilidade_maxima = float(probabilidade)
        self.definir_probabilidade(probabilidade)

    def definir_probabilidade(self, probabilidade):
        if not 0 <= probabilidade < 1:
            raise ValueError("A probabilidade do dropout deve estar em [0,1).")
        self.probabilidade = float(probabilidade)
        for camada in self.camadas:
            camada.p = self.probabilidade

    def estado(self):
        return {"tipo": "alexnet_classifier", "posicoes": self.posicoes,
                "probabilidade": self.probabilidade,
                "probabilidade_maxima": self.probabilidade_maxima}

    def carregar_estado(self, estado):
        if (estado.get("tipo") != "alexnet_classifier"
                or tuple(estado.get("posicoes", ())) != self.posicoes
                or estado.get("probabilidade_maxima", estado.get("probabilidade"))
                   != self.probabilidade_maxima):
            raise ValueError("Configuração de dropout diverge do checkpoint.")
        self.definir_probabilidade(estado["probabilidade"])


class AlexTheo(nn.Module):
    def __init__(self, origem, congelar_backbone=False, dropout_probabilidade=0.1):
        super().__init__()
        self.features = origem.features
        self.avgpool = origem.avgpool
        self.classifier = origem.classifier
        self.classifier[6] = nn.Linear(4096, 1)
        nn.init.kaiming_normal_(self.classifier[6].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.classifier[6].bias)
        self.dropout_comum = ControleDropoutAlexNet(
            (self.classifier[0], self.classifier[3]), dropout_probabilidade)
        self.congelar_backbone(congelar_backbone)

    @property
    def conv1(self):
        return self.features[0]

    def congelar_backbone(self, congelar=True):
        self.backbone_congelado = congelar
        self.features.requires_grad_(not congelar)
        for camada in self.classifier[:-1]:
            camada.requires_grad_(not congelar)
        self.features.train(self.training and not congelar)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_congelado:
            self.features.eval()
        return self

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @torch.inference_mode()
    def predict_proba(self, x):
        if self.training:
            raise RuntimeError("Chame modelo.eval() antes de predict_proba().")
        return torch.sigmoid(self(x))


def criar_alex_theo(congelar_backbone=False, config=None):
    from .config_alex_theo import CONFIG
    config = CONFIG if config is None else config
    config.validar()
    origem = alexnet(weights=PESOS if config.usar_pesos_pre_treinados else None)
    modelo = AlexTheo(origem, congelar_backbone, config.dropout_probabilidade)
    resumo = {"pesos": PESOS.name, "arquitetura": "AlexNet",
              "saida_original": 1000, "saida_nova": 1,
              "inicializacao_saida": config.inicializacao_do_zero,
              "conv1_epocas_congelada": config.epocas_conv1_congelada}
    print("AlexNet ImageNet: estrutura preservada; classifier[6] substituída por Linear(4096,1).")
    print(f"Dropout do classificador: máximo p={config.dropout_probabilidade}; "
          f"crescente={config.dropout_crescente}.")
    return modelo, resumo

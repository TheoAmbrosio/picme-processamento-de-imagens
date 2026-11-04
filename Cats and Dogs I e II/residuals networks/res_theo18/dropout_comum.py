"""nn.Dropout configurável após posições fixas; zero desativa o dropout."""
from torch import nn


class DropoutComum(nn.Module):
    def __init__(self, posicoes, probabilidade=0.3):
        super().__init__()
        self.posicoes = tuple(posicoes)
        self.probabilidade = probabilidade
        self.camadas = nn.ModuleList(nn.Dropout(probabilidade) for _ in self.posicoes)
        self.handles = []

    def hook(self, indice):
        def aplicar(modulo, entrada, saida):
            return self.camadas[indice](saida)
        return aplicar

    def estado(self):
        return {"tipo": "comum", "posicoes": self.posicoes,
                "probabilidade": self.probabilidade}

    def carregar_estado(self, estado):
        if estado != self.estado():
            raise ValueError("Configuração de dropout diverge do checkpoint.")


def instalar_dropout(modelo, config):
    from .config_res_theo18 import validar_dropout
    validar_dropout(config)
    if hasattr(modelo, "dropout_comum"):
        raise ValueError("Dropout já instalado.")
    modulos = dict(modelo.named_modules())
    tipos = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d,
             nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear)
    posicoes = tuple(config.dropout_posicoes or ()) if config.dropout_probabilidade else ()
    for nome in posicoes:
        if nome not in modulos or not isinstance(modulos[nome], tipos):
            raise ValueError(f"Ponto de dropout inválido ou residual: {nome}")
    controle = DropoutComum(posicoes, config.dropout_probabilidade)
    modelo.add_module("dropout_comum", controle)
    controle.train(modelo.training)
    for indice, nome in enumerate(controle.posicoes):
        controle.handles.append(modulos[nome].register_forward_hook(controle.hook(indice)))
    print(f"Dropout comum p={config.dropout_probabilidade}; posições fixas={controle.posicoes}; "
          "desativado em avaliação.", flush=True)
    return controle

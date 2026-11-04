"""Agenda definida pelo participante: 50 épocas originais + 30 ampliadas."""
from dataclasses import asdict, dataclass, replace
from theo_ajuste.config_res_theo_especial import CONFIG as BASE, ConfigTreino as ConfigBase


@dataclass
class ConfigTreino(ConfigBase):
    warmup_epoca_inicio_global: int = 51
    dropout_epoca_intermediaria: int = 50
    dropout_probabilidade_intermediaria: float = 0.1


def configuracao(posicoes=None):
    return replace(ConfigTreino(**asdict(BASE)), epocas=80, epocas_iniciais_sem_ampliacao=50,
                   otimizador="AdamW", learning_rate=0.001,
                   argumentos_otimizador={"weight_decay": 0.01},
                   checkpoint_origem="", dropout_probabilidade=0.2,
                   dropout_epoca_inicio=1, dropout_epoca_maximo=25,
                   dropout_epoca_inicio_reducao=25, dropout_epoca_fim_reducao=75,
                   dropout_probabilidade_final=0.05,
                   dropout_posicoes=posicoes or BASE.dropout_posicoes,
                   warmup_linear=True, warmup_epoca_final=25)


def taxa_aprendizado_epoca(config, epoca):
    if epoca <= config.epocas_iniciais_sem_ampliacao:
        return config.learning_rate
    local = epoca - config.epocas_iniciais_sem_ampliacao
    if local >= config.warmup_epoca_final:
        return config.learning_rate
    inicial = config.learning_rate / config.warmup_fator_inicial
    return inicial + (config.learning_rate - inicial) * (local - 1) / (config.warmup_epoca_final - 1)


def probabilidade_dropout_epoca(config, epoca):
    if epoca <= 25:
        return config.dropout_probabilidade * (epoca - 1) / 24
    if epoca <= 50:
        return 0.2 + (0.1 - 0.2) * (epoca - 25) / 25
    if epoca <= 75:
        return 0.1 + (0.05 - 0.1) * (epoca - 50) / 25
    return 0.05

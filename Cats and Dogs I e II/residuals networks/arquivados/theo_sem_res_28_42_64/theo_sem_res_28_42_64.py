from modelo_sem_res import criar
from .config_theo_sem_res_28_42_64 import CANAIS, CONFIG


def criar_para_treino(congelar_backbone=False, config=CONFIG):
    return criar(CANAIS, config, congelar_backbone)

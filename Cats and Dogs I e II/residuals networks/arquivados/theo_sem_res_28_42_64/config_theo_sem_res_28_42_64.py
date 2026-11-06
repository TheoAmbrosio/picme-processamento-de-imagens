from agenda_duas_fases import ConfigTreino, configuracao
from modelo_sem_res import posicoes_dropout

CANAIS = (28,42,64)
CONFIG = configuracao(posicoes=posicoes_dropout(CANAIS))

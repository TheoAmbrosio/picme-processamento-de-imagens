from agenda_duas_fases import ConfigTreino, configuracao
from theo_ajuste.config_res_theo_especial import CONFIG as BASE

CONFIG = configuracao(posicoes=tuple(
    n for n in BASE.dropout_posicoes if not n.startswith("layer3.")))

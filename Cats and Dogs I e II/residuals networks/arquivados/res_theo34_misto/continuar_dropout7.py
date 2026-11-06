"""Continua os pesos salvos com AdamW e dropout fixo de 7%."""
from dataclasses import replace
import torch
from .config_res_theo34 import CONFIG as BASE, CONFIG_ATUAL_ANTERIOR
from . import treinar_res_theo34 as treino

CONFIG = replace(BASE, otimizador='AdamW', argumentos_otimizador={'weight_decay': 0.01},
                 dropout_probabilidade=0.07, dropout_crescente=False,
                 dropout_posicoes=CONFIG_ATUAL_ANTERIOR.dropout_posicoes)


if __name__ == '__main__':
    CONFIG.validar()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponível; continuação não iniciada.')
    print('CUDA:',torch.cuda.get_device_name(), 'memória:',torch.cuda.mem_get_info(),flush=True)
    treino.CONFIG = CONFIG
    treino.main()

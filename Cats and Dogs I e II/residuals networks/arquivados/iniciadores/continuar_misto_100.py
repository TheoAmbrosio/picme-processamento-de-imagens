"""Aguarda o serviço atual concluir 30 épocas e retoma em pasta própria até100."""
from pathlib import Path
import subprocess
import time
import os
import shutil
import sys

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / 'residuals networks'))
from res_theo34_misto.config_res_theo34 import CONFIG
from registro_residuals import novo_diretorio, registrar_execucao


def main():
    while subprocess.run(['systemctl','--user','is-active','--quiet','res-theo34-misto']).returncode == 0:
        time.sleep(15)
    import torch
    from res_theo34_misto.treinar_res_theo34 import treinar
    assert torch.cuda.is_available(), 'CUDA indisponível; continuação não iniciada.'
    origem = RAIZ / 'resultados_residuals_networks/res_theo34_misto/20261105_191439_421574_bs96_ep30_cong0ep_do10_dadosamp_pre'
    checkpoint = origem / 'ultima_epoca.pt'
    estado = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert estado['epoca'] == 30, 'Etapa anterior não concluiu 30 épocas.'
    del estado
    CONFIG.validar()
    assert CONFIG.epocas == 100
    pasta = novo_diretorio(CONFIG, 'res_theo34_misto')
    pasta.mkdir(parents=True, exist_ok=True)
    CONFIG.checkpoint = str(pasta / 'ultima_epoca.pt')
    for nome in ('melhor_modelo.pt', 'melhor_modelo.json'):
        shutil.copy2(origem / nome, pasta / nome)
    for p in (pasta/'treinamento.pid', pasta.parent/'treinamento.pid'):
        p.write_text(str(os.getpid())+'\n')
    print(f'Continuação: {pasta}', flush=True)
    with (pasta/'treinamento.log').open('a', buffering=1) as log:
        os.dup2(log.fileno(),1)
        os.dup2(log.fileno(),2)
        registrar_execucao(CONFIG,'res_theo34_misto',extras={'checkpoint_retomada':str(checkpoint), 'epoca_retomada':31})
        treinar(CONFIG, retomar=checkpoint)


if __name__ == '__main__':
    main()

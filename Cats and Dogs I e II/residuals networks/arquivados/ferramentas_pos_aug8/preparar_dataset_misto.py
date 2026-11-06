"""Exporta apenas manifestos virtuais e parâmetros; preserva as imagens existentes."""
import csv
import json
import os
from pathlib import Path
import sys

RAIZ=Path(__file__).resolve().parent
sys.path.insert(0,str(RAIZ/'residuals networks'))
from dataset_misto import CONFIG,carregar_divisao


def main():
    origem=RAIZ/'dataset_ampliado/224x224/divisao_balanceada/manifesto_variantes.csv'
    destino=RAIZ/'dataset_ampliado/misto_intensificado'
    if destino.exists() and any(destino.iterdir()):
        raise FileExistsError(f'Pasta já contém dados; preservada: {destino}')
    destino.mkdir(exist_ok=True)
    conjuntos=carregar_divisao(origem)
    for split,amostras in conjuntos.items():
        with (destino/f'{split}.csv').open('w',encoding='utf-8',newline='') as f:
            escritor=csv.writer(f)
            escritor.writerow(('caminho','rotulo','variante'))
            escritor.writerows((os.path.relpath(p,destino),r,v) for p,r,v in amostras)
    meta=dict(config=CONFIG.parametros_efetivos(),manifesto_divisao_origem=str(origem),
              contagens={s:len(a) for s,a in conjuntos.items()},
              formato='ampliações virtuais por leitura; validação histórica preservada')
    (destino/'parametros.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta['contagens']))
    print(destino)


if __name__=='__main__':main()

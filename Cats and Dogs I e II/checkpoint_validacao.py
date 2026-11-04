"""Preserva o checkpoint de maior acurácia de validação; empates mantêm o primeiro."""
import json
from pathlib import Path
import shutil


def preservar_melhor_checkpoint(checkpoint, historico):
    atual = historico[-1]
    anteriores = historico[:-1]
    if anteriores and atual['acuracia_validacao'] <= max(
            h['acuracia_validacao'] for h in anteriores):
        return False
    origem = Path(checkpoint)
    destino = origem.parent / 'melhor_modelo.pt'
    temporario = destino.with_suffix('.pt.tmp')
    shutil.copyfile(origem, temporario)
    temporario.replace(destino)
    resumo = dict(criterio='acuracia_validacao', modo='max',
                  empate='preserva_primeiro', epoca=atual['epoca'],
                  acuracia_validacao=atual['acuracia_validacao'],
                  checkpoint=destino.name)
    arquivo = origem.parent / 'melhor_modelo.json'
    temporario = arquivo.with_suffix('.json.tmp')
    temporario.write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporario.replace(arquivo)
    print(f"Melhor checkpoint: época {atual['epoca']}, acerto validação "
          f"{atual['acuracia_validacao']:.2%} — {destino}", flush=True)
    return True

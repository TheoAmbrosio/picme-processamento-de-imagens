import json
from checkpoint_validacao import preservar_melhor_checkpoint


def test_melhor_preserva_pesos_em_piora_e_empate(tmp_path):
    origem = tmp_path / 'ultima_epoca.pt'
    historico = []
    for epoca, (acerto, esperado) in enumerate(((.8, True), (.7, False), (.8, False), (.9, True)), 1):
        origem.write_bytes(f'pesos_epoca_{epoca}'.encode())
        historico.append(dict(epoca=epoca, acuracia_validacao=acerto))
        assert preservar_melhor_checkpoint(origem, historico) is esperado
        melhor = 4 if epoca == 4 else 1
        assert (tmp_path / 'melhor_modelo.pt').read_bytes() == f'pesos_epoca_{melhor}'.encode()
        resumo = json.loads((tmp_path / 'melhor_modelo.json').read_text())
        assert resumo['epoca'] == melhor
        assert origem.read_bytes() == f'pesos_epoca_{epoca}'.encode()

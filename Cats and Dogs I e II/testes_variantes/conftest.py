"""Caminhos dos pacotes também para executar testes isoladamente."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
for caminho in (RAIZ, RAIZ / 'residuals networks', RAIZ / 'residuals networks/arquivados',
                RAIZ / 'residuals networks/arquivados/ferramentas_pos_aug8'):
    sys.path.insert(0, str(caminho))

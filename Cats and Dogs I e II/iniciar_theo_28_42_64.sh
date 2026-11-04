#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/residuals networks:$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=/home/al.theo.ambrosio/Documentos/.venvs/res_theo/bin/python
BASE="resultados_residuals_networks/theo_28_42_64"
PIDFILE="$BASE/treinamento.pid"
ATUAL="resultados_residuals_networks/res_theo_especial/treinamento.pid"
if [[ -f "$ATUAL" ]] && kill -0 "$(cat "$ATUAL")" 2>/dev/null; then
    echo "O treino original ainda está ativo. Aguarde terminar antes de iniciar o fine-tuning."
    exit 1
fi
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Já existe um treinamento ativo: PID $(cat "$PIDFILE")"
    exit 1
fi
"$PYTHON" - <<'CHECK'
from pathlib import Path
import torch, torchvision
from theo_28_42_64.config_theo_28_42_64 import CONFIG
from theo_28_42_64.treinar_theo_28_42_64 import validar_config_dataset
CONFIG.validar()
assert torch.cuda.is_available(), 'CUDA indisponível: treinamento não iniciado.'
validar_config_dataset(CONFIG, Path(CONFIG.manifesto))
relatorio = Path(CONFIG.manifesto).parent / 'relatorio.txt'
assert relatorio.is_file() and 'RESULTADO DA CHECAGEM: APROVADO' in relatorio.read_text(), 'Preparação não aprovada.'
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(), 'memória livre/total:', torch.cuda.mem_get_info())
print('Configuração:', CONFIG)
CHECK
RUN=$("$PYTHON" - <<'PASTA'
from theo_28_42_64.config_theo_28_42_64 import CONFIG
from registro_residuals import novo_diretorio
print(novo_diretorio(CONFIG, "theo_28_42_64"))
PASTA
)
LOG="$RUN/treinamento.log"
nohup "$PYTHON" -u -m theo_28_42_64.treinar_theo_28_42_64 --resultados "$RUN" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PIDFILE"
printf '%s\n' "$PID" > "$RUN/treinamento.pid"
printf 'PID: %s\nResultados: %s\nLog: %s\n' "$PID" "$RUN" "$LOG"
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
    cat "$LOG"
    exit 1
fi
printf 'Para acompanhar: tail -f "%s"\n' "$LOG"

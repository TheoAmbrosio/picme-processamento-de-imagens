#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=/home/al.theo.ambrosio/Documentos/.venvs/res_theo/bin/python
bash iniciar_res_theo18.sh
PID=$(cat resultados_residuals_networks/res_theo18/treinamento.pid)
RUN=""
for ARQUIVO in resultados_residuals_networks/res_theo18/*/treinamento.pid; do
    if [[ -f "$ARQUIVO" ]] && [[ "$(cat "$ARQUIVO")" == "$PID" ]]; then
        RUN="${ARQUIVO%/treinamento.pid}"
        break
    fi
done
if [[ -z "$RUN" ]]; then
    echo "Não foi possível identificar a execução do 18; o 34 não será iniciado."
    exit 1
fi
printf 'Aguardando res_theo18 (PID %s) concluir...\n' "$PID"
tail --pid="$PID" -f /dev/null
"$PYTHON" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
pasta=Path(sys.argv[1])
metadados=pasta/'parametros.json'
if (not metadados.exists() or json.loads(metadados.read_text()).get('estado') != 'concluido'
        or not (pasta/'ultima_epoca.pt').exists()):
    raise SystemExit('O 18 não concluiu com sucesso; o 34 não será iniciado. Confira o treinamento.log do 18.')
PY
printf 'res_theo18 concluído. Iniciando res_theo34...\n'
bash iniciar_res_theo34.sh

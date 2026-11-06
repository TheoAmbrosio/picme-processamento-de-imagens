#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PIDFILE='resultados_residuals_networks/res_theo34_aug8/treinamento.pid'
if [[ -f "$PIDFILE" ]]; then
    PID_TREINO=$(cat "$PIDFILE")
    while kill -0 "$PID_TREINO" 2>/dev/null; do
        COMANDO=$(ps -p "$PID_TREINO" -o args= || true)
        [[ "$COMANDO" == *res_theo34_aug8.treinar_res_theo34* ]] || break
        sleep 30
    done
fi
exec /home/al.theo.ambrosio/Documentos/.venvs/res_theo/bin/python docs/diagramas/gerar_res_theo34_aug8.py

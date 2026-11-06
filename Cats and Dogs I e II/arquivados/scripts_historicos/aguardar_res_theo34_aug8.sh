#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
ANTERIOR="resultados_residuals_networks/res_theo34_finetuning/treinamento.pid"
if [[ -f "$ANTERIOR" ]]; then
    PID_ANTERIOR=$(cat "$ANTERIOR")
    while kill -0 "$PID_ANTERIOR" 2>/dev/null; do
        COMANDO=$(ps -p "$PID_ANTERIOR" -o args= || true)
        [[ "$COMANDO" == *res_theo34_finetuning.treinar_res_theo34* ]] || break
        echo "Aguardando fine-tuning atual PID $PID_ANTERIOR."
        sleep 30
    done
fi
exec bash iniciar_res_theo34_aug8.sh

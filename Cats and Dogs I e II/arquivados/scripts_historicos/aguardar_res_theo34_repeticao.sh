#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PID_AGUARDADO="${1:?Informe o PID do treino anterior}"
echo "Aguardando o treino PID $PID_AGUARDADO terminar."
while kill -0 "$PID_AGUARDADO" 2>/dev/null; do
    COMANDO=$(ps -p "$PID_AGUARDADO" -o args= || true)
    [[ "$COMANDO" == *theo_sem_res_28_42_64_88.treinar_theo_sem_res_28_42_64_88* ]] || break
    sleep 10
done
echo "Treino anterior encerrado. Iniciando repetição histórica ResTheo34."
exec bash iniciar_res_theo34_repeticao.sh

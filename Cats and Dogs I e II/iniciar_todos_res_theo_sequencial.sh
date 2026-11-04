#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=/home/al.theo.ambrosio/Documentos/.venvs/res_theo/bin/python

executar_e_aguardar() {
    local modelo="$1"
    local iniciador="$2"
    local base="resultados_residuals_networks/$modelo"
    local pidfile="$base/treinamento.pid"
    local pid run arquivo

    printf '\nIniciando %s...\n' "$modelo"
    bash "$iniciador"
    pid="$(cat "$pidfile")"
    run=""

    for arquivo in "$base"/*/treinamento.pid; do
        if [[ -f "$arquivo" ]] && [[ "$(cat "$arquivo")" == "$pid" ]]; then
            run="${arquivo%/treinamento.pid}"
            break
        fi
    done
    if [[ -z "$run" ]]; then
        echo "Não foi possível identificar a execução de $modelo. Sequência interrompida."
        exit 1
    fi

    printf 'Aguardando %s (PID %s). Log: %s/treinamento.log\n' "$modelo" "$pid" "$run"
    tail --pid="$pid" -f /dev/null

    "$PYTHON" - "$modelo" "$run" <<'PY'
import json
import sys
from pathlib import Path

modelo, caminho = sys.argv[1:]
pasta = Path(caminho)
metadados = pasta / "parametros.json"
if (not metadados.exists()
        or json.loads(metadados.read_text(encoding="utf-8")).get("estado") != "concluido"
        or not (pasta / "ultima_epoca.pt").exists()):
    raise SystemExit(
        f"{modelo} não concluiu corretamente; sequência interrompida. "
        f"Confira {pasta / 'treinamento.log'}"
    )
PY
    printf '%s concluído corretamente.\n' "$modelo"
}

executar_e_aguardar res_theo18 iniciar_res_theo18.sh
executar_e_aguardar res_theo34 iniciar_res_theo34.sh
executar_e_aguardar res_theo_especial iniciar_res_theo_especial.sh

printf '\nTodos os modelos concluíram corretamente.\n'

#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON=/home/al.theo.ambrosio/Documentos/.venvs/res_theo/bin/python

executar_e_aguardar() {
    local modelo="$1" iniciador="$2"
    local base="resultados_residuals_networks/$modelo"
    local pidfile="$base/treinamento.pid"
    local pid run="" arquivo

    printf '\nIniciando %s...\n' "$modelo"
    bash "$iniciador"
    pid="$(cat "$pidfile")"
    for arquivo in "$base"/*/treinamento.pid; do
        if [[ -f "$arquivo" ]] && [[ "$(cat "$arquivo")" == "$pid" ]]; then
            run="${arquivo%/treinamento.pid}"
            break
        fi
    done
    [[ -n "$run" ]] || { echo "Execução de $modelo não identificada."; exit 1; }
    printf 'Aguardando %s (PID %s). Log: %s/treinamento.log\n' "$modelo" "$pid" "$run"
    tail --pid="$pid" -f /dev/null
    "$PYTHON" - "$modelo" "$run" <<'PY'
import json, sys
from pathlib import Path
modelo, caminho = sys.argv[1:]
pasta = Path(caminho)
metadados = pasta / "parametros.json"
if (not metadados.exists()
        or json.loads(metadados.read_text(encoding="utf-8")).get("estado") != "concluido"
        or not (pasta / "ultima_epoca.pt").exists()):
    raise SystemExit(f"{modelo} não concluiu corretamente; confira {pasta / 'treinamento.log'}")
PY
    printf '%s concluído corretamente.\n' "$modelo"
}

executar_e_aguardar res_theo_especial iniciar_res_theo_especial.sh
executar_e_aguardar res_theo_especial34 iniciar_res_theo_especial34.sh
executar_e_aguardar res_theo_especial50 iniciar_res_theo_especial50.sh

printf '\nTodas as ResTheo especiais concluíram corretamente.\n'

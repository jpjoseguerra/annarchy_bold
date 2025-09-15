#!/usr/bin/env bash
# Reanuda simulaciones del repo para una condición (input_factor, stimulus)
# Saltando seeds que ya tienen resultados válidos.
# Uso:
#   ./run_or_resume_repo.sh 1.2 3 0 39 1     # if=1.2, stim=3, seeds 0..39, P=1
#   ./run_or_resume_repo.sh 5   1 0 39 1     # if=5,   stim=1, seeds 0..39, P=1
# Opcional:
#   FORCE=1  ./run_or_resume_repo.sh ...     # fuerza re-ejecución aunque exista
#   DRY=1    ./run_or_resume_repo.sh ...     # solo muestra qué correría

set -euo pipefail

IF="${1:?input_factor}"; STIM="${2:?stimulus}"
SEED_START="${3:-0}";    SEED_END="${4:-39}"; JOBS="${5:-1}"

# tags de archivo (1.2 -> 1_2 ; 5   -> 5)
IF_TAG="$(printf "%s" "$IF" | tr . _)"
REC_PAT="dataRaw/simulations_BOLDfromDifferentSources_recordingsB_${IF_TAG}_${STIM}__%s.npy"

# limitar hilos para no saturar WSL
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p srcSim/annarchy_folders logs

# Función: ¿existe y es "válido" el recordings? (contiene '1;BOLD' y >1000 muestras)
is_valid_recordings () {
  local f="$1"
  python - "$f" <<'PY'
import sys, numpy as np
p=sys.argv[1]
try:
    import os
    if not os.path.exists(p) or os.path.getsize(p) < 1024:
        raise SystemExit(2)
    d=np.load(p, allow_pickle=True).item()
    ok = isinstance(d, dict) and ('1;BOLD' in d) and isinstance(d['1;BOLD'], np.ndarray) and d['1;BOLD'].size>1000
    raise SystemExit(0 if ok else 3)
except SystemExit as e:
    raise
except Exception:
    raise SystemExit(4)
PY
}

# Construir lista de seeds pendientes
pending=()
for sid in $(seq "$SEED_START" "$SEED_END"); do
  rec=$(printf "$REC_PAT" "$sid")
  if [[ "${FORCE:-0}" == "1" ]]; then
    pending+=("$sid")
  else
    if is_valid_recordings "$rec"; then
      echo "✓ seed=$sid ya existe ($rec) — salto"
    else
      echo "→ seed=$sid falta o inválido — agendar"
      pending+=("$sid")
    fi
  fi
done

if [[ ${#pending[@]} -eq 0 ]]; then
  echo "✔ Nada pendiente para if=${IF} stim=${STIM} seeds ${SEED_START}..${SEED_END}."
  exit 0
fi

echo "Correré ${#pending[@]} seeds: ${pending[*]}"
[[ "${DRY:-0}" == "1" ]] && { echo "(DRY RUN) — no ejecuto"; exit 0; }

# Ejecutar en paralelo con -P $JOBS
printf "%s\n" "${pending[@]}" | xargs -n 1 -P "$JOBS" -I {} bash -lc '
  sid="{}"
  echo "⇒ simID=$sid (if='"$IF"', stim='"$STIM"')"
  cd srcSim
  source ../fix_annarchy_env.sh
  python simulations.py '"$IF"' '"$STIM"' "$sid" > ../logs/sim_${sid}.log 2>&1
'
echo "✅ Listo."

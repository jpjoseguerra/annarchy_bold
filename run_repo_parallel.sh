#!/usr/bin/env bash
set -euo pipefail

# Parámetros (puedes sobrescribir por env al invocar)
INPUT_FACTOR="${INPUT_FACTOR:-1.2}"   # intensidad Poisson
STIMULUS="${STIMULUS:-1}"            # 1 = pulso 100 ms
START="${START:-1}"                  # simID inicial
END="${END:-39}"                     # simID final (incl.)
JOBS="${JOBS:-8}"                    # núm. de procesos concurrentes

# Evita sobre-suscripción BLAS cuando paralelizas
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p srcSim/annarchy_folders dataRaw

echo "input_factor=$INPUT_FACTOR  stimulus=$STIMULUS  seeds=$START..$END  jobs=$JOBS"

run_one() {
  local sid="$1"
  # si ya hay artefactos de la seed, salta (idempotente)
  if compgen -G "dataRaw/*__${sid}.*" > /dev/null; then
    echo "  simID=${sid} (ya existe algo en dataRaw/*__${sid}.*, salto)"
    return 0
  fi
  echo "  simID=${sid} (lanzando)"
  ( cd srcSim && python simulations.py "$INPUT_FACTOR" "$STIMULUS" "$sid" \
      > "../dataRaw/sim${sid}.log" 2>&1 )
}

# Lanza en paralelo con control de concurrencia
pids=()
active=0
for sid in $(seq "$START" "$END"); do
  run_one "$sid" &
  pids+=($!)
  active=$((active+1))
  if (( active >= JOBS )); then
    wait -n   # espera a que termine cualquiera
    active=$((active-1))
  fi
done

# espera a que terminen todos
wait
echo "✔ Listo. Revisa dataRaw/ y los logs por seed."

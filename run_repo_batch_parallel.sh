
#!/usr/bin/env bash
# Lanza varias simID en paralelo.
# Variables (puedes sobreescribir por env o editando aquí):
#   INPUT_FACTOR  (default 1.2)   -> intensidad del estímulo Poisson
#   STIMULUS      (default 1)     -> 1 = pulso 100 ms
#   START         (default 1)     -> simID inicial
#   END           (default 39)    -> simID final (inclusive)
#   JOBS          (default 4)     -> nº de procesos en paralelo (≈ nº de núcleos libres)

set -euo pipefail

INPUT_FACTOR="${INPUT_FACTOR:-1.2}"
STIMULUS="${STIMULUS:-1}"
START="${START:-1}"
END="${END:-39}"
JOBS="${JOBS:-4}"

echo "==> input_factor=${INPUT_FACTOR}  stimulus=${STIMULUS}  simID=${START}..${END}  jobs=${JOBS}"

# Evitar sobre-suscripción de hilos BLAS/OpenMP dentro de cada proceso Python
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p srcSim/annarchy_folders logs

# Lanza seeds en paralelo, salta las que ya existen en dataRaw/
seq "${START}" "${END}" | xargs -I{} -P "${JOBS}" bash -lc '
  sid="{}"
  if compgen -G "dataRaw/*__${sid}.*" >/dev/null; then
    echo "→ simID=${sid} (skip: ya existe algo en dataRaw/*__${sid}.*)"
  else
    echo "→ simID=${sid} (run)"
    ( cd srcSim && python simulations.py "'"${INPUT_FACTOR}"'" "'"${STIMULUS}"'" "${sid}" ) > "logs/sim_${sid}.out" 2>&1
    echo "✓ simID=${sid} (done)"
  fi
'
echo "✔ Batch terminado. Logs por seed en ./logs/sim_<id>.out"

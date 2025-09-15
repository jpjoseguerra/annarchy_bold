#!/usr/bin/env bash
# Ejecuta seeds en paralelo con entorno de compilación preparado.
# Uso:
#   ./run_40_parallel.sh [input_factor] [stimulus] [seed_ini] [seed_fin] [concurrencia]
# Ejemplos:
#   ./run_40_parallel.sh 1.2 3 0 39 3   # bloque sostenido ×1.2, seeds 0..39, 3 en paralelo
#   ./run_40_parallel.sh 5   1 0 39 3   # pulso 100ms ×5, seeds 0..39, 3 en paralelo

set -euo pipefail

INPUT_FACTOR="${1:-1.2}"
STIMULUS="${2:-3}"
SEED_START="${3:-0}"
SEED_END="${4:-39}"
JOBS="${5:-3}"

# Para no saturar CPU con múltiples procesos
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Comprobaciones básicas
[[ -f "srcSim/simulations.py" ]] || { echo "✖ Ejecuta desde la RAÍZ del repo."; exit 1; }
[[ -f "fix_annarchy_env.sh"   ]] || { echo "✖ Falta fix_annarchy_env.sh en la raíz."; exit 1; }

mkdir -p srcSim/annarchy_folders logs

# xargs lanza en paralelo de a $JOBS procesos; cada job hace 'source' del fix
seq "${SEED_START}" "${SEED_END}" | xargs -n 1 -P "${JOBS}" -I {} bash -lc '
  cd srcSim
  source ../fix_annarchy_env.sh
  echo "→ simID={}  (input_factor='"${INPUT_FACTOR}"', stimulus='"${STIMULUS}"')"
  python simulations.py '"${INPUT_FACTOR}"' '"${STIMULUS}"' "{}" > ../logs/sim_{}.log 2>&1
'

# Verificación: contar archivos generados
IF_TAG="${INPUT_FACTOR//./_}"   # 1.2 -> 1_2
count=$(ls dataRaw | grep -o "recordingsB_${IF_TAG}_${STIMULUS}__.*\.npy" | wc -l || true)
echo "✔ Generados ${count} archivos para recordingsB_${IF_TAG}_${STIMULUS}__*.npy"

# Reintento sólo de faltantes (por si algún job cayó)
present=$(ls dataRaw | grep -oP "recordingsB_${IF_TAG}_${STIMULUS}__\\K\\d+(?=\\.npy)" | sort -n | uniq || true)
for sid in $(seq "${SEED_START}" "${SEED_END}"); do
  if ! echo "$present" | grep -qx "$sid"; then
    echo "↻ Re-ejecutando simID=$sid"
    ( cd srcSim && source ../fix_annarchy_env.sh && python simulations.py '"${INPUT_FACTOR}"' '"${STIMULUS}"' "$sid" > ../logs/sim_${sid}.log 2>&1 )
  fi
done

echo "✅ Batch terminado."

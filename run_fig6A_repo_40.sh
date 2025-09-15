#!/usr/bin/env bash
# Ejecuta las 40 simulaciones spiking del repositorio para la Fig. 6 (Modelo A–F).
# Args usados por el repo: simulations.py <INPUT_FACTOR> <STIMULUS> <simID>
#   INPUT_FACTOR=5 (pulso x5), STIMULUS=1 (pulso 100 ms), simID=0..39

set -euo pipefail

# 0) Asegúrate de ejecutarlo desde la RAÍZ del repo (donde existen srcSim/ y srcAna/)
if [[ ! -f "srcSim/simulations.py" ]] || [[ ! -f "srcAna/BOLDfromDifferentSources_ANA.py" ]]; then
  echo "✖ Debes ejecutar este script desde la raíz del repositorio."
  exit 1
fi

# 1) Parámetros del experimento (paper/repositorio)
INPUT_FACTOR=5
STIMULUS=1
SEEDS=40   # simID = 0..39

echo "==> [1/2] Corriendo ${SEEDS} simulaciones spiking del repo..."
for sid in $(seq 0 $((SEEDS-1))); do
  echo "   → simID=$sid"
  python srcSim/simulations.py "${INPUT_FACTOR}" "${STIMULUS}" "${sid}"
done

# 2) (Opcional) Generar las figuras del repo (incluye Fig. 6). Activa con DO_PLOTS=1
if [[ "${DO_PLOTS:-0}" == "1" ]]; then
  echo "==> [2/2] Generando figuras del repositorio..."
  python srcAna/BOLDfromDifferentSources_ANA.py
fi

echo "✔ Listo. Datos crudos en dataRaw/ (y figuras en results/ si DO_PLOTS=1)."

#!/usr/bin/env bash
# Corre un rango de simID con los scripts del repo.
# Variables (puedes sobreescribir por env):
#   INPUT_FACTOR  (default 1.2)   -> intensidad del estímulo Poisson
#   STIMULUS      (default 1)     -> 1 = pulso 100 ms
#   START         (default 1)     -> simID inicial
#   END           (default 39)    -> simID final (inclusive)

set -euo pipefail

INPUT_FACTOR="${INPUT_FACTOR:-1.2}"
STIMULUS="${STIMULUS:-1}"
START="${START:-1}"
END="${END:-39}"

echo "==> input_factor=${INPUT_FACTOR}  stimulus=${STIMULUS}  simID=${START}..${END}"

# Compilación local que espera el repo (desde srcSim/)
mkdir -p srcSim/annarchy_folders

pushd srcSim >/dev/null
for sid in $(seq "${START}" "${END}"); do
  # Si ya existe algún archivo de dataRaw para ese sid, saltamos (idempotente)
  if compgen -G "../dataRaw/*__${sid}.*" >/dev/null; then
    echo "   → simID=${sid} (ya existe algo en dataRaw/*__${sid}.*, salto)"
    continue
  fi
  echo "   → simID=${sid}"
  python simulations.py "${INPUT_FACTOR}" "${STIMULUS}" "${sid}"
done
popd >/dev/null

echo "✔ Hecho. Revisa dataRaw/ para los artefactos por seed."

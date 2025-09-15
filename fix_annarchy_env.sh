# === Preparar entorno de compilación ANNarchy (pyenv 3.7) ===
INC=$(python - <<'PY'
import sysconfig; print(sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_paths().get("include",""))
PY
)
LIBD=$(python - <<'PY'
import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")
PY
)
LIBPL=$(python - <<'PY'
import sysconfig; print(sysconfig.get_config_var("LIBPL") or "")
PY
)

export CPATH="${INC}:${CPATH:-}"
export LIBRARY_PATH="${LIBD}:${LIBPL}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LIBD}:${LIBPL}:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-L${LIBD} -L${LIBPL} ${LDFLAGS:-}"

for D in "$LIBD" "$LIBPL"; do
  if [ -f "$D/libpython3.7m.so.1.0" ] && [ ! -f "$D/libpython3.7m.so" ]; then
    ln -sf "$D/libpython3.7m.so.1.0" "$D/libpython3.7m.so"
  fi
done

[ -f "$INC/Python.h" ] || { echo "✖ No existe $INC/Python.h"; return 1; }
echo "✔ Python.h en: $INC"

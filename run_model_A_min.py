# -*- coding: utf-8 -*-
# Modelo A (Figura 6): Balloon clásico con I_CBF <- syn (actividad sináptica total normalizada)
# Protocolo: 2 s init (sin monitor) -> 5 s baseline (normalización) -> pulso 100 ms (factor 5) -> post hasta 25 s

import os
import numpy as np
import matplotlib.pyplot as plt

from ANNarchy import Neuron, Population, setup, simulate, clear, get_time, compile
from ANNarchy.extensions.bold import BoldMonitor, balloon_RN

# --- (opcional) pequeños fixes de include/lib por si ANNarchy fue instalado localmente ---
import sysconfig, pathlib, os as _os
inc   = sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_paths().get("include")
libd  = sysconfig.get_config_var("LIBDIR") or ""
libpl = sysconfig.get_config_var("LIBPL") or ""
if inc: _os.environ["CPATH"] = f"{inc}:" + _os.environ.get("CPATH", "")
for k, v in [("LDFLAGS", libd), ("LIBRARY_PATH", f"{libd}:{libpl}"), ("LD_LIBRARY_PATH", f"{libd}:{libpl}")]:
    if v and _os.environ.get(k, "") == "": _os.environ[k] = v
for D in [libd, libpl]:
    if D:
        so10 = pathlib.Path(D) / "libpython3.7m.so.1.0"
        so   = pathlib.Path(D) / "libpython3.7m.so"
        if so10.exists() and not so.exists():
            try: so.symlink_to(so10)
            except Exception: pass
# -----------------------------------------------------------------------------------------

# ----------------------------
# 1) Parámetros de simulación
# ----------------------------
DT = 1.0  # ms
INIT_MS     = 2000   # warm-up SIN monitor (paper: 2 s antes de baseline)
BASELINE_MS = 5000   # baseline para normalización del monitor (paper: 5 s)
PULSE_MS    = 100
TOTAL_MS    = 25000  # 25 s total (paper)
POST_MS     = TOTAL_MS - (INIT_MS + BASELINE_MS + PULSE_MS)

INPUT_BASE   = 0.20
INPUT_FACTOR = 5.0   # pulso x5
TAU_SYN      = 30.0  # constante de 'actividad sináptica' (ms)
N_E, N_I     = 400, 100  # 4:1 como en el microcircuito

SAVE_DIR = os.path.join("results", "ModelA_min")
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------
# 2) Neuronas "rate" con variable 'syn' (LPF)
#    syn ~ actividad sináptica total normalizada
# ------------------------------------------
clear()
setup(dt=DT)

rate_unit = Neuron(
    parameters="""
        tau_syn = 30.0 : population
        I = 0.0        : population   # "entrada sináptica efectiva"
    """,
    equations="""
        dsyn/dt = (-syn + I)/tau_syn : init=0, min=0
    """
)
rate_unit.tau_syn = TAU_SYN

corE = Population(geometry=N_E, neuron=rate_unit, name="corE")
corI = Population(geometry=N_I, neuron=rate_unit, name="corI")
corE.I = INPUT_BASE
corI.I = INPUT_BASE

# -------------------------------------------------
# 3) Monitor BOLD del Modelo A: I_CBF <- 'syn'
#     - Balloon clásico (balloon_RN)
#     - Normalización de línea base (BASELINE_MS)
# -------------------------------------------------
bold_A = BoldMonitor(
    populations=[corE, corI],         # ROI = corE + corI (como en Fig. 6)
    bold_model=balloon_RN,            # Balloon clásico
    mapping={"I_CBF": "syn"},         # *** Modelo A ***
    normalize_input=BASELINE_MS,      # baseline normalization
    recorded_variables=["I_CBF", "BOLD"]
)

compile()

# ----------------------------
# 4) Secuencia de simulación
# ----------------------------
# 4.1) Warm-up SIN monitor (simula 2 s antes de empezar a grabar/normalizar)
simulate(INIT_MS)

# 4.2) Arranca el monitor y estima baseline por BASELINE_MS
bold_A.start()
simulate(BASELINE_MS)

# 4.3) Pulso (100 ms, factor 5)
corE.I = INPUT_BASE * INPUT_FACTOR
corI.I = INPUT_BASE * INPUT_FACTOR
simulate(PULSE_MS)

# 4.4) Post-estímulo hasta 25 s totales
corE.I = INPUT_BASE
corI.I = INPUT_BASE
simulate(POST_MS)

# ----------------------------
# 5) Guardar y graficar
# ----------------------------
t = np.arange(int(get_time())) * DT
icbf = bold_A.get("I_CBF")   # (time, n_pops) -> promediamos ROI
bold = bold_A.get("BOLD")

if getattr(icbf, "ndim", 1) == 2: icbf = icbf.mean(axis=1)
if getattr(bold, "ndim", 1) == 2: bold = bold.mean(axis=1)

np.save(os.path.join(SAVE_DIR, "t_ms.npy"), t)
np.save(os.path.join(SAVE_DIR, "I_CBF.npy"), icbf)
np.save(os.path.join(SAVE_DIR, "BOLD.npy"), bold)

# Figuras
# (1) BOLD
plt.figure()
plt.title("Modelo A — BOLD (balloon_RN, I_CBF ← syn)")
plt.plot(t/1000.0, bold, label="BOLD (Δrel.)")
plt.axvspan((INIT_MS+BASELINE_MS)/1000.0,
            (INIT_MS+BASELINE_MS+PULSE_MS)/1000.0, alpha=0.15, label="pulso")
plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "BOLD_ModelA.png"), dpi=150)

# (2) I_CBF normalizado
plt.figure()
plt.title("Modelo A — I_CBF (normalizado)")
plt.plot(t/1000.0, icbf, label="I_CBF (norm)")
plt.axvspan((INIT_MS+BASELINE_MS)/1000.0,
            (INIT_MS+BASELINE_MS+PULSE_MS)/1000.0, alpha=0.15, label="pulso")
plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "ICBF_ModelA.png"), dpi=150)

print("OK. Salida ->", SAVE_DIR)

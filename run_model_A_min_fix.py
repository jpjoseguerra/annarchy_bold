# -*- coding: utf-8 -*-
# Modelo A: balloon_RN con I_CBF <- syn (LPF de r)

import os
import numpy as np
import matplotlib.pyplot as plt

from ANNarchy import Neuron, Population, setup, simulate, clear, get_time, compile
from ANNarchy.extensions.bold import BoldMonitor, balloon_RN

# ----- Parámetros -----
DT = 1.0
INIT_MS, BASELINE_MS, PULSE_MS, TOTAL_MS = 2000, 5000, 100, 25000
POST_MS = TOTAL_MS - (INIT_MS + BASELINE_MS + PULSE_MS)
INPUT_BASE, INPUT_FACTOR = 0.20, 5.0
TAU_R, TAU_SYN = 50.0, 30.0
N_E, N_I = 400, 100
SAVE_DIR = os.path.join("results", "ModelA"); os.makedirs(SAVE_DIR, exist_ok=True)

# ----- Modelo -----
clear(); setup(dt=DT)
rate_unit = Neuron(
    parameters="""tau_r=50.0:population; tau_syn=30.0:population; I=0.0:population""",
    equations="""dr/dt=(-r+I)/tau_r : init=0, min=0
                 dsyn/dt=(-syn+r)/tau_syn : init=0, min=0"""
)
rate_unit.tau_r = TAU_R; rate_unit.tau_syn = TAU_SYN
corE = Population(N_E, rate_unit, name="corE")
corI = Population(N_I, rate_unit, name="corI")
corE.I = INPUT_BASE; corI.I = INPUT_BASE

bold_A = BoldMonitor(
    populations=[corE, corI],
    bold_model=balloon_RN,
    mapping={"I_CBF": "syn"},
    normalize_input=BASELINE_MS,
    recorded_variables=["I_CBF", "BOLD"]
)

compile()

# ----- Simulación -----
simulate(INIT_MS)              # warm-up (sin monitor)
bold_A.start()                 # aquí empieza a grabar el monitor
simulate(BASELINE_MS)          # baseline p/ normalización
corE.I = INPUT_BASE*INPUT_FACTOR; corI.I = INPUT_BASE*INPUT_FACTOR
simulate(PULSE_MS)             # pulso 100 ms
corE.I = INPUT_BASE; corI.I = INPUT_BASE
simulate(POST_MS)              # post hasta TOTAL_MS

# ----- Datos -----
T_tot = int(get_time())                     # 25000
t_all = np.arange(T_tot) * DT               # 0..24999 ms
# Eje temporal alineado con el monitor (desde INIT_MS)
t_rec_ms = t_all[INIT_MS:]                  # 2000..24999 ms (23000 muestras)

icbf = bold_A.get("I_CBF")
bold = bold_A.get("BOLD")
if getattr(icbf, "ndim", 1) == 2: icbf = icbf.mean(axis=1)
if getattr(bold, "ndim", 1) == 2: bold = bold.mean(axis=1)

# Sanity check: longitudes iguales
assert len(t_rec_ms) == len(bold) == len(icbf), \
    f"Len mismatch: t={len(t_rec_ms)} bold={len(bold)} icbf={len(icbf)}"

# Guardado (eje temporal que corresponde a lo grabado por el monitor)
np.save(os.path.join(SAVE_DIR, "t_ms.npy"), t_rec_ms)
np.save(os.path.join(SAVE_DIR, "I_CBF.npy"), icbf)
np.save(os.path.join(SAVE_DIR, "BOLD.npy"), bold)

# ----- Gráficas -----
t_s = t_rec_ms / 1000.0
pulso_ini = (INIT_MS + BASELINE_MS)/1000.0
pulso_fin = (INIT_MS + BASELINE_MS + PULSE_MS)/1000.0

plt.figure()
plt.title("Modelo A — BOLD (balloon_RN, I_CBF ← syn)")
plt.plot(t_s, bold, label="BOLD (Δrel.)")
plt.axvspan(pulso_ini, pulso_fin, alpha=0.15, label="pulso")
plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "BOLD_ModelA.png"), dpi=150)

plt.figure()
plt.title("Modelo A — I_CBF (normalizado)")
plt.plot(t_s, icbf, label="I_CBF (norm)")
plt.axvspan(pulso_ini, pulso_fin, alpha=0.15, label="pulso")
plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "ICBF_ModelA.png"), dpi=150)

print("OK. Salida ->", SAVE_DIR)

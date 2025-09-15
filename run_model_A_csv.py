# -*- coding: utf-8 -*-
# Modelo A (Figura 6):
# Balloon clásico (balloon_RN) con I_CBF <- syn (LPF de r)
# Secuencia: 2 s warm-up (sin monitor) -> 5 s baseline (normalización) -> pulso 100 ms (x5) -> post hasta 25 s (total 25 s)

import os, json
import numpy as np
import matplotlib.pyplot as plt

from ANNarchy import Neuron, Population, setup, simulate, clear, get_time, compile
from ANNarchy.extensions.bold import BoldMonitor, balloon_RN

# ======================
# 1) Parámetros
# ======================
DT = 1.0  # ms
INIT_MS     = 2000    # warm-up SIN monitor
BASELINE_MS = 5000    # baseline para normalización
PULSE_MS    = 100
TOTAL_MS    = 25000   # total 25 s
POST_MS     = TOTAL_MS - (INIT_MS + BASELINE_MS + PULSE_MS)

INPUT_BASE   = 0.20
INPUT_FACTOR = 5.0
TAU_R        = 50.0
TAU_SYN      = 30.0
N_E, N_I     = 400, 100  # 4:1

OUT_DIR = os.path.join("results", "ModelA_csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# 2) Modelo de neurona (rate)
# ======================
clear()
setup(dt=DT)

rate_unit = Neuron(
    parameters="""
        tau_r = 50.0   : population
        tau_syn = 30.0 : population
        I = 0.0        : population
    """,
    equations="""
        dr/dt   = (-r   + I)/tau_r   : init=0, min=0
        dsyn/dt = (-syn + r)/tau_syn : init=0, min=0
    """
)
rate_unit.tau_r = TAU_R
rate_unit.tau_syn = TAU_SYN

corE = Population(geometry=N_E, neuron=rate_unit, name="corE")
corI = Population(geometry=N_I, neuron=rate_unit, name="corI")
corE.I = INPUT_BASE
corI.I = INPUT_BASE

# ======================
# 3) Monitor BOLD (Modelo A)
# ======================
bold_A = BoldMonitor(
    populations=[corE, corI],           # ROI = corE + corI
    bold_model=balloon_RN,              # Balloon clásico
    mapping={"I_CBF": "syn"},           # *** Modelo A ***
    normalize_input=BASELINE_MS,        # normaliza usando los primeros 5 s tras start()
    recorded_variables=["I_CBF", "BOLD"]
)

compile()

# ======================
# 4) Simulación
# ======================
# Warm-up (sin monitor)
simulate(INIT_MS)

# Arranca monitor y calcula baseline (5 s)
bold_A.start()
simulate(BASELINE_MS)

# Pulso 100 ms (x5)
corE.I = INPUT_BASE * INPUT_FACTOR
corI.I = INPUT_BASE * INPUT_FACTOR
simulate(PULSE_MS)

# Post hasta 25 s totales
corE.I = INPUT_BASE
corI.I = INPUT_BASE
simulate(POST_MS)

# ======================
# 5) Recolección de datos
# ======================
T_tot = int(get_time())                     # 25000
t_all = np.arange(T_tot) * DT               # 0..24999 ms
t_rec_ms = t_all[INIT_MS:]                  # eje temporal del monitor (23000 muestras)

I_CBF = bold_A.get("I_CBF")                 # (time, n_pops) o (time,)
BOLD  = bold_A.get("BOLD")

# Promedio ROI (E+I) si vienen por población
if getattr(I_CBF, "ndim", 1) == 2: I_CBF = I_CBF.mean(axis=1)
if getattr(BOLD,  "ndim", 1) == 2: BOLD  = BOLD.mean(axis=1)

# Alineación defensiva
n = min(len(t_rec_ms), len(I_CBF), len(BOLD))
t_rec_ms = t_rec_ms[:n]
I_CBF    = I_CBF[:n]
BOLD     = BOLD[:n]

# ======================
# 6) Exportar: NPY + CSV + PNG + METADATA + METRICS
# ======================
# NPY
np.save(os.path.join(OUT_DIR, "t_ms.npy"),   t_rec_ms)
np.save(os.path.join(OUT_DIR, "I_CBF.npy"),  I_CBF)
np.save(os.path.join(OUT_DIR, "BOLD.npy"),   BOLD)

# CSV (intenta pandas; si no, usa numpy)
def save_csv(path, header, cols):
    try:
        import pandas as pd
        df = pd.DataFrame({h:c for h, c in zip(header, cols)})
        df.to_csv(path, index=False)
    except Exception:
        arr = np.column_stack(cols)
        np.savetxt(path, arr, delimiter=",", header=",".join(header), comments="", fmt="%.9g")

# Series en un único CSV (fácil de compartir)
save_csv(
    os.path.join(OUT_DIR, "timeseries_modelA.csv"),
    header=["t_ms", "BOLD", "I_CBF"],
    cols=[t_rec_ms, BOLD, I_CBF]
)

# PNG de las curvas
t_s = t_rec_ms / 1000.0
pulso_ini = (INIT_MS + BASELINE_MS)/1000.0
pulso_fin = (INIT_MS + BASELINE_MS + PULSE_MS)/1000.0

plt.figure()
plt.title("Modelo A — BOLD (balloon_RN, I_CBF ← syn)")
plt.plot(t_s, BOLD, label="BOLD (Δrel.)")
plt.axvspan(pulso_ini, pulso_fin, alpha=0.15, label="pulso")
plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "BOLD_ModelA.png"), dpi=150)

plt.figure()
plt.title("Modelo A — I_CBF (normalizado)")
plt.plot(t_s, I_CBF, label="I_CBF (norm)")
plt.axvspan(pulso_ini, pulso_fin, alpha=0.15, label="pulso")
plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ICBF_ModelA.png"), dpi=150)

# METADATA (JSON)
meta = {
    "model": "A",
    "balloon_model": "balloon_RN",
    "mapping": {"I_CBF": "syn"},
    "dt_ms": DT,
    "durations_ms": {
        "warmup": INIT_MS, "baseline": BASELINE_MS, "pulse": PULSE_MS,
        "post": POST_MS, "total": TOTAL_MS
    },
    "input": {"baseline": INPUT_BASE, "factor": INPUT_FACTOR},
    "neuron": {"tau_r": TAU_R, "tau_syn": TAU_SYN, "N_E": N_E, "N_I": N_I}
}
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

# METRICS (CSV)
def seg_stats(x, a, b):
    s = x[a:b]
    return dict(mean=float(s.mean()), std=float(s.std()), min=float(s.min()), max=float(s.max()))

# Índices de segmentos (en el eje del monitor)
b0 = int(BASELINE_MS/DT)
p1 = b0
p2 = b0 + int(PULSE_MS/DT)
q1 = p2
q2 = len(t_rec_ms)

B_base = seg_stats(BOLD, 0, b0)
B_pulse= seg_stats(BOLD, p1, p2)
B_post = seg_stats(BOLD, q1, q2)
I_base = seg_stats(I_CBF, 0, b0)
I_pulse= seg_stats(I_CBF, p1, p2)
I_post = seg_stats(I_CBF, q1, q2)

metrics_rows = [
    ["dt_ms", DT],
    ["t_total_ms_recorded", float(t_rec_ms[-1] - t_rec_ms[0]) + DT],
    ["BOLD_baseline_mean", B_base["mean"]],
    ["BOLD_baseline_std",  B_base["std"]],
    ["BOLD_peak_during_pulse", B_pulse["max"]],
    ["BOLD_peak_minus_base",    B_pulse["max"] - B_base["mean"]],
    ["BOLD_min_post",           B_post["min"]],
    ["BOLD_min_minus_base",     B_post["min"] - B_base["mean"]],
    ["ICBF_baseline_mean", I_base["mean"]],
    ["ICBF_baseline_std",  I_base["std"]],
    ["ICBF_mean_during_pulse", I_pulse["mean"]],
    ["ICBF_mean_minus_base",   I_pulse["mean"] - I_base["mean"]],
]

# Guardar métricas
try:
    import pandas as pd
    pd.DataFrame(metrics_rows, columns=["metric", "value"]).to_csv(
        os.path.join(OUT_DIR, "metrics_modelA.csv"),
        index=False
    )
except Exception:
    with open(os.path.join(OUT_DIR, "metrics_modelA.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in metrics_rows:
            f.write(f"{k},{v}\n")

print("OK. Salida ->", OUT_DIR)
print(" - timeseries_modelA.csv (t_ms, BOLD, I_CBF)")
print(" - metrics_modelA.csv")
print(" - BOLD_ModelA.png / ICBF_ModelA.png")
print(" - metadata.json / *.npy")

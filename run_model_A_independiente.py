# run_model_A_independiente.py
# Modelo A (Figura 6): balloon_RN con I_CBF <- x  (x = LPF de r)
# Protocolo: baseline 2000 ms -> pulso 100 ms (factor 5) -> post 2000 ms

import os
import numpy as np
import matplotlib.pyplot as plt

from ANNarchy import Neuron, Population, setup, simulate, clear, get_time, compile
from ANNarchy.extensions.bold import BoldMonitor, balloon_RN

# --- Fix de includes y libs para ANNarchy sin instalar nada ---
import sysconfig, pathlib
inc   = sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_paths().get("include")
libd  = sysconfig.get_config_var("LIBDIR") or ""
libpl = sysconfig.get_config_var("LIBPL") or ""
if inc:
    os.environ["CPATH"] = f"{inc}:" + os.environ.get("CPATH", "")
if libd or libpl:
    os.environ["LDFLAGS"]         = f"-L{libd} -L{libpl} " + os.environ.get("LDFLAGS", "")
    os.environ["LIBRARY_PATH"]    = f"{libd}:{libpl}:" + os.environ.get("LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{libd}:{libpl}:" + os.environ.get("LD_LIBRARY_PATH", "")
# Si solo existe libpython3.7m.so.1.0, crea alias libpython3.7m.so (opcional)
for D in [libd, libpl]:
    if D:
        so10 = pathlib.Path(D) / "libpython3.7m.so.1.0"
        so   = pathlib.Path(D) / "libpython3.7m.so"
        if so10.exists() and not so.exists():
            try:
                so.symlink_to(so10)
            except Exception:
                pass
# --------------------------------------------------------------

# ----------------------------
# 1) Parámetros globales
# ----------------------------
DT = 1.0                # ms
BASELINE_MS = 2000
PULSE_MS = 100
POST_MS = 2000
INPUT_BASE = 0.20
INPUT_FACTOR = 5.0
TAU_R = 50.0
TAU_SYN = 30.0
N_E = 400
N_I = 100

SAVE_DIR = os.path.join("results", "ModelA")
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# 2) Modelo de neurona (rate)
#    dr/dt = (-r + I)/tau_r
#    dx/dt = (-x + r)/tau_syn
# ----------------------------
clear()
setup(dt=DT)

rate_unit = Neuron(
    parameters="""
        tau_r = 50.0 : population
        tau_syn = 30.0 : population
        I = 0.0 : population
    """,
    equations="""
        dr/dt = (-r + I) / tau_r     : init=0, min=0
        dx/dt = (-x + r) / tau_syn   : init=0, min=0
    """
)
# fija valores de parámetros
rate_unit.tau_r = TAU_R
rate_unit.tau_syn = TAU_SYN

# Poblaciones E/I
corE = Population(geometry=N_E, neuron=rate_unit, name="corE")
corI = Population(geometry=N_I, neuron=rate_unit, name="corI")
corE.I = INPUT_BASE
corI.I = INPUT_BASE

# ----------------------------
# 3) Monitor BOLD: Modelo A (I_CBF <- x)
# ----------------------------
bold_A = BoldMonitor(
    populations=[corE, corI],
    bold_model=balloon_RN,
    mapping={"I_CBF": "x"},
    normalize_input=BASELINE_MS,
    recorded_variables=["I_CBF", "BOLD"]
)

compile()

# ----------------------------
# 4) Simulación por bloques
# ----------------------------
bold_A.start()

# Baseline
simulate(BASELINE_MS)

# Pulso
corE.I = INPUT_BASE * INPUT_FACTOR
corI.I = INPUT_BASE * INPUT_FACTOR
simulate(PULSE_MS)

# Post
corE.I = INPUT_BASE
corI.I = INPUT_BASE
simulate(POST_MS)

# ----------------------------
# 5) Recuperar y guardar resultados
# ----------------------------
t = np.arange(int(get_time())) * DT
i_cbf = bold_A.get("I_CBF")   # (time, n_pops)
bold  = bold_A.get("BOLD")    # (time, n_pops)

# Promedio del ROI (E e I)
if hasattr(i_cbf, "ndim") and i_cbf.ndim == 2:
    i_cbf = i_cbf.mean(axis=1)
if hasattr(bold, "ndim") and bold.ndim == 2:
    bold = bold.mean(axis=1)

np.save(os.path.join(SAVE_DIR, "t_ms.npy"), t)
np.save(os.path.join(SAVE_DIR, "BOLD.npy"), bold)
np.save(os.path.join(SAVE_DIR, "I_CBF.npy"), i_cbf)

# ----------------------------
# 6) Gráficas
# ----------------------------
plt.figure()
plt.title("Modelo A — BOLD (balloon_RN, I_CBF <- x)")
plt.plot(t, bold, label="BOLD (rel.)")
plt.xlabel("Tiempo [ms]"); plt.ylabel("ΔBOLD (rel.)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "BOLD_ModelA.png"), dpi=150)

plt.figure()
plt.title("Input hemodinámico — I_CBF (norm)")
plt.plot(t, i_cbf, label="I_CBF (norm)")
plt.xlabel("Tiempo [ms]"); plt.ylabel("I_CBF (a.u.)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "ICBF_ModelA.png"), dpi=150)

print("OK. Salida ->", SAVE_DIR)


# check_modelA_results.py
import os, numpy as np

D = "results/ModelA"  # ajusta si usaste otra carpeta
t     = np.load(os.path.join(D, "t_ms.npy"))
bold  = np.load(os.path.join(D, "BOLD.npy"))
icbf  = np.load(os.path.join(D, "I_CBF.npy"))

dt = float(np.median(np.diff(t)))
BASELINE_MS, PULSE_MS, POST_MS = 2000, 100, 2000

b0  = int(BASELINE_MS/dt)
p1  = b0
p2  = int((BASELINE_MS+PULSE_MS)/dt)
q1  = p2
q2  = int((BASELINE_MS+PULSE_MS+POST_MS)/dt)

def seg_stats(x, a, b):
    s = x[a:b]
    return dict(mean=float(s.mean()), std=float(s.std()),
                min=float(s.min()), max=float(s.max()))

B_base = seg_stats(bold, 0, b0)
B_pulse= seg_stats(bold, p1, p2)
B_post = seg_stats(bold, q1, q2)

I_base = seg_stats(icbf, 0, b0)
I_pulse= seg_stats(icbf, p1, p2)
I_post = seg_stats(icbf, q1, q2)

print("dt(ms)=", dt, "T(ms)=", float(t[-1]))
print("BOLD  baseline mean±std =", B_base["mean"], "±", B_base["std"])
print("BOLD  peak during pulse =", B_pulse["max"], " (Δ vs base =", B_pulse["max"]-B_base["mean"], ")")
print("BOLD  min post (undershoot) =", B_post["min"], " (Δ vs base =", B_post["min"]-B_base["mean"], ")")
print("I_CBF baseline mean±std =", I_base["mean"], "±", I_base["std"])
print("I_CBF mean during pulse =", I_pulse["mean"], " (Δ vs base =", I_pulse["mean"]-I_base["mean"], ")")

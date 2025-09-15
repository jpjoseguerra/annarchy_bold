# -*- coding: utf-8 -*-
"""
Tren de pulsos Fig.6A (~600 s) concatenando EXCLUSIVAMENTE segmentos
de pulso ×5 (stimulus=1) del repositorio Hamkerlab.
- Recorta cada segmento a 25 s exactos (para evitar solapes y “líneas diagonales”).
- Cose segmentos de 25 s contiguos (ITI=25 s) → ~24 pulsos ≈ 600 s totales.
- Exporta CSV, PNGs y métricas; copia los .npy crudos por segmento (trazabilidad).

Uso:
  source ./fix_annarchy_env.sh   # si ANNarchy necesita headers/libs
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  python run_fig6A_pulse_train_600s.py

Opcional:
  - Para variabilidad entre pulsos, activar USE_SEGMENT_SEED=True (cada segmento usa seed=seg_idx).
"""

import os, sys, json, shutil, subprocess, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Parámetros del tren
# =========================
INPUT_FACTOR = 5.0      # Fig. 6A
STIMULUS     = 1        # pulso 100 ms
SEED_FIXED   = 0        # misma seed en todos los segmentos (pulsos idénticos)
USE_SEGMENT_SEED = False  # True: usa seed = índice de segmento (pulsos con variabilidad)

TARGET_T_S   = 600.0    # duración total deseada
SEG_T_S      = 25.0     # duración útil por segmento (recorte)
LKEEP_MS     = int(SEG_T_S * 1000)  # 25 s -> 25000 muestras
ITI_S        = 25.0     # separación entre onsets (contiguo si 25 s)
PULSE_ONSET_LOCAL_S = 5.0   # en el protocolo del repo, el pulso empieza a ~5 s
PULSE_DUR_S         = 0.1   # 100 ms

# =========================
# Salidas
# =========================
OUTDIR = os.path.join("results", "fig6A_pulse_train_600s")
SEGR   = os.path.join(OUTDIR, "seg_raw")
os.makedirs(SEGR, exist_ok=True)

# Nombres esperados en dataRaw (del repo) — el seed forma parte del nombre
TAG_IF  = str(INPUT_FACTOR).replace(".", "_")  # 5.0 -> "5_0"

def rec_path_for_seed(seed: int) -> str:
    return os.path.join(
        "dataRaw",
        f"simulations_BOLDfromDifferentSources_recordingsB_{TAG_IF}_{STIMULUS}__{seed}.npy"
    )

def par_path_for_seed(seed: int) -> str:
    return os.path.join(
        "dataRaw",
        f"simulations_BOLDfromDifferentSources_simParams_{TAG_IF}_{STIMULUS}__{seed}.npy"
    )

# =========================
# Utilidades
# =========================
def arr1(x):
    if isinstance(x, np.ndarray):
        return x.mean(axis=1) if x.ndim == 2 else x.ravel()
    return None

def run_one_segment(seg_idx: int):
    """
    Ejecuta UN segmento (pulso ×5, stim=1) y devuelve (t_local_ms, B, I) recortados a 25 s.
    """
    seed = seg_idx if USE_SEGMENT_SEED else SEED_FIXED

    # Lanza el simulador dentro de srcSim (las rutas relativas del repo dependen de eso)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = [sys.executable, "simulations.py", str(INPUT_FACTOR), str(STIMULUS), str(seed)]
    subprocess.run(cmd, check=True, env=env, cwd="srcSim")

    # Pequeño margen por timestamps en WSL
    time.sleep(0.1)

    rec_abs = rec_path_for_seed(seed)
    par_abs = par_path_for_seed(seed)

    if not os.path.exists(rec_abs):
        raise SystemExit(f"✖ No encontré {rec_abs} tras la simulación.")

    # Copias crudas con sufijo de segmento (para trazabilidad)
    base_rec = os.path.basename(rec_abs)
    base_par = os.path.basename(par_abs) if os.path.exists(par_abs) else None

    rec_dst = os.path.join(SEGR, base_rec.replace(f"__{seed}.npy", f"__{seed}_seg{seg_idx:02d}.npy"))
    shutil.copy2(rec_abs, rec_dst)
    if base_par:
        par_dst = os.path.join(SEGR, base_par.replace(f"__{seed}.npy", f"__{seed}_seg{seg_idx:02d}.npy"))
        shutil.copy2(par_abs, par_dst)

    # Carga Modelo A (1;BOLD, 1;I_CBF) y recorta a 25 s
    d = np.load(rec_abs, allow_pickle=True).item()
    B = arr1(d.get("1;BOLD"))
    I = arr1(d.get("1;I_CBF"))
    if B is None:
        raise SystemExit(f"✖ No hallé '1;BOLD' en {rec_abs}")
    if I is None:
        I = np.zeros_like(B)

    if len(B) < LKEEP_MS:
        raise SystemExit(f"✖ Segmento demasiado corto ({len(B)} ms). Esperaba ≥{LKEEP_MS} ms.")
    B = B[:LKEEP_MS]
    I = I[:LKEEP_MS]

    # Tiempo local del segmento: 0..25 s (dt=1 ms)
    t_local = np.arange(0.0, float(LKEEP_MS), 1.0, dtype=float)
    return t_local, B, I

def stitch(segments):
    """
    Concatena segmentos recortados de 25 s, desplazando cada uno por ITI_S (25 s).
    Asegura monotonicidad estricta del tiempo.
    """
    T, B, I = [], [], []
    t_off = 0.0
    for (t,b,i) in segments:
        T.append(t + t_off); B.append(b); I.append(i)
        t_off += SEG_T_S * 1000.0   # siguiente segmento 25 s después

    T = np.concatenate(T); B = np.concatenate(B); I = np.concatenate(I)
    if not np.all(np.diff(T) > 0):
        raise RuntimeError("Tiempo no monótono tras el cosido. Revisa recorte/offset.")
    return T, B, I

# =========================
# Main
# =========================
def main():
    need = int(round(TARGET_T_S / SEG_T_S))
    print(f"Objetivo ~{TARGET_T_S}s -> segmentos = {need}")

    segments = []
    for j in range(need):
        print(f"[{j+1}/{need}] segmento pulso ×{INPUT_FACTOR} (stim=1)…")
        segments.append(run_one_segment(j))

    # Unimos todo
    T, B, I = stitch(segments)

    # Guardar CSV
    os.makedirs(OUTDIR, exist_ok=True)
    csv_path = os.path.join(OUTDIR, "timeseries_modelA_600s.csv")
    pd.DataFrame({"t_ms": T, "BOLD": B, "I_CBF": I}).to_csv(csv_path, index=False)

    # Métricas simples (baseline=5 s)
    dt = float(np.median(np.diff(T)))
    b0 = int(5000.0/dt)
    metrics = pd.DataFrame([
        ["dt_ms", dt],
        ["len_samples", int(len(T))],
        ["series_duration_s", float((T[-1]-T[0])/1000.0)],
        ["BOLD_baseline_mean", float(B[:b0].mean())],
        ["BOLD_baseline_std",  float(B[:b0].std())],
        ["BOLD_peak_val", float(B[b0:].max())],
        ["BOLD_peak_time_s", float(T[b0:][B[b0:].argmax()]/1000.0)],
        ["BOLD_min_val",  float(B[b0:].min())],
        ["BOLD_min_time_s", float(T[b0:][B[b0:].argmin()]/1000.0)],
    ], columns=["metric","value"])
    metrics.to_csv(os.path.join(OUTDIR, "metrics_modelA_600s.csv"), index=False)

    # Plots
    t_s = T/1000.0
    plt.figure()
    plt.title("Modelo A — BOLD (tren de pulsos, ~600 s)")
    plt.plot(t_s, B, label="BOLD (Δrel.)")
    for k in range(need):
        start = (k*SEG_T_S) + PULSE_ONSET_LOCAL_S
        plt.axvspan(start, start+PULSE_DUR_S, alpha=0.08)
    plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "plot_modelA_600s_BOLD.png"), dpi=150); plt.close()

    plt.figure()
    plt.title("Modelo A — I_CBF (tren de pulsos, ~600 s)")
    plt.plot(t_s, I, label="I_CBF (norm)")
    for k in range(need):
        start = (k*SEG_T_S) + PULSE_ONSET_LOCAL_S
        plt.axvspan(start, start+PULSE_DUR_S, alpha=0.08)
    plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "plot_modelA_600s_ICBF.png"), dpi=150); plt.close()

    with open(os.path.join(OUTDIR, "provenance_modelA_600s.json"), "w") as f:
        json.dump(dict(
            model="A", input_factor=INPUT_FACTOR, stimulus=STIMULUS,
            seed=("seg_idx" if USE_SEGMENT_SEED else SEED_FIXED),
            seg_duration_s=SEG_T_S, n_segments=need, iti_s=ITI_S,
            note="Concatenación estricta de pulso ×5 (stim=1), recorte 25 s por segmento."
        ), f, indent=2)

    # Sanity print
    print("\nOK  →", OUTDIR)
    print(" -", csv_path)
    print(" - plot_modelA_600s_BOLD.png / plot_modelA_600s_ICBF.png")
    print(" - metrics_modelA_600s.csv")
    print(" - seg_raw/ (copias crudas)")

if __name__ == "__main__":
    main()

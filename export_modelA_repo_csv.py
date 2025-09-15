# -*- coding: utf-8 -*-
"""
Exporta el Modelo A (Fig. 6) del REPO (spiking) a CSV:
- Lee dataRaw/ de las 40 corridas (simID=0..39) generadas por srcSim/simulations.py
- Extrae del monitor BOLD el mapeo A (I_CBF <- 'syn'), promedia ROI (E+I)
- Alinea los ejes temporales (monitor empieza tras warm-up) y normaliza baseline (ya lo hace el monitor)
- Guarda por-seed y promedio en CSV + PNG + métricas

Uso:
  python export_modelA_repo_csv.py --out results/Fig6A_repo_csv

Si cambian rutas/formatos del repo, el script intenta detectar ficheros automáticamente.
"""

import os, re, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------- Parámetros esperados del protocolo (paper/repo) ----------
DT_MS = 1.0
INIT_MS = 2000
BASELINE_MS = 5000
PULSE_MS = 100
TOTAL_MS = 25000
# --------------------------------------------------------------------

def find_candidate_files():
    """
    Busca en dataRaw/ ficheros que contengan salidas del monitor BOLD.
    El repo puede guardar .npz/.npy/.pkl. Probamos heurísticas simples.
    """
    patterns = [
        "dataRaw/**/*.npz",
        "dataRaw/**/*.npy",
        "dataRaw/**/*.pkl",
        "dataRaw/**/*.pickle",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(files)

def load_any(path):
    """
    Carga npz/npy/pickle y devuelve un dict {name: array} lo mejor posible.
    No hace falta conocer claves exactas; intentamos mapear BOLD/I_CBF.
    """
    data = {}
    try:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=True) as z:
                for k in z.files:
                    data[k] = z[k]
        elif path.endswith(".npy"):
            arr = np.load(path, allow_pickle=True)
            # si es dict guardado con numpy, conviértelo
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1 and isinstance(arr.item(), dict):
                data = arr.item()
            else:
                data["array"] = arr
        elif path.endswith((".pkl", ".pickle")):
            import pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                data = obj
            else:
                data["object"] = obj
    except Exception:
        pass
    return data

def extract_modelA_from_dict(d):
    """
    Dado un dict con arrays del monitor, intenta devolver (t_ms, BOLD, I_CBF) del Modelo A.
    Reglas:
      - preferimos claves que contengan 'modelA' o 'A' junto con 'BOLD'/'I_CBF'
      - si hay varias poblaciones (E/I), promediamos en axis=1 -> (time,)
    """
    # Candidatos de claves
    key_map = {
        "BOLD": [ "BOLD_A", "BOLD_modelA", "BOLD_A_mean", "BOLD" ],
        "ICBF": [ "I_CBF_A", "ICBF_A", "I_CBF_modelA", "ICBF_modelA", "I_CBF" ],
        "time": [ "t_ms", "time_ms", "time", "t" ],
    }

    def pick_first(keys):
        for k in keys:
            if k in d and isinstance(d[k], np.ndarray):
                return d[k]
        # búsqueda aproximada
        for k in d.keys():
            low = k.lower()
            for target in keys:
                if target.lower() in low:
                    if isinstance(d[k], np.ndarray):
                        return d[k]
        return None

    bold = pick_first(key_map["BOLD"])
    icbf = pick_first(key_map["ICBF"])
    t_ms = pick_first(key_map["time"])

    # Si no hay tiempo, construimos desde parámetros conocidos
    if t_ms is None and bold is not None:
        # El monitor empieza tras INIT_MS
        n = bold.shape[0] if bold.ndim == 1 else bold.shape[0]
        t_ms = np.arange(INIT_MS, INIT_MS + n, 1.0)

    # Promedio ROI si vienen (time, npops)
    if isinstance(bold, np.ndarray) and bold.ndim == 2:
        bold = bold.mean(axis=1)
    if isinstance(icbf, np.ndarray) and icbf.ndim == 2:
        icbf = icbf.mean(axis=1)

    # Valida longitudes
    if isinstance(t_ms, np.ndarray) and isinstance(bold, np.ndarray) and len(t_ms) == len(bold):
        return t_ms.astype(float), bold.astype(float), (icbf.astype(float) if isinstance(icbf, np.ndarray) else None)
    return None, None, None

def load_all_seeds():
    """
    Recorre dataRaw y arma un dict {simID: (t_ms, bold, icbf)} para Modelo A.
    """
    files = find_candidate_files()
    by_seed = {}

    # simID suele aparecer en rutas o nombres de archivo
    seed_re = re.compile(r"(?:^|[_-])(?:sim)?ID[_-]?(\d+)|(?:^|[_-])(\d{1,3})(?:[_-]|$)")

    for path in files:
        d = load_any(path)
        if not d:
            continue
        t_ms, bold, icbf = extract_modelA_from_dict(d)
        if t_ms is None:
            continue

        # intenta deducir simID
        seed = None
        m = seed_re.search(os.path.basename(path))
        if m:
            seed = next(g for g in m.groups() if g is not None)
        if seed is None:
            # prueba con la carpeta padre
            m = seed_re.search(os.path.dirname(path))
            if m:
                seed = next(g for g in m.groups() if g is not None)

        if seed is None:
            # mete en una lista "unknown"
            seed = f"unknown_{len(by_seed)}"

        # si ya existe, prioriza el que tenga I_CBF
        old = by_seed.get(seed)
        if old is None or (old[2] is None and icbf is not None):
            by_seed[seed] = (t_ms, bold, icbf)

    return by_seed

def save_series_csv(out_dir, name, t_ms, bold, icbf):
    df = pd.DataFrame({"t_ms": t_ms, "BOLD": bold})
    if icbf is not None:
        df["I_CBF"] = icbf
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path

def metrics_and_plots(out_dir, name, t_ms, bold, icbf):
    os.makedirs(out_dir, exist_ok=True)
    dt = np.median(np.diff(t_ms))
    b0 = int(BASELINE_MS/dt)
    p1 = b0
    p2 = b0 + int(PULSE_MS/dt)
    q1 = p2
    q2 = len(t_ms)

    def seg_stats(x, a, b):
        s = x[a:b]
        return dict(mean=float(s.mean()), std=float(s.std()), min=float(s.min()), max=float(s.max()))

    B_base = seg_stats(bold, 0, b0)
    B_pulse= seg_stats(bold, p1, p2)
    B_post = seg_stats(bold, q1, q2)

    rows = [
        ["dt_ms", float(dt)],
        ["len_samples", int(len(t_ms))],
        ["BOLD_baseline_mean", B_base["mean"]],
        ["BOLD_baseline_std",  B_base["std"]],
        ["BOLD_peak_during_pulse", B_pulse["max"]],
        ["BOLD_peak_minus_base",   B_pulse["max"] - B_base["mean"]],
        ["BOLD_min_post",          B_post["min"]],
        ["BOLD_min_minus_base",    B_post["min"] - B_base["mean"]],
    ]
    if icbf is not None:
        I_base = seg_stats(icbf, 0, b0); I_pulse= seg_stats(icbf, p1, p2)
        rows += [
            ["ICBF_baseline_mean", I_base["mean"]],
            ["ICBF_baseline_std",  I_base["std"]],
            ["ICBF_mean_during_pulse", I_pulse["mean"]],
            ["ICBF_mean_minus_base",   I_pulse["mean"] - I_base["mean"]],
        ]
    pd.DataFrame(rows, columns=["metric","value"]).to_csv(
        os.path.join(out_dir, f"metrics_{name}.csv"), index=False
    )

    # plot
    t_s = t_ms/1000.0
    plt.figure()
    plt.title(f"{name} — BOLD (Modelo A)")
    plt.plot(t_s, bold, label="BOLD (Δrel.)")
    plt.axvspan(t_s[p1], t_s[p2-1], alpha=0.15, label="pulso")
    plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_{name}_BOLD.png"), dpi=150); plt.close()

    if icbf is not None:
        plt.figure()
        plt.title(f"{name} — I_CBF (Modelo A)")
        plt.plot(t_s, icbf, label="I_CBF (norm)")
        plt.axvspan(t_s[p1], t_s[p2-1], alpha=0.15, label="pulso")
        plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_{name}_ICBF.png"), dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/Fig6A_repo_csv", help="Directorio de salida.")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    by_seed = load_all_seeds()
    if not by_seed:
        raise RuntimeError(
            "No encontré datos del monitor en dataRaw/. "
            "Asegúrate de haber corrido primero srcSim/simulations.py (ver paso 1)."
        )

    # Exporta por seed
    per_seed_csv = []
    aligned = []

    for seed, (t_ms, bold, icbf) in sorted(by_seed.items(), key=lambda kv: kv[0]):
        name = f"modelA_seed{seed}"
        csv_path = save_series_csv(out, name, t_ms, bold, icbf)
        metrics_and_plots(out, name, t_ms, bold, icbf)
        per_seed_csv.append(csv_path)
        # Alineamos a la longitud mínima para promediar
        aligned.append(np.column_stack([t_ms, bold, (icbf if icbf is not None else np.zeros_like(bold))]))

    # Promedio (re-sincronizamos en el eje del primero tras cut a min len)
    min_len = min(a.shape[0] for a in aligned)
    T0 = aligned[0][:min_len, 0]
    B_stack = np.stack([a[:min_len, 1] for a in aligned], axis=1)
    I_stack = np.stack([a[:min_len, 2] for a in aligned], axis=1)

    B_mean = B_stack.mean(axis=1); B_std = B_stack.std(axis=1)
    I_mean = I_stack.mean(axis=1); I_std = I_stack.std(axis=1)

    # CSV promedio
    df_mean = pd.DataFrame({
        "t_ms": T0, "BOLD_mean": B_mean, "BOLD_std": B_std,
        "I_CBF_mean": I_mean, "I_CBF_std": I_std
    })
    mean_csv = os.path.join(out, "timeseries_modelA_mean.csv")
    df_mean.to_csv(mean_csv, index=False)

    # Métricas y plots del promedio
    metrics_and_plots(out, "modelA_mean", T0, B_mean, I_mean)

    # Metadatos
    meta = dict(
        model="A",
        mapping={"I_CBF": "syn"},
        dt_ms=DT_MS,
        durations_ms={"init": INIT_MS, "baseline": BASELINE_MS, "pulse": PULSE_MS, "total": TOTAL_MS},
        seeds=len(by_seed),
        per_seed_csv=per_seed_csv,
        mean_csv=mean_csv,
    )
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("OK. Export listo en:", out)
    print(" - CSV por seed: modelA_seed*.csv")
    print(" - Promedio: timeseries_modelA_mean.csv")
    print(" - PNGs: plot_*")
    print(" - Métricas: metrics_*")

if __name__ == "__main__":
    main()

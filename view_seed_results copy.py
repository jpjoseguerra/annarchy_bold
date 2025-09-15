# -*- coding: utf-8 -*-
"""
Visualiza y exporta resultados de una sola seed (simID) generados por el repo.
Por defecto intenta el Modelo A (I_CBF <- syn). Si no encuentra las claves,
te lista todas las variables relacionadas a BOLD/I_CBF detectadas para que elijas.

Uso:
  python view_seed_results.py --simID 0 --out results/seed0_view
Opcional:
  --model A|B|C|D|E|F  (solo orientativo para elegir claves; si no matchea, se hace autoselección)
"""

import os, re, glob, argparse, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del protocolo esperados por el repo/paper
DT_MS = 1.0
INIT_MS = 2000
BASELINE_MS = 5000
PULSE_MS = 100
TOTAL_MS = 25000

def path_matches_seed(path, seed: int):
    # Coincidencia "entera" de número (evita que 0 matchee 10, etc.)
    pat = re.compile(rf"(?<!\d){seed}(?!\d)")
    return bool(pat.search(os.path.basename(path))) or bool(pat.search(path))

def find_files_for_seed(seed):
    files = []
    for ext in ("npz","npy","pkl","pickle"):
        files += glob.glob(f"dataRaw/**/*.{ext}", recursive=True)
    return [p for p in sorted(files) if path_matches_seed(p, seed)]

def load_any(path):
    d = {}
    try:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=True) as z:
                for k in z.files:
                    d[k] = z[k]
        elif path.endswith(".npy"):
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1 and isinstance(arr.item(), dict):
                d = arr.item()
            else:
                d["array"] = arr
        elif path.endswith((".pkl",".pickle")):
            import pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)
            d = obj if isinstance(obj, dict) else {"object": obj}
    except Exception:
        return {}
    return d

def pick_keys_for_modelA(keys):
    # claves típicas para Modelo A según repos/papers
    candidates_B = ["BOLD_A","BOLD_modelA","BOLDsyn","BOLD_A_mean","BOLD"]
    candidates_I = ["I_CBF_A","ICBF_A","I_CBF_modelA","ICBF_modelA","I_CBF","ICBF","ICBFsyn"]
    def first_match(cands):
        for c in cands:
            for k in keys:
                if c.lower() in k.lower():
                    return k
        return None
    return first_match(candidates_B), first_match(candidates_I)

def extract_timeseries(d, prefer_model="A"):
    keys = list(d.keys())
    kB, kI = (None, None)
    if prefer_model.upper() == "A":
        kB, kI = pick_keys_for_modelA(keys)

    # fallback genérico si no hay coincidencia clara
    if kB is None:
        kB = next((k for k in keys if "bold" in k.lower()), None)
    if kI is None:
        kI = next((k for k in keys if "icbf" in k.lower() or "i_cbf" in k.lower()), None)

    # tiempo
    kT = next((k for k in keys if k.lower() in ("t_ms","time_ms","time","t")), None)

    t_ms = None
    if kT and isinstance(d[kT], np.ndarray):
        t_ms = d[kT].astype(float).ravel()

    def as_series(x):
        if not isinstance(x, np.ndarray):
            return None
        return x.mean(axis=1) if x.ndim == 2 else x.ravel()

    bold = as_series(d[kB]) if kB else None
    icbf = as_series(d[kI]) if kI else None

    # si no hay tiempo, lo reconstruimos desde que inicia el monitor (tras INIT_MS)
    if t_ms is None and bold is not None:
        t_ms = np.arange(INIT_MS, INIT_MS + len(bold), DT_MS)

    found = dict(keys=keys, kB=kB, kI=kI, kT=kT)
    return t_ms, bold, icbf, found

def seg_indices(t_ms, dt=DT_MS):
    b0 = int(BASELINE_MS/dt)
    p1 = b0
    p2 = b0 + int(PULSE_MS/dt)
    q1 = p2
    q2 = len(t_ms)
    return b0, p1, p2, q1, q2

def seg_stats(x, a, b):
    s = x[a:b]
    return dict(mean=float(s.mean()), std=float(s.std()), min=float(s.min()), max=float(s.max()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simID", type=int, default=0)
    ap.add_argument("--model", type=str, default="A")
    ap.add_argument("--out", type=str, default="results/seed_view")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = find_files_for_seed(args.simID)
    if not files:
        raise SystemExit(f"No encontré archivos en dataRaw/ para simID={args.simID}. ¿Corriste simulations.py desde srcSim/?")

    # Recorremos los archivos y guardamos el primero que tenga BOLD
    picked = None
    picked_meta = None
    for p in files:
        d = load_any(p)
        if not d:
            continue
        t_ms, bold, icbf, found = extract_timeseries(d, prefer_model=args.model)
        if bold is not None and t_ms is not None and len(t_ms) == len(bold):
            picked = (t_ms, bold, icbf)
            picked_meta = dict(path=p, **found)
            break

    if picked is None:
        msg = "No pude extraer BOLD/tiempo de los archivos. Claves vistas por archivo:\n"
        for p in files:
            d = load_any(p)
            if d:
                ks = ", ".join(sorted(d.keys())[:20])
                msg += f" - {p} -> {ks}\n"
        raise SystemExit(msg)

    t_ms, bold, icbf = picked
    b0, p1, p2, q1, q2 = seg_indices(t_ms, dt=np.median(np.diff(t_ms)))

    # Métricas
    B_base = seg_stats(bold, 0, b0)
    B_pulse= seg_stats(bold, p1, p2)
    B_post = seg_stats(bold, q1, q2)
    metrics = [
        ["dt_ms", float(np.median(np.diff(t_ms)))],
        ["len_samples", int(len(t_ms))],
        ["BOLD_baseline_mean", B_base["mean"]],
        ["BOLD_baseline_std",  B_base["std"]],
        ["BOLD_peak_during_pulse", B_pulse["max"]],
        ["BOLD_peak_minus_base",   B_pulse["max"] - B_base["mean"]],
        ["BOLD_min_post",          B_post["min"]],
        ["BOLD_min_minus_base",    B_post["min"] - B_base["mean"]],
    ]
    if icbf is not None:
        I_base = seg_stats(icbf, 0, b0); I_pulse = seg_stats(icbf, p1, p2)
        metrics += [
            ["ICBF_baseline_mean", I_base["mean"]],
            ["ICBF_baseline_std",  I_base["std"]],
            ["ICBF_mean_during_pulse", I_pulse["mean"]],
            ["ICBF_mean_minus_base",   I_pulse["mean"] - I_base["mean"]],
        ]
    pd.DataFrame(metrics, columns=["metric","value"]).to_csv(
        os.path.join(args.out, f"metrics_seed{args.simID}_model{args.model}.csv"),
        index=False
    )

    # CSV series
    df = pd.DataFrame({"t_ms": t_ms, "BOLD": bold})
    if icbf is not None:
        df["I_CBF"] = icbf
    csv_path = os.path.join(args.out, f"timeseries_seed{args.simID}_model{args.model}.csv")
    df.to_csv(csv_path, index=False)

    # Plots
    t_s = t_ms/1000.0
    plt.figure()
    plt.title(f"simID={args.simID} — BOLD (Modelo {args.model})")
    plt.plot(t_s, bold, label="BOLD (Δrel.)")
    plt.axvspan(t_s[p1], t_s[p2-1], alpha=0.15, label="pulso")
    plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"plot_seed{args.simID}_model{args.model}_BOLD.png"), dpi=150); plt.close()

    if icbf is not None:
        plt.figure()
        plt.title(f"simID={args.simID} — I_CBF (Modelo {args.model})")
        plt.plot(t_s, icbf, label="I_CBF (norm)")
        plt.axvspan(t_s[p1], t_s[p2-1], alpha=0.15, label="pulso")
        plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"plot_seed{args.simID}_model{args.model}_ICBF.png"), dpi=150); plt.close()

    # Guardamos “qué claves se usaron” para transparencia
    with open(os.path.join(args.out, f"provenance_seed{args.simID}.json"), "w") as f:
        json.dump(picked_meta, f, indent=2)

    print("OK. Resultados en:", args.out)
    print(" -", csv_path)
    print(" - metrics_seed{sid}_model{m}.csv, plot_*_BOLD.png, plot_*_ICBF.png".format(
        sid=args.simID, m=args.model))
    print("Claves usadas:", picked_meta)

if __name__ == "__main__":
    main()

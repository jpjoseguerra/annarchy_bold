# -*- coding: utf-8 -*-
# Exporta Modelo A (monitor 1) desde dataRaw/ para la condición 1.2 × pulso 100 ms.
# Crea CSV/PNGs/métricas por seed y el promedio.
#
# Uso:
#   python export_modelA_from_dataRaw.py --out results/Fig6A_repo_csv_1p2_pulse
#
# Opcional:
#   --pattern 1_2_1    # triplete en el nombre de archivo (por defecto 1_2_1)
#   --prefer recordingsB  # preferencia de fuente (recordingsB|recordings)

import os, re, glob, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DT_MS = 1.0
INIT_MS = 2000
BASELINE_MS = 5000
PULSE_MS = 100
TOTAL_MS = 25000

def find_files(pattern="1_2_1"):
    # Busca ambos tipos: recordingsB y recordings, y simParams (los ignoramos)
    recB = sorted(glob.glob(f"dataRaw/*recordingsB*_{pattern}__*.npy"))
    rec  = sorted(glob.glob(f"dataRaw/*recordings_*_{pattern}__*.npy"))
    return recB, rec

def seed_from_path(p):
    m = re.search(r"__([0-9]+)\.npy$", p)
    return int(m.group(1)) if m else None

def load_npy_dict(path):
    arr = np.load(path, allow_pickle=True)
    # Muchos archivos vienen como dict dentro de un array-objeto
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1 and isinstance(arr.item(), dict):
        return arr.item()
    # En algunos dumps es un dict "raw"
    if isinstance(arr, dict):
        return arr
    # Fallback: lo envolvemos como {"array": arr}
    return {"array": arr}

def extract_modelA(d):
    # Modelo A = monitor 1
    # claves típicas: '1;BOLD', '1;I_CBF'
    kB = next((k for k in d.keys() if k.lower() in ("1;bold","1; bold","1;BOLD","1;Bold".lower())), None)
    if kB is None:
        kB = next((k for k in d.keys() if "1;bold" in k.lower()), None)
    kI = next((k for k in d.keys() if k.lower() in ("1;i_cbf","1;icbf")), None)
    if kI is None:
        kI = next((k for k in d.keys() if ("1;i_cbf" in k.lower() or "1;icbf" in k.lower())), None)

    bold = d.get(kB, None)
    icbf = d.get(kI, None)

    def as_series(x):
        if not isinstance(x, np.ndarray):
            return None
        # Si viene (time, npops) → promedio ROI
        return x.mean(axis=1) if x.ndim == 2 else x.ravel()

    bold = as_series(bold) if bold is not None else None
    icbf = as_series(icbf) if icbf is not None else None

    if bold is None:
        return None, None
    # tiempo: el monitor arranca tras INIT_MS
    n = len(bold)
    t_ms = np.arange(INIT_MS, INIT_MS + n, DT_MS)
    return t_ms, (bold, icbf)

def seg_stats(x, a, b):
    s = x[a:b]
    return dict(mean=float(s.mean()), std=float(s.std()), min=float(s.min()), max=float(s.max()))

def metrics_and_plots(out_dir, name, t_ms, bold, icbf):
    os.makedirs(out_dir, exist_ok=True)
    dt = float(np.median(np.diff(t_ms)))
    b0 = int(BASELINE_MS/dt)
    p1 = b0
    p2 = b0 + int(PULSE_MS/dt)
    q1 = p2
    q2 = len(t_ms)

    B_base = seg_stats(bold, 0, b0)
    B_pulse= seg_stats(bold, p1, p2)
    B_post = seg_stats(bold, q1, q2)

    rows = [
        ["dt_ms", dt],
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

    # Plots
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
    ap.add_argument("--out", default="results/Fig6A_repo_csv_1p2_pulse")
    ap.add_argument("--pattern", default="1_2_1", help="Triplete en nombre (ej. 1_2_1 para 1.2, pulso 1)")
    ap.add_argument("--prefer", default="recordingsB", choices=["recordingsB","recordings"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    recB, rec = find_files(args.pattern)

    # Indexa por seed y elige la mejor fuente
    by_seed = {}
    for p in recB + rec:
        sid = seed_from_path(p)
        if sid is None: continue
        # preferencia: recordingsB > recordings
        if sid not in by_seed:
            by_seed[sid] = p
        else:
            if args.prefer == "recordingsB" and ("recordingsB" in p) and ("recordingsB" not in by_seed[sid]):
                by_seed[sid] = p

    if not by_seed:
        raise SystemExit(f"No encontré archivos con patrón {args.pattern} en dataRaw/. Revisa ls dataRaw")

    # Ordena por seed
    seeds = sorted(by_seed.keys())
    series = []
    csv_paths = []

    for sid in seeds:
        path = by_seed[sid]
        d = load_npy_dict(path)
        t_ms, data = extract_modelA(d)
        if t_ms is None:
            print(f"[WARN] No pude extraer Modelo A en {path}, salto.")
            continue
        bold, icbf = data
        # Guarda CSV por seed
        df = pd.DataFrame({"t_ms": t_ms, "BOLD": bold})
        if icbf is not None: df["I_CBF"] = icbf
        name = f"modelA_seed{sid}"
        csv_path = os.path.join(args.out, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)

        # Métricas y plots por seed
        metrics_and_plots(args.out, name, t_ms, bold, icbf)

        # Acumula para promedio (alineado a longitud mínima)
        series.append(np.column_stack([t_ms, bold, (icbf if icbf is not None else np.zeros_like(bold))]))

    if not series:
        raise SystemExit("No se pudo extraer ninguna seed válida.")

    min_len = min(s.shape[0] for s in series)
    T0 = series[0][:min_len, 0]
    B_stack = np.stack([s[:min_len, 1] for s in series], axis=1)
    I_stack = np.stack([s[:min_len, 2] for s in series], axis=1)

    B_mean = B_stack.mean(axis=1); B_std = B_stack.std(axis=1)
    I_mean = I_stack.mean(axis=1); I_std = I_stack.std(axis=1)

    # CSV promedio
    dfm = pd.DataFrame({
        "t_ms": T0,
        "BOLD_mean": B_mean, "BOLD_std": B_std,
        "I_CBF_mean": I_mean, "I_CBF_std": I_std
    })
    mean_csv = os.path.join(args.out, "timeseries_modelA_mean.csv")
    dfm.to_csv(mean_csv, index=False)

    # Métricas y plots del promedio
    metrics_and_plots(args.out, "modelA_mean", T0, B_mean, I_mean)

    # Metadata
    meta = dict(
        pattern=args.pattern, prefer=args.prefer,
        seeds=seeds, per_seed_csv=csv_paths, mean_csv=mean_csv,
        dt_ms=DT_MS, durations_ms=dict(init=INIT_MS, baseline=BASELINE_MS, pulse=PULSE_MS, total=TOTAL_MS),
        model="A", monitor_id=1, mapping="I_CBF <- syn", balloon_model="balloon_RN"
    )
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("OK. Export listo en:", args.out)
    print("Seeds:", seeds)
    print("Promedio:", mean_csv)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Extrae y guarda resultados por seed (en subcarpetas) y luego calcula el promedio
para un modelo (A..F) en una condición dada (input_factor, stimulus).

Salida:
  <out>/
    seeds/seed_<id>/timeseries_modelA_seed<id>.csv
    seeds/seed_<id>/metrics_modelA_seed<id>.csv
    seeds/seed_<id>/plot_modelA_seed<id>_BOLD.png
    seeds/seed_<id>/plot_modelA_seed<id>_ICBF.png
    seeds_index.csv   (lista de seeds y archivos origen)
    seeds_metrics.csv (todas las métricas juntas)

    timeseries_modelA_mean.csv
    plot_modelA_mean_BOLD.png
    plot_modelA_mean_ICBF.png
    metrics_modelA_mean.csv
    provenance_modelA_mean.json

Uso típico:
  python view_mean_and_seeds.py --model A --if 1.2 --stim 3 --pulse_ms 20000 \
      --out results/mean_A_sostenido1p2

  python view_mean_and_seeds.py --model A --if 5 --stim 1 --pulse_ms 100 \
      --out results/mean_A_pulse5
"""
import os, re, glob, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DT_MS = 1.0
INIT_MS = 2000
BASELINE_MS_DEFAULT = 5000
MODEL2ID = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}

def find_seed_files(input_factor, stimulus):
    if_tag = str(input_factor).replace(".", "_")
    pats = [
        f"dataRaw/**/*recordingsB_{if_tag}_{stimulus}__*.npy",
        f"dataRaw/**/*recordingsB_{if_tag}_{stimulus}__*.npz",
        f"dataRaw/**/*recordingsB_{if_tag}_{stimulus}__*.pkl",
        f"dataRaw/**/*recordingsB_{if_tag}_{stimulus}__*.pickle",
    ]
    files = []
    for p in pats: files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    seed_re = re.compile(rf"recordingsB_{re.escape(if_tag)}_{stimulus}__(\d+)\.")
    items = []
    for path in files:
        m = seed_re.search(os.path.basename(path))
        if m: items.append((int(m.group(1)), path))
    items.sort(key=lambda x: x[0])
    return items  # [(seed, path), ...]

def load_any(path):
    try:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=True) as z:
                return {k: z[k] for k in z.files}
        elif path.endswith(".npy"):
            arr = np.load(path, allow_pickle=True)
            if (isinstance(arr, np.ndarray) and arr.dtype==object and arr.size==1
                and isinstance(arr.item(), dict)):
                return arr.item()
            return {"array": arr}
        elif path.endswith((".pkl",".pickle")):
            import pickle
            with open(path, "rb") as f: obj = pickle.load(f)
            return obj if isinstance(obj, dict) else {"object": obj}
    except Exception:
        return {}
    return {}

def extract_series_for_model(d, model_letter):
    mid = MODEL2ID[model_letter]
    kB, kI = f"{mid};BOLD", f"{mid};I_CBF"
    t = None
    for k in ("t_ms","time_ms","time","t"):
        if isinstance(d.get(k), np.ndarray):
            t = d[k].astype(float).ravel(); break
    def as_series(x):
        if not isinstance(x, np.ndarray): return None
        return x.mean(axis=1) if x.ndim==2 else x.ravel()
    bold = as_series(d.get(kB))
    icbf = as_series(d.get(kI))
    if t is None and bold is not None:
        t = np.arange(INIT_MS, INIT_MS + len(bold), DT_MS, dtype=float)
    if t is None or bold is None or len(t) != len(bold):
        return None, None, None
    return t, bold, icbf

def seg_indices(t_ms, baseline_ms, pulse_ms):
    dt = float(np.median(np.diff(t_ms)))
    b0 = int(baseline_ms/dt); p1 = b0; p2 = b0 + int(pulse_ms/dt)
    q1, q2 = p2, len(t_ms)
    return dt, b0, p1, p2, q1, q2

def seg_stats(x, a, b):
    s = x[a:b]
    return dict(mean=float(np.mean(s)), std=float(np.std(s)),
                min=float(np.min(s)), max=float(np.max(s)))

def metrics_table(t_ms, bold, icbf, baseline_ms, pulse_ms):
    dt, b0, p1, p2, q1, q2 = seg_indices(t_ms, baseline_ms, pulse_ms)
    base = seg_stats(bold, 0, b0)
    post_peak_idx = int(np.argmax(bold[q1:q2])); post_peak_val = float(bold[q1+post_peak_idx]); post_peak_time_s = float(t_ms[q1+post_peak_idx]/1000.0)
    post_min_idx  = int(np.argmin(bold[q1:q2])); post_min_val  = float(bold[q1+post_min_idx]);  post_min_time_s  = float(t_ms[q1+post_min_idx]/1000.0)
    def fwhm(x, t, baseline):
        peak = float(x.max()); half = baseline + 0.5*(peak - baseline)
        idx = np.where(x >= half)[0]
        if len(idx)==0: return np.nan
        return float((t[idx[-1]] - t[idx[0]]))
    fwhm_ms = fwhm(bold[b0:], t_ms[b0:], base["mean"])
    auc = float(np.trapz(bold[q1:] - base["mean"], t_ms[q1:] )) / 1000.0
    rows = [
        ["dt_ms", dt],
        ["len_samples", int(len(t_ms))],
        ["baseline_ms", float(baseline_ms)],
        ["pulse_ms", float(pulse_ms)],
        ["BOLD_baseline_mean", base["mean"]],
        ["BOLD_baseline_std",  base["std"]],
        ["BOLD_post_peak_val", post_peak_val],
        ["BOLD_post_peak_minus_base", post_peak_val - base["mean"]],
        ["BOLD_post_peak_time_s", post_peak_time_s],
        ["BOLD_post_min_val", post_min_val],
        ["BOLD_post_min_minus_base", post_min_val - base["mean"]],
        ["BOLD_post_min_time_s", post_min_time_s],
        ["BOLD_FWHM_s", fwhm_ms/1000.0 if not np.isnan(fwhm_ms) else np.nan],
        ["BOLD_AUC_from_pulse_s_units", auc],
    ]
    if icbf is not None:
        I_base = seg_stats(icbf, 0, b0)
        rows += [
            ["ICBF_baseline_mean", I_base["mean"]],
            ["ICBF_baseline_std",  I_base["std"]],
            ["ICBF_mean_post_mean", float(np.mean(icbf[q1:q2]))],
        ]
    return pd.DataFrame(rows, columns=["metric","value"])

def plot_seed(out_dir, model, t_ms, bold, icbf, baseline_ms, pulse_ms, seed):
    os.makedirs(out_dir, exist_ok=True)
    dt, b0, p1, p2, q1, q2 = seg_indices(t_ms, baseline_ms, pulse_ms)
    t_s = t_ms/1000.0
    plt.figure()
    plt.title(f"Modelo {model} — BOLD (seed {seed})")
    plt.plot(t_s, bold, label="BOLD (Δrel.)")
    plt.axvspan(t_s[p1], t_s[min(p2-1,len(t_s)-1)], alpha=0.12, label="estímulo")
    plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_model{model}_seed{seed}_BOLD.png"), dpi=140); plt.close()
    if icbf is not None:
        plt.figure()
        plt.title(f"Modelo {model} — I_CBF (seed {seed})")
        plt.plot(t_s, icbf, label="I_CBF (norm)")
        plt.axvspan(t_s[p1], t_s[min(p2-1,len(t_s)-1)], alpha=0.12, label="estímulo")
        plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_model{model}_seed{seed}_ICBF.png"), dpi=140); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="A", help="A..F")
    ap.add_argument("--if", dest="input_factor", type=float, required=True)
    ap.add_argument("--stim", dest="stimulus", type=int, required=True)
    ap.add_argument("--pulse_ms", type=int, required=True, help="100 (pulso) o 20000 (sostenido)")
    ap.add_argument("--baseline_ms", type=int, default=BASELINE_MS_DEFAULT)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--no-spaghetti", action="store_true")
    args = ap.parse_args()

    model = args.model.upper()
    if model not in MODEL2ID: raise SystemExit("model debe ser A..F")

    items = find_seed_files(args.input_factor, args.stimulus)
    if not items: raise SystemExit("No hay archivos para esa condición en dataRaw/.")

    os.makedirs(args.out, exist_ok=True)
    seeds_dir = os.path.join(args.out, "seeds"); os.makedirs(seeds_dir, exist_ok=True)

    used, per_seed, metrics_rows = [], [], []
    index_rows = []

    # --- Procesa cada seed y guarda en subcarpeta ---
    for seed, path in items:
        d = load_any(path)
        t_ms, bold, icbf = extract_series_for_model(d, model)
        if t_ms is None: continue
        # recorta si fuera necesario para alinear luego
        if per_seed:
            min_len = min(len(t_ms), min(arr.shape[0] for arr in per_seed))
            t_ms, bold = t_ms[:min_len], bold[:min_len]
            if icbf is not None: icbf = icbf[:min_len]
        seed_dir = os.path.join(seeds_dir, f"seed_{seed:02d}")
        os.makedirs(seed_dir, exist_ok=True)

        # CSV por seed
        df = pd.DataFrame({"t_ms": t_ms, "BOLD": bold})
        if icbf is not None: df["I_CBF"] = icbf
        df.to_csv(os.path.join(seed_dir, f"timeseries_model{model}_seed{seed}.csv"),
                  index=False)

        # plots por seed
        plot_seed(seed_dir, model, t_ms, bold, icbf, args.baseline_ms, args.pulse_ms, seed)

        # métricas por seed
        m = metrics_table(t_ms, bold, icbf, args.baseline_ms, args.pulse_ms)
        m.to_csv(os.path.join(seed_dir, f"metrics_model{model}_seed{seed}.csv"), index=False)

        # acumular para promedio
        per_seed.append(np.column_stack([t_ms, bold, (icbf if icbf is not None else np.zeros_like(bold))]))
        used.append(seed)
        index_rows.append([seed, path])

    # índice y métricas combinadas
    pd.DataFrame(index_rows, columns=["seed","file"]).to_csv(
        os.path.join(args.out, "seeds_index.csv"), index=False
    )
    # compilar todas las métricas individuales (leyéndolas desde disco)
    all_metrics = []
    for seed in used:
        p = os.path.join(seeds_dir, f"seed_{seed:02d}", f"metrics_model{model}_seed{seed}.csv")
        dfm = pd.read_csv(p); dfm.insert(0, "seed", seed); all_metrics.append(dfm)
    pd.concat(all_metrics, ignore_index=True).to_csv(
        os.path.join(args.out, "seeds_metrics.csv"), index=False
    )

    # --- Promedio (media/σ) ---
    min_len = min(arr.shape[0] for arr in per_seed)
    T = per_seed[0][:min_len, 0]
    B = np.stack([arr[:min_len, 1] for arr in per_seed], axis=1)
    I = np.stack([arr[:min_len, 2] for arr in per_seed], axis=1)

    B_mean, B_std = B.mean(axis=1), B.std(axis=1)
    I_mean, I_std = (I.mean(axis=1), I.std(axis=1)) if np.any(I) else (None, None)

    out_csv = os.path.join(args.out, f"timeseries_model{model}_mean.csv")
    dfm = pd.DataFrame({"t_ms": T, "BOLD_mean": B_mean, "BOLD_std": B_std})
    if I_mean is not None:
        dfm["I_CBF_mean"] = I_mean; dfm["I_CBF_std"] = I_std
    dfm.to_csv(out_csv, index=False)

    # Plots del promedio
    t_s = T/1000.0
    dt = float(np.median(np.diff(T))); b0=int(args.baseline_ms/dt); p1=b0; p2=b0+int(args.pulse_ms/dt)
    # BOLD
    plt.figure()
    plt.title(f"Modelo {model} — BOLD (media de {len(used)} seeds)")
    if not args.no_spaghetti:
        for k in range(B.shape[1]): plt.plot(t_s, B[:,k], alpha=0.12, linewidth=0.6)
    plt.plot(t_s, B_mean, linewidth=2.0, label="media")
    plt.fill_between(t_s, B_mean-B_std, B_mean+B_std, alpha=0.2, label="±1σ")
    plt.axvspan(t_s[p1], t_s[min(p2-1,len(t_s)-1)], alpha=0.12, label="estímulo")
    plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"plot_model{model}_mean_BOLD.png"), dpi=150); plt.close()
    # I_CBF
    if I_mean is not None:
        plt.figure()
        plt.title(f"Modelo {model} — I_CBF (media de {len(used)} seeds)")
        if not args.no_spaghetti:
            for k in range(I.shape[1]): plt.plot(t_s, I[:,k], alpha=0.12, linewidth=0.6)
        plt.plot(t_s, I_mean, linewidth=2.0, label="media")
        plt.fill_between(t_s, I_mean-I_std, I_mean+I_std, alpha=0.2, label="±1σ")
        plt.axvspan(t_s[p1], t_s[min(p2-1,len(t_s)-1)], alpha=0.12, label="estímulo")
        plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"plot_model{model}_mean_ICBF.png"), dpi=150); plt.close()

    # Métricas del promedio
    metrics_table(T, B_mean, (I_mean if I_mean is not None else None),
                  args.baseline_ms, args.pulse_ms).to_csv(
        os.path.join(args.out, f"metrics_model{model}_mean.csv"), index=False
    )

    with open(os.path.join(args.out, f"provenance_model{model}_mean.json"), "w") as f:
        json.dump(dict(model=model, input_factor=args.input_factor, stimulus=args.stimulus,
                       pulse_ms=args.pulse_ms, baseline_ms=args.baseline_ms,
                       seeds_used=used, n_seeds=len(used)), f, indent=2)

    print(f"OK. {len(used)} seeds procesadas. Salida en: {args.out}")
    print("  - CSV promedio:", out_csv)
    print("  - Subcarpetas por seed en:", os.path.join(args.out, "seeds"))
    print("  - Métricas por seed: seeds_metrics.csv")
    print("  - Gráficos: plot_model*_mean_*.png y por seed en cada subcarpeta")
if __name__ == "__main__":
    main()

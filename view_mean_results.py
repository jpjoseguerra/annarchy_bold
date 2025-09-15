# -*- coding: utf-8 -*-
"""
Promedia todas las seeds presentes en dataRaw/ para un modelo (A–F) y una condición (input_factor, stimulus).
Genera:
- CSV: timeseries_model<MODEL>_mean.csv  (t_ms, BOLD_mean/std, I_CBF_mean/std)
- PNG: plot_model<MODEL>_mean_BOLD.png (+ banda ±std y 'spaghetti' opcional)
- PNG: plot_model<MODEL>_mean_ICBF.png
- CSV: metrics_model<MODEL>_mean.csv     (baseline, pico, undershoot, FWHM, AUC)
- JSON: provenance (seeds usadas, patrón de archivos)

Uso típico:
  python view_mean_results.py --model A --if 1.2 --stim 3 --pulse_ms 20000 \
      --out results/mean_A_sostenido1p2

  python view_mean_results.py --model A --if 5 --stim 1 --pulse_ms 100 \
      --out results/mean_A_pulse5

Se puede re-ejecutar en cualquier momento; tomará todas las seeds que existan.
"""
import os, re, glob, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Protocolo (coincidente con el repo/paper)
DT_MS = 1.0
INIT_MS = 2000
BASELINE_MS_DEFAULT = 5000

# Monitor IDs del repo: 1..6 -> A..F
MODEL2ID = { "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6 }

def load_any(path):
    """
    Carga .npz/.npy/.pkl -> dict; robusto a guardar diccionario en .npy
    """
    try:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=True) as z:
                return {k: z[k] for k in z.files}
        elif path.endswith(".npy"):
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1 and isinstance(arr.item(), dict):
                return arr.item()
            return {"array": arr}
        elif path.endswith((".pkl",".pickle")):
            import pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, dict) else {"object": obj}
    except Exception:
        return {}
    return {}

def extract_series_for_model(d, model_letter):
    """
    Devuelve (t_ms, BOLD, I_CBF) para el modelo indicado.
    Claves esperadas: '<id>;BOLD' y '<id>;I_CBF'. Si no hay tiempo, se reconstruye desde INIT_MS.
    Promedia ROI si vienen (time, n_pops).
    """
    mid = MODEL2ID[model_letter]
    kB = f"{mid};BOLD"
    kI = f"{mid};I_CBF"
    kT_candidates = ("t_ms","time_ms","time","t")

    bold = d.get(kB, None)
    icbf = d.get(kI, None)
    t_ms = None
    for k in kT_candidates:
        if isinstance(d.get(k, None), np.ndarray):
            t_ms = d[k].astype(float).ravel()
            break

    def as_series(x):
        if not isinstance(x, np.ndarray): return None
        return x.mean(axis=1) if x.ndim == 2 else x.ravel()

    bold = as_series(bold)
    icbf = as_series(icbf)

    if t_ms is None and bold is not None:
        t_ms = np.arange(INIT_MS, INIT_MS + len(bold), DT_MS, dtype=float)

    if t_ms is None or bold is None or len(t_ms) != len(bold):
        return None, None, None
    return t_ms, bold, icbf

def find_seed_files(input_factor, stimulus):
    """
    Encuentra archivos de la forma:
      dataRaw/simulations_BOLDfromDifferentSources_recordingsB_<ifTag>_<stim>__<seed>.npy
    (acepta .npz/.pkl también)
    """
    if_tag = str(input_factor).replace(".", "_")
    pats = [
        f"dataRaw/**/*recordingsB_{if_tag}_{stimulus}__*.npy",
        f"dataRaw/**/*recordingsB_{if_tag}_{stimulus}__*.npz",
        f"dataRaw/**/*recordingsB_{if_tag}_{stimulus}__*.pkl",
        f"dataRaw/**/*recordingsB_{if_tag}_{stimulus}__*.pickle",
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    # extrae seed de nombre
    seed_re = re.compile(rf"recordingsB_{re.escape(if_tag)}_{stimulus}__(\d+)\.")
    items = []
    for path in files:
        m = seed_re.search(os.path.basename(path))
        if m:
            items.append( (int(m.group(1)), path) )
    # ordenados por seed
    items.sort(key=lambda x: x[0])
    return items  # [(seed, path), ...]

def metrics_and_plots(out_dir, model, t_ms, bold_mean, bold_std, icbf_mean, icbf_std, pulse_ms, baseline_ms):
    dt = float(np.median(np.diff(t_ms)))
    b0 = int(baseline_ms/dt)
    p1 = b0
    p2 = b0 + int(pulse_ms/dt)
    q1 = p2
    q2 = len(t_ms)

    def seg_stats(x, a, b):
        s = x[a:b]
        return dict(mean=float(np.mean(s)), std=float(np.std(s)), min=float(np.min(s)), max=float(np.max(s)))

    B_base = seg_stats(bold_mean, 0, b0)
    B_post = seg_stats(bold_mean, q1, q2)

    # Pico y mínimo post-pulso
    post_peak_idx = int(np.argmax(bold_mean[q1:q2])); post_peak_val = float(bold_mean[q1+post_peak_idx]); post_peak_time_s = float(t_ms[q1+post_peak_idx]/1000.0)
    post_min_idx  = int(np.argmin(bold_mean[q1:q2])); post_min_val  = float(bold_mean[q1+post_min_idx]);  post_min_time_s  = float(t_ms[q1+post_min_idx]/1000.0)

    # FWHM sobre el lóbulo positivo (respecto a baseline)
    def fwhm(x, t, baseline):
        peak = float(x.max()); half = baseline + 0.5*(peak - baseline)
        idx = np.where(x >= half)[0]
        if len(idx)==0: return np.nan
        return float((t[idx[-1]] - t[idx[0]]))
    fwhm_ms = fwhm(bold_mean[b0:], t_ms[b0:], B_base["mean"])

    # AUC desde fin del estímulo (baseline-corrected)
    auc = float(np.trapz(bold_mean[q1:] - B_base["mean"], t_ms[q1:] )) / 1000.0

    rows = [
        ["dt_ms", dt],
        ["len_samples", int(len(t_ms))],
        ["baseline_ms", float(baseline_ms)],
        ["pulse_ms", float(pulse_ms)],
        ["BOLD_baseline_mean", B_base["mean"]],
        ["BOLD_baseline_std",  B_base["std"]],
        ["BOLD_post_peak_val", post_peak_val],
        ["BOLD_post_peak_minus_base", post_peak_val - B_base["mean"]],
        ["BOLD_post_peak_time_s", post_peak_time_s],
        ["BOLD_post_min_val", post_min_val],
        ["BOLD_post_min_minus_base", post_min_val - B_base["mean"]],
        ["BOLD_post_min_time_s", post_min_time_s],
        ["BOLD_FWHM_s", fwhm_ms/1000.0 if not np.isnan(fwhm_ms) else np.nan],
        ["BOLD_AUC_from_pulse_s_units", auc],
    ]
    if icbf_mean is not None:
        I_base = seg_stats(icbf_mean, 0, b0)
        rows += [
            ["ICBF_baseline_mean", I_base["mean"]],
            ["ICBF_baseline_std",  I_base["std"]],
            ["ICBF_mean_post_mean", float(np.mean(icbf_mean[q1:q2]))],
        ]
    pd.DataFrame(rows, columns=["metric","value"]).to_csv(
        os.path.join(out_dir, f"metrics_model{model}_mean.csv"), index=False
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="A", help="A..F")
    ap.add_argument("--if", dest="input_factor", type=float, required=True, help="input_factor (p.ej. 1.2 o 5)")
    ap.add_argument("--stim", dest="stimulus", type=int, required=True, help="stimulus (1=pulso 100ms, 3=sostenido ~20s)")
    ap.add_argument("--pulse_ms", type=int, required=True, help="duración del estímulo en ms (100 o 20000)")
    ap.add_argument("--baseline_ms", type=int, default=BASELINE_MS_DEFAULT)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--no-spaghetti", action="store_true", help="no dibujar las curvas por seed")
    args = ap.parse_args()

    model = args.model.upper()
    if model not in MODEL2ID:
        raise SystemExit("model debe ser A..F")

    os.makedirs(args.out, exist_ok=True)

    items = find_seed_files(args.input_factor, args.stimulus)
    if not items:
        raise SystemExit("No encontré archivos en dataRaw/ para esa condición. ¿Ya corriste simulations.py?")

    # Cargar y apilar seeds
    per_seed = []
    used = []
    t_ref = None
    for seed, path in items:
        d = load_any(path)
        t_ms, b, i = extract_series_for_model(d, model)
        if t_ms is None: 
            continue
        if t_ref is None:
            t_ref = t_ms
        # recorta a la longitud mínima entre t_ref y esta seed
        L = min(len(t_ref), len(t_ms))
        per_seed.append(np.column_stack([t_ms[:L], b[:L], (i[:L] if i is not None else np.zeros(L))]))
        used.append(seed)

    if not per_seed:
        raise SystemExit("Encontré archivos pero no pude extraer BOLD/I_CBF para ese modelo.")

    # Alinear y promediar
    min_len = min(arr.shape[0] for arr in per_seed)
    T = per_seed[0][:min_len, 0]
    B = np.stack([arr[:min_len, 1] for arr in per_seed], axis=1)
    I = np.stack([arr[:min_len, 2] for arr in per_seed], axis=1)

    B_mean, B_std = B.mean(axis=1), B.std(axis=1)
    I_mean, I_std = (I.mean(axis=1), I.std(axis=1)) if np.any(I) else (None, None)

    # Guardar CSV de promedio
    df = pd.DataFrame({"t_ms": T, "BOLD_mean": B_mean, "BOLD_std": B_std})
    if I_mean is not None:
        df["I_CBF_mean"] = I_mean; df["I_CBF_std"] = I_std
    mean_csv = os.path.join(args.out, f"timeseries_model{model}_mean.csv")
    df.to_csv(mean_csv, index=False)

    # Plots
    t_s = T/1000.0
    # BOLD
    plt.figure()
    plt.title(f"Modelo {model} — BOLD (promedio de {len(used)} seeds)")
    if not args.no_spaghetti:
        for k in range(B.shape[1]):
            plt.plot(t_s, B[:,k], alpha=0.15, linewidth=0.6)
    plt.plot(t_s, B_mean, linewidth=2.0, label="media")
    plt.fill_between(t_s, B_mean-B_std, B_mean+B_std, alpha=0.2, label="±1σ")
    # ventana de estímulo
    dt = float(np.median(np.diff(T)))
    b0 = int(args.baseline_ms/dt); p1=b0; p2=b0+int(args.pulse_ms/dt)
    plt.axvspan(t_s[p1], t_s[min(p2-1, len(t_s)-1)], alpha=0.12, label="estímulo")
    plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"plot_model{model}_mean_BOLD.png"), dpi=150); plt.close()

    # I_CBF
    if I_mean is not None:
        plt.figure()
        plt.title(f"Modelo {model} — I_CBF (promedio de {len(used)} seeds)")
        if not args.no_spaghetti:
            for k in range(I.shape[1]):
                plt.plot(t_s, I[:,k], alpha=0.12, linewidth=0.6)
        plt.plot(t_s, I_mean, linewidth=2.0, label="media")
        plt.fill_between(t_s, I_mean-I_std, I_mean+I_std, alpha=0.2, label="±1σ")
        plt.axvspan(t_s[p1], t_s[min(p2-1, len(t_s)-1)], alpha=0.12, label="estímulo")
        plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"plot_model{model}_mean_ICBF.png"), dpi=150); plt.close()

    # Métricas
    metrics_and_plots(args.out, model, T, B_mean, B_std, I_mean, I_std, args.pulse_ms, args.baseline_ms)

    # Proveniencia
    with open(os.path.join(args.out, f"provenance_model{model}_mean.json"), "w") as f:
        json.dump(dict(model=model, input_factor=args.input_factor, stimulus=args.stimulus,
                       pulse_ms=args.pulse_ms, baseline_ms=args.baseline_ms,
                       seeds_used=used, n_seeds=len(used)), f, indent=2)

    print("OK. Promedio listo en:", args.out)
    print(" -", mean_csv)
    print(" - plot_model{m}_mean_BOLD.png".format(m=model),
          "(y plot_model{m}_mean_ICBF.png)".format(m=model))
    print(" - metrics_model{m}_mean.csv".format(m=model))
    print("Seeds usadas:", used)

if __name__ == "__main__":
    main()

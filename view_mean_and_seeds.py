# -*- coding: utf-8 -*-
"""
Extrae y guarda resultados por seed (en subcarpetas) y calcula el promedio
para un modelo (A..F) en una condición dada (input_factor, stimulus).

Corrección clave:
- El sombreado del estímulo se alinea con el onset/duración reales leídos de simParams.
- En el promedio se usa la mediana de onsets/duraciones para la franja.

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
INIT_MS = 2000.0
BASELINE_MS_DEFAULT = 5000.0
MODEL2ID = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}

# ---------- utilidades ----------
def guess_if_tag(v):
    s = str(v)
    return s.replace(".", "_")

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
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, dict) else {"object": obj}
    except Exception:
        return {}
    return {}

def load_simparams(if_tag, stim, seed):
    p = f"dataRaw/simulations_BOLDfromDifferentSources_simParams_{if_tag}_{stim}__{seed}.npy"
    if not os.path.exists(p):
        return {}
    try:
        sp = np.load(p, allow_pickle=True).item()
        return sp if isinstance(sp, dict) else {}
    except Exception:
        return {}

def pick(d, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default

def as_series(x):
    if not isinstance(x, np.ndarray): return None
    return x.mean(axis=1) if x.ndim==2 else x.ravel()

def extract_series_for_model(d, model_letter):
    mid = MODEL2ID[model_letter]
    kB, kI = f"{mid};BOLD", f"{mid};I_CBF"
    # tiempo si está guardado
    for k in ("t_ms","time_ms","time","t"):
        if isinstance(d.get(k), np.ndarray):
            t = d[k].astype(float).ravel()
            break
    else:
        t = None
    bold = as_series(d.get(kB))
    icbf = as_series(d.get(kI))
    if bold is None:
        return None, None, None
    if t is None:
        t = np.arange(INIT_MS, INIT_MS + len(bold), DT_MS, dtype=float)
    return t, bold, icbf

def find_seed_files(input_factor, stimulus):
    if_tag = guess_if_tag(input_factor)
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
    seed_re = re.compile(rf"recordingsB_{re.escape(if_tag)}_{stimulus}__(\d+)\.")
    items = []
    for path in files:
        m = seed_re.search(os.path.basename(path))
        if m:
            items.append((int(m.group(1)), path))
    items.sort(key=lambda x: x[0])
    return items  # [(seed, path)]

def metrics_table_event(t_ms, bold, onset_ms, dur_ms, baseline_ms):
    # baseline: primeros baseline_ms desde el inicio de la serie
    dt = float(np.median(np.diff(t_ms)))
    b0 = int(baseline_ms/dt)
    base_mean = float(np.mean(bold[:b0]))
    base_std  = float(np.std(bold[:b0]))
    # índices del evento
    p1 = int(max(0, round((onset_ms - t_ms[0]) / dt)))
    p2 = int(min(len(t_ms), p1 + round(dur_ms / dt)))
    q1, q2 = p2, len(t_ms)
    post = bold[q1:q2]
    Tpost = t_ms[q1:q2]
    post_peak_val = float(post.max()) if len(post) else np.nan
    post_peak_time_s = float(Tpost[np.argmax(post)]/1000.0) if len(post) else np.nan
    post_min_val = float(post.min()) if len(post) else np.nan
    post_min_time_s = float(Tpost[np.argmin(post)]/1000.0) if len(post) else np.nan
    # FWHM relativo a baseline en la parte post
    if len(post):
        peak = float(post.max())
        half = base_mean + 0.5*(peak - base_mean)
        idx = np.where(post >= half)[0]
        fwhm_s = float((Tpost[idx[-1]] - Tpost[idx[0]])/1000.0) if len(idx) else np.nan
    else:
        fwhm_s = np.nan
    # AUC post-estímulo en unidades de segundo
    auc = float(np.trapz((post - base_mean), Tpost)/1000.0) if len(post) else np.nan
    rows = [
        ["dt_ms", dt],
        ["len_samples", int(len(t_ms))],
        ["baseline_ms", float(baseline_ms)],
        ["stim_onset_ms", float(onset_ms)],
        ["stim_duration_ms", float(dur_ms)],
        ["BOLD_baseline_mean", base_mean],
        ["BOLD_baseline_std", base_std],
        ["BOLD_post_peak_val", post_peak_val],
        ["BOLD_post_peak_minus_base", post_peak_val - base_mean if not np.isnan(post_peak_val) else np.nan],
        ["BOLD_post_peak_time_s", post_peak_time_s],
        ["BOLD_post_min_val", post_min_val],
        ["BOLD_post_min_minus_base", post_min_val - base_mean if not np.isnan(post_min_val) else np.nan],
        ["BOLD_post_min_time_s", post_min_time_s],
        ["BOLD_FWHM_s", fwhm_s],
        ["BOLD_AUC_from_post_s", auc],
    ]
    return pd.DataFrame(rows, columns=["metric","value"])

# ---------- plotting helpers ----------
def shade_event(ax, t_ms, onset_ms, dur_ms, **kw):
    t_s = t_ms/1000.0
    s0 = onset_ms/1000.0
    s1 = (onset_ms + dur_ms)/1000.0
    ax.axvspan(s0, s1, **kw)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="A", help="A..F")
    ap.add_argument("--if", dest="input_factor", type=float, required=True)
    ap.add_argument("--stim", dest="stimulus", type=int, required=True)
    ap.add_argument("--pulse_ms", type=float, default=None, help="(opcional) duración estímulo/bloque si no está en simParams")
    ap.add_argument("--stim_start_ms", type=float, default=None, help="(opcional) onset si no está en simParams")
    ap.add_argument("--baseline_ms", type=float, default=BASELINE_MS_DEFAULT)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--no-spaghetti", action="store_true")
    args = ap.parse_args()

    model = args.model.upper()
    if model not in MODEL2ID:
        raise SystemExit("model debe ser A..F")

    if_tag = guess_if_tag(args.input_factor)
    items = find_seed_files(args.input_factor, args.stimulus)
    if not items:
        raise SystemExit("No hay archivos para esa condición en dataRaw/.")

    os.makedirs(args.out, exist_ok=True)
    seeds_dir = os.path.join(args.out, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    # Defaults si simParams no trae nada:
    default_onset = 5000.0 if args.stimulus==1 else 50000.0
    default_dur   = 100.0  if args.stimulus==1 else 20000.0
    if args.pulse_ms is not None:
        default_dur = float(args.pulse_ms)
    if args.stim_start_ms is not None:
        default_onset = float(args.stim_start_ms)

    per_seed = []
    index_rows = []
    used = []
    all_metrics_rows = []

    # --- 1) Procesar cada seed ---
    for seed, rec_path in items:
        D = load_any(rec_path)
        t_ms, bold, icbf = extract_series_for_model(D, model)
        if t_ms is None:
            continue

        # simParams → onset/duración por seed
        SP = load_simparams(if_tag, args.stimulus, seed)
        onset_ms = pick(SP,
                        "stim_start_ms","stimStart_ms","stim_onset_ms","onset_ms","stimOnset_ms",
                        default=default_onset)
        dur_ms   = pick(SP,
                        "stimDuration_ms","stim_duration_ms","stimDur_ms",
                        default=default_dur)

        per_seed.append(dict(
            seed=seed, rec_path=rec_path, t_ms=t_ms, bold=bold, icbf=icbf,
            onset_ms=float(onset_ms), dur_ms=float(dur_ms)
        ))
        used.append(seed)
        index_rows.append([seed, rec_path])

    if not per_seed:
        raise SystemExit("No se pudieron extraer series para ninguna seed.")

    # Para alinear, usa longitud mínima
    min_len = min(len(x["t_ms"]) for x in per_seed)
    # time base del promedio
    T = per_seed[0]["t_ms"][:min_len]
    dt = float(np.median(np.diff(T)))

    # --- Guardar por seed (CSV, plots, métricas) ---
    for s in per_seed:
        seed = s["seed"]
        t_ms = s["t_ms"][:min_len]
        B = s["bold"][:min_len]
        I = s["icbf"][:min_len] if s["icbf"] is not None else None
        onset_ms = s["onset_ms"]
        dur_ms   = s["dur_ms"]

        seed_dir = os.path.join(seeds_dir, f"seed_{seed:02d}")
        os.makedirs(seed_dir, exist_ok=True)

        # CSV
        df = pd.DataFrame({"t_ms": t_ms, "BOLD": B})
        if I is not None:
            df["I_CBF"] = I
        df.to_csv(os.path.join(seed_dir, f"timeseries_model{model}_seed{seed}.csv"), index=False)

        # Métricas (evento real)
        m = metrics_table_event(t_ms, B, onset_ms, dur_ms, args.baseline_ms)
        m.to_csv(os.path.join(seed_dir, f"metrics_model{model}_seed{seed}.csv"), index=False)
        mm = m.copy(); mm.insert(0, "seed", seed); all_metrics_rows.append(mm)

        # Plots
        t_s = t_ms/1000.0
        # BOLD
        fig, ax = plt.subplots()
        ax.set_title(f"Modelo {model} — BOLD (seed {seed})")
        ax.plot(t_s, B, label="BOLD (Δrel.)")
        shade_event(ax, t_ms, onset_ms, dur_ms, alpha=0.12, label="estímulo")
        ax.set_xlabel("Tiempo [s]"); ax.set_ylabel("ΔBOLD (rel.)"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(seed_dir, f"plot_model{model}_seed{seed}_BOLD.png"), dpi=150)
        plt.close(fig)
        # I_CBF
        if I is not None:
            fig, ax = plt.subplots()
            ax.set_title(f"Modelo {model} — I_CBF (seed {seed})")
            ax.plot(t_s, I, label="I_CBF (norm)")
            shade_event(ax, t_ms, onset_ms, dur_ms, alpha=0.12, label="estímulo")
            ax.set_xlabel("Tiempo [s]"); ax.set_ylabel("I_CBF (a.u.)"); ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(seed_dir, f"plot_model{model}_seed{seed}_ICBF.png"), dpi=150)
            plt.close(fig)

    # Índice y métricas combinadas
    pd.DataFrame(index_rows, columns=["seed","file"]).to_csv(
        os.path.join(args.out, "seeds_index.csv"), index=False
    )
    pd.concat(all_metrics_rows, ignore_index=True).to_csv(
        os.path.join(args.out, "seeds_metrics.csv"), index=False
    )

    # --- 2) Promedio ---
    B_stack = np.stack([s["bold"][:min_len] for s in per_seed], axis=1)
    I_avail = any(s["icbf"] is not None for s in per_seed)
    if I_avail:
        I_stack = np.stack([(s["icbf"][:min_len] if s["icbf"] is not None else np.zeros(min_len)) for s in per_seed], axis=1)
    else:
        I_stack = None

    B_mean, B_std = B_stack.mean(axis=1), B_stack.std(axis=1)
    if I_stack is not None:
        I_mean, I_std = I_stack.mean(axis=1), I_stack.std(axis=1)
    else:
        I_mean = I_std = None

    # onset/duración del promedio = mediana de seeds
    onset_med = float(np.median([s["onset_ms"] for s in per_seed]))
    dur_med   = float(np.median([s["dur_ms"]   for s in per_seed]))
    # índices del evento para sombreado del promedio
    p1 = int(max(0, round((onset_med - T[0]) / dt)))
    p2 = int(min(len(T), p1 + round(dur_med / dt)))

    # CSV del promedio
    dfm = pd.DataFrame({"t_ms": T, "BOLD_mean": B_mean, "BOLD_std": B_std})
    if I_mean is not None:
        dfm["I_CBF_mean"] = I_mean; dfm["I_CBF_std"] = I_std
    out_csv = os.path.join(args.out, f"timeseries_model{model}_mean.csv")
    dfm.to_csv(out_csv, index=False)

    # Plot del promedio — BOLD
    t_s = T/1000.0
    fig, ax = plt.subplots()
    ax.set_title(f"Modelo {model} — BOLD (media de {len(per_seed)} seeds)")
    if not args.no_spaghetti:
        for k in range(B_stack.shape[1]):
            ax.plot(t_s, B_stack[:,k], alpha=0.12, linewidth=0.6)
    ax.plot(t_s, B_mean, linewidth=2.0, label="media")
    ax.fill_between(t_s, B_mean-B_std, B_mean+B_std, alpha=0.2, label="±1σ")
    ax.axvspan(t_s[p1], t_s[min(p2-1, len(t_s)-1)], alpha=0.12, label="estímulo")
    ax.set_xlabel("Tiempo [s]"); ax.set_ylabel("ΔBOLD (rel.)"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"plot_model{model}_mean_BOLD.png"), dpi=150)
    plt.close(fig)

    # Plot del promedio — I_CBF
    if I_mean is not None:
        fig, ax = plt.subplots()
        ax.set_title(f"Modelo {model} — I_CBF (media de {len(per_seed)} seeds)")
        if not args.no_spaghetti:
            for k in range(I_stack.shape[1]):
                ax.plot(t_s, I_stack[:,k], alpha=0.12, linewidth=0.6)
        ax.plot(t_s, I_mean, linewidth=2.0, label="media")
        ax.fill_between(t_s, I_mean-I_std, I_mean+I_std, alpha=0.2, label="±1σ")
        ax.axvspan(t_s[p1], t_s[min(p2-1, len(t_s)-1)], alpha=0.12, label="estímulo")
        ax.set_xlabel("Tiempo [s]"); ax.set_ylabel("I_CBF (a.u.)"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"plot_model{model}_mean_ICBF.png"), dpi=150)
        plt.close(fig)

    # Métricas del promedio (evento real mediano)
    metrics_table_event(T, B_mean, onset_med, dur_med, args.baseline_ms).to_csv(
        os.path.join(args.out, f"metrics_model{model}_mean.csv"), index=False
    )

    with open(os.path.join(args.out, f"provenance_model{model}_mean.json"), "w") as f:
        json.dump(dict(
            model=model, input_factor=args.input_factor, stimulus=args.stimulus,
            baseline_ms=args.baseline_ms,
            onset_ms_median=onset_med, duration_ms_median=dur_med,
            seeds_used=sorted(used), n_seeds=len(used)
        ), f, indent=2)

    print(f"OK. {len(used)} seeds procesadas. Salida en: {args.out}")
    print("  - CSV promedio:", out_csv)
    print("  - Subcarpetas por seed en:", os.path.join(args.out, "seeds"))
    print("  - Métricas por seed: seeds_metrics.csv")
    print("  - Gráficos: plot_model*_mean_*.png y por seed en cada subcarpeta")

if __name__ == "__main__":
    main()

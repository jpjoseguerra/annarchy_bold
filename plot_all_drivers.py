# -*- coding: utf-8 -*-
"""
plot_all_drivers.py — Plotea por seed y mean±σ los drivers/series del repo Hamkerlab.

Regla de ejes:
  - BOLD  → eje x en segundos (s)
  - neural (E/r) → eje x en milisegundos (ms)
  - demás canales (I_CBF, I_CMRO2, f_in, f_out, q, v, Eraw/Iraw) → segundos (s)

Ejemplos:
  python plot_all_drivers.py --model A --if 1.2 --stim 3 \
    --channels BOLD,I_CBF,neural \
    --out results/plots_A_sostenido1p2 --verbose

  python plot_all_drivers.py --model D --if 5 --stim 1 \
    --channels BOLD,I_CBF,I_CMRO2,neural,Eraw_I_CBF,Iraw_I_CBF \
    --out results/plots_D_pulse5 --verbose
"""
import os, re, glob, argparse, json
import numpy as np

# Backend no interactivo (útil en WSL/servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODEL2ID = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
DEFAULT_BASELINE_MS = 5000.0
DEFAULTS_BY_STIM = {
    1: dict(onset_ms=5000.0,  dur_ms=100.0),    # pulso corto (Fig. 6)
    3: dict(onset_ms=50000.0, dur_ms=20000.0),  # bloque sostenido (~Fig. 7)
}

def if_tag(v):
    return str(v).replace(".", "_")

def find_seed_files(infac, stim, verbose=True):
    tag = if_tag(infac)
    pats = [f"dataRaw/**/*recordingsB_{tag}_{stim}__*.npy"]
    files = []
    for p in pats:
        m = glob.glob(p, recursive=True)
        if verbose: print(f"[scan] patrón {p} → {len(m)} archivos")
        files += m
    files = sorted(set(files))
    rx = re.compile(rf"recordingsB_{re.escape(tag)}_{stim}__(\d+)\.npy$")
    out=[]
    for p in files:
        m = rx.search(os.path.basename(p))
        if m: out.append((int(m.group(1)), p))
    if verbose:
        print(f"[scan] seeds detectadas: {len(out)} → {[s for s,_ in out]}")
    return sorted(out, key=lambda x: x[0])

def load_simparams(infac, stim, seed, verbose=False):
    path = f"dataRaw/simulations_BOLDfromDifferentSources_simParams_{if_tag(infac)}_{stim}__{seed}.npy"
    if not os.path.exists(path):
        if verbose: print(f"[simParams] no existe: {path} (uso defaults)")
        return {}
    try:
        obj = np.load(path, allow_pickle=True).item()
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        if verbose: print(f"[simParams] error leyendo {path}: {e} (uso defaults)")
        return {}

def arr1(x):
    if isinstance(x, np.ndarray):
        return x.mean(axis=1) if x.ndim==2 else x.ravel()
    return None

def get_time_axis(n, init_ms=2000.0, dt_ms=1.0):
    return np.arange(init_ms, init_ms+n, dt_ms, dtype=float)

def shade(ax, onset_ms, dur_ms, xunits="s", **kw):
    # Dibuja el sombreado en las unidades del eje x de ese gráfico.
    if xunits == "ms":
        a0, a1 = onset_ms, onset_ms + dur_ms
    else:
        a0, a1 = onset_ms/1000.0, (onset_ms + dur_ms)/1000.0
    ax.axvspan(a0, a1, **kw)

def extract_channels(d, mid, want, verbose=False):
    """
    Devuelve dict nombre_logico -> serie (1D):
    nombres lógicos válidos: BOLD, I_CBF, I_CMRO2, neural,
      Eraw_I_CBF, Iraw_I_CBF, Eraw_I_CMRO2, Iraw_I_CMRO2, f_in, f_out, q, v
    """
    mapping = {
        "BOLD":        f"{mid};BOLD",
        "I_CBF":       f"{mid};I_CBF",
        "I_CMRO2":     f"{mid};I_CMRO2",
        "Eraw_I_CBF":  f"{mid}Eraw;I_CBF",
        "Iraw_I_CBF":  f"{mid}Iraw;I_CBF",
        "Eraw_I_CMRO2":f"{mid}Eraw;I_CMRO2",
        "Iraw_I_CMRO2":f"{mid}Iraw;I_CMRO2",
        "f_in":        f"{mid};f_in",
        "f_out":       f"{mid};f_out",
        "q":           f"{mid};q",
        "v":           f"{mid};v",
    }
    out = {}
    # Señal neuronal: E (A-C), r (D-F)
    neu_key = ("E" if mid in (1,2,3) else "r")
    if "neural" in want:
        out["neural"] = arr1(d.get(f"{mid};{neu_key}"))
        if verbose and out["neural"] is None:
            print(f"  [warn] falta {mid};{neu_key}")

    for w in want:
        if w == "neural":
            continue
        key = mapping.get(w)
        if key:
            out[w] = arr1(d.get(key))
            if verbose and out[w] is None:
                print(f"  [warn] falta {key}")
        else:
            if verbose: print(f"  [skip] canal desconocido: {w}")
    return out

def channel_xunits(name):
    """Regla de unidades por canal: BOLD→s, neural→ms, demás→s."""
    if name == "neural": return "ms"
    return "s"  # BOLD y el resto

def channel_label(name, mid):
    """Etiqueta legible del canal para el plot."""
    if name == "neural":
        return "E (excitatory)" if mid in (1,2,3) else "r (rate)"
    return name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="A..F")
    ap.add_argument("--if", dest="infac", required=True, help="1.2 o 5, etc.")
    ap.add_argument("--stim", type=int, required=True, help="1=pulso, 3=sostenido")
    ap.add_argument("--channels", default="BOLD,I_CBF,neural",
                    help="coma-separado: BOLD,I_CBF,I_CMRO2,neural,Eraw_I_CBF,Iraw_I_CBF,Eraw_I_CMRO2,Iraw_I_CMRO2,f_in,f_out,q,v")
    ap.add_argument("--baseline_ms", type=float, default=DEFAULT_BASELINE_MS)
    ap.add_argument("--stim_start_ms", type=float, default=None)
    ap.add_argument("--pulse_ms", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-spaghetti", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    M = args.model.upper()
    if M not in MODEL2ID:
        raise SystemExit("model debe ser A..F")
    mid = MODEL2ID[M]
    want = [w.strip() for w in args.channels.split(",") if w.strip()]
    print(f"[cfg] model={M} (id={mid})  if={args.infac}  stim={args.stim}  channels={want}")
    print(f"[cfg] out={args.out}")

    seeds = find_seed_files(args.infac, args.stim, verbose=True)
    if not seeds:
        raise SystemExit("No hay seeds para esa condición en dataRaw/.")

    os.makedirs(args.out, exist_ok=True)
    seeds_dir = os.path.join(args.out, "seeds"); os.makedirs(seeds_dir, exist_ok=True)

    # Defaults por tipo de estímulo (overrides CLI si se pasan)
    dflt = DEFAULTS_BY_STIM.get(args.stim, DEFAULTS_BY_STIM[1])
    onset_default = args.stim_start_ms if args.stim_start_ms is not None else dflt["onset_ms"]
    dur_default   = args.pulse_ms       if args.pulse_ms is not None       else dflt["dur_ms"]
    print(f"[cfg] defaults evento: onset={onset_default} ms  dur={dur_default} ms")

    # --- Recolectar por seed ---
    per_seed = []
    for seed, path in seeds:
        print(f"[seed {seed:02d}] leyendo {path}")
        d = np.load(path, allow_pickle=True).item()
        B = arr1(d.get(f"{mid};BOLD"))
        if B is None:
            print(f"  [warn] seed {seed}: no hay {mid};BOLD → salto")
            continue
        t_ms = get_time_axis(len(B), init_ms=2000.0, dt_ms=1.0)
        chans = extract_channels(d, mid, want, verbose=args.verbose)
        chans = {"BOLD":B, **chans}
        sp = load_simparams(args.infac, args.stim, seed, verbose=args.verbose)
        onset_ms = next((sp[k] for k in ("stim_start_ms","stimStart_ms","stim_onset_ms","onset_ms","stimOnset_ms") if k in sp), onset_default)
        dur_ms   = next((sp[k] for k in ("stimDuration_ms","stim_duration_ms","stimDur_ms") if k in sp), dur_default)
        per_seed.append(dict(seed=seed, path=path, t_ms=t_ms, chans=chans,
                             onset=float(onset_ms), dur=float(dur_ms)))
        print(f"  [ok] canales: {[k for k,v in chans.items() if v is not None]}  onset={onset_ms} dur={dur_ms}")

    if not per_seed:
        raise SystemExit("No se pudieron extraer canales para ninguna seed.")

    # Alinear por longitud mínima
    min_len = min(len(s["t_ms"]) for s in per_seed)
    T = per_seed[0]["t_ms"][:min_len]
    dt = float(np.median(np.diff(T)))
    onset_med = float(np.median([s["onset"] for s in per_seed]))
    dur_med   = float(np.median([s["dur"]   for s in per_seed]))
    print(f"[align] min_len={min_len} muestras, dt={dt} ms → onset_med={onset_med}, dur_med={dur_med}")

    # --- Plots por seed ---
    for s in per_seed:
        seed = s["seed"]
        t_ms = s["t_ms"][:min_len]
        onset_ms, dur_ms = s["onset"], s["dur"]
        sdir = os.path.join(seeds_dir, f"seed_{seed:02d}")
        os.makedirs(sdir, exist_ok=True)
        for name, arr in s["chans"].items():
            if arr is None:
                if args.verbose: print(f"  [seed {seed:02d}] skip {name} (None)")
                continue
            y = arr[:min_len]
            # Unidades por canal
            xunits = channel_xunits(name)
            if xunits == "ms":
                t_x, xlabel = t_ms, "Tiempo [ms]"
            else:
                t_x, xlabel = t_ms/1000.0, "Tiempo [s]"
            # Etiqueta de canal
            plot_label = channel_label(name, mid)
            fig, ax = plt.subplots()
            ax.set_title(f"Modelo {M} — {plot_label} (seed {seed})")
            ax.plot(t_x, y, label=plot_label)
            shade(ax, onset_ms, dur_ms, xunits=xunits, alpha=0.12, label="estímulo")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(plot_label + (" (a.u.)" if plot_label!="BOLD" else " (Δrel.)"))
            ax.legend()
            fig.tight_layout()
            out_png = os.path.join(sdir, f"seed{seed:02d}_{M}_{name}.png")
            fig.savefig(out_png, dpi=150); plt.close(fig)
            print(f"    [save] {out_png}")

    # --- Mean ± σ por canal + CSV ---
    by_channel = {}
    for s in per_seed:
        for name, arr in s["chans"].items():
            if arr is None: continue
            y = arr[:min_len]
            by_channel.setdefault(name, []).append(y)

    os.makedirs(os.path.join(args.out, "means_csv"), exist_ok=True)
    for name, arrs in by_channel.items():
        X = np.stack(arrs, axis=1)
        mu, sd = X.mean(axis=1), X.std(axis=1)
        # Unidades por canal
        xunits = channel_xunits(name)
        if xunits == "ms":
            t_x, xlabel = T, "Tiempo [ms]"
        else:
            t_x, xlabel = T/1000.0, "Tiempo [s]"
        # Etiqueta de canal
        plot_label = channel_label(name, mid)
        fig, ax = plt.subplots()
        ax.set_title(f"Modelo {M} — {plot_label} (media de {X.shape[1]} seeds)")
        if not args.no_spaghetti:
            for k in range(X.shape[1]):
                ax.plot(t_x, X[:,k], alpha=0.08, linewidth=0.6)
        ax.plot(t_x, mu, linewidth=2.0, label=f"{plot_label} (mean)")
        ax.fill_between(t_x, mu-sd, mu+sd, alpha=0.2, label="±1σ")
        shade(ax, onset_med, dur_med, xunits=xunits, alpha=0.12, label="estímulo (mediana)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(plot_label + (" (a.u.)" if plot_label!="BOLD" else " (Δrel.)"))
        ax.legend()
        fig.tight_layout()
        out_png = os.path.join(args.out, f"mean_{M}_{name}.png")
        fig.savefig(out_png, dpi=150); plt.close(fig)
        print(f"[mean save] {out_png}")

        # CSV de mean (tiempo SIEMPRE en ms en los CSVs)
        df = pd.DataFrame({"t_ms": T, f"{name}_mean": mu, f"{name}_std": sd})
        out_csv = os.path.join(args.out, "means_csv", f"mean_{M}_{name}.csv")
        df.to_csv(out_csv, index=False)
        print(f"[mean save] {out_csv}")

    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(dict(
            model=M, model_id=mid, input_factor=args.infac, stimulus=args.stim,
            channels=want, baseline_ms=args.baseline_ms,
            onset_ms_median=onset_med, duration_ms_median=dur_med,
            n_seeds=len(per_seed), seeds=[s["seed"] for s in per_seed],
            axis_rule=dict(BOLD="s", neural="ms", others="s")
        ), f, indent=2)
    print(f"\nOK. Seeds: {len(per_seed)} → {args.out}")
    print(" - Subcarpetas por seed con PNGs")
    print(" - PNGs de mean por canal: mean_<M>_<canal>.png")
    print(" - CSVs de mean por canal (t_ms siempre en ms): means_csv/*.csv")

if __name__ == "__main__":
    main()

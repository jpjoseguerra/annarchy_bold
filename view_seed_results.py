# -*- coding: utf-8 -*-
"""
Visualiza y exporta la serie de tiempo de UNA seed del repo Hamkerlab:
- Soporta modelos A..F (A=1, B=2, ..., F=6).
- Lee onset/duración reales desde simParams cuando están presentes.
- Permite overrides (--stim_start_ms, --pulse_ms, --baseline_ms).
- Exporta CSV (t_ms, BOLD, I_CBF, I_CMRO2*, Eraw/Iraw*, f_in/f_out/q/v si existen)
  y guarda dos PNG (BOLD, I_CBF) con la franja de estímulo correcta.

Uso típico:
  # Sostenido 1.2 (Fig. tipo 7), seed 38:
  python view_seed_results.py --model A --if 1.2 --stim 3 --simID 38 \
    --out results/seeds_A_sostenido1p2/seed_38

  # Pulso ×5 (Fig. 6), seed 0:
  python view_seed_results.py --model B --if 5 --stim 1 --simID 0 \
    --out results/seeds_B_pulse5/seed_00
"""
import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IDMAP = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}

def arr1(x):
    if isinstance(x, np.ndarray):
        return x.mean(axis=1) if x.ndim==2 else x.ravel()
    return None

def guess_if_tag(v):
    # 1.2 -> "1_2", 5 -> "5"
    s = str(v)
    return s.replace(".", "_")

def load_simparams(path):
    sp = {}
    if os.path.exists(path):
        try:
            sp = np.load(path, allow_pickle=True).item()
        except Exception:
            sp = {}
    return sp if isinstance(sp, dict) else {}

def pick(key_dict, *names, default=None):
    for n in names:
        if n in key_dict: return key_dict[n]
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False, default="A", help="A..F (default A)")
    ap.add_argument("--if", dest="infac", required=True, help="input factor, ej. 1.2 o 5")
    ap.add_argument("--stim", type=int, required=True, help="1=pulso, 3=sostenido")
    ap.add_argument("--simID", type=int, required=True, help="seed")
    ap.add_argument("--out", required=True, help="carpeta de salida")
    # overrides opcionales
    ap.add_argument("--stim_start_ms", type=float, default=None, help="onset estímulo (ms)")
    ap.add_argument("--pulse_ms", type=float, default=None, help="duración estímulo/bloque (ms)")
    ap.add_argument("--baseline_ms", type=float, default=None, help="ventana de normalización (ms)")
    args = ap.parse_args()

    model = args.model.upper()
    mid = IDMAP.get(model, 1)
    if_tag = guess_if_tag(args.infac)
    seed   = int(args.simID)
    stim   = int(args.stim)

    # rutas del repo
    rec_path = f"dataRaw/simulations_BOLDfromDifferentSources_recordingsB_{if_tag}_{stim}__{seed}.npy"
    par_path = f"dataRaw/simulations_BOLDfromDifferentSources_simParams_{if_tag}_{stim}__{seed}.npy"
    if not os.path.exists(rec_path):
        raise SystemExit(f"✖ No existe {rec_path}")

    # cargar recordings
    D = np.load(rec_path, allow_pickle=True).item()
    kB = f"{mid};BOLD"
    kI = f"{mid};I_CBF"
    kM = f"{mid};I_CMRO2"  # solo D..F
    kEin = f"{mid}Eraw;I_CBF"
    kIin = f"{mid}Iraw;I_CBF"
    kEinM = f"{mid}Eraw;I_CMRO2"
    kIinM = f"{mid}Iraw;I_CMRO2"

    B = arr1(D.get(kB))
    I = arr1(D.get(kI))
    M = arr1(D.get(kM)) if (kM in D) else None
    if B is None:
        raise SystemExit(f"✖ No hallé '{kB}' en {rec_path}")

    # extras opcionales
    f_in  = arr1(D.get(f"{mid};f_in"))
    f_out = arr1(D.get(f"{mid};f_out"))
    q     = arr1(D.get(f"{mid};q"))
    v     = arr1(D.get(f"{mid};v"))
    Evar  = arr1(D.get(f"{mid};E"))
    Irate = arr1(D.get(f"{mid};r"))
    Ein   = arr1(D.get(kEin))
    Iin   = arr1(D.get(kIin))
    EinM  = arr1(D.get(kEinM)) if (kEinM in D) else None
    IinM  = arr1(D.get(kIinM)) if (kIinM in D) else None

    # tiempo (dt=1 ms). Los recordings suelen iniciar en 2000 ms; si hay init en simparams, úsalo.
    SP = load_simparams(par_path)
    init_ms = pick(SP, "init_ms", "INIT_ms", "initTime_ms", default=2000.0)
    L = len(B)
    t_ms = np.arange(init_ms, init_ms + L, 1.0)

    # onset/duración del estímulo (leer simparams; si no hay, defaults razonables por protocolo)
    # nombres posibles en simparams:
    stim_start = args.stim_start_ms
    stim_dur   = args.pulse_ms

    if stim_start is None:
        stim_start = pick(SP,
                          "stim_start_ms", "stimStart_ms", "stim_onset_ms",
                          "onset_ms", "stimOnset_ms",
                          default=(5000.0 if stim==1 else 50000.0))
    if stim_dur is None:
        stim_dur = pick(SP,
                        "stimDuration_ms", "stim_duration_ms", "stimDur_ms",
                        default=(100.0 if stim==1 else 20000.0))

    # baseline para normalización/métricas
    baseline_ms = args.baseline_ms if args.baseline_ms is not None else 5000.0
    dt = float(np.median(np.diff(t_ms)))
    b0 = max(1, int(baseline_ms/dt))

    # dataframe y export
    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame({"t_ms": t_ms, "BOLD": B})
    if I is not None: df["I_CBF"] = I
    if M is not None: df["I_CMRO2"] = M
    if Ein is not None: df["Eraw_I_CBF"] = Ein
    if Iin is not None: df["Iraw_I_CBF"] = Iin
    if EinM is not None: df["Eraw_I_CMRO2"] = EinM
    if IinM is not None: df["Iraw_I_CMRO2"] = IinM
    if f_in is not None:  df["f_in"] = f_in
    if f_out is not None: df["f_out"] = f_out
    if q is not None:     df["q"] = q
    if v is not None:     df["v"] = v
    if Evar is not None:  df["E"] = Evar
    if Irate is not None: df["r"] = Irate

    csv_path = os.path.join(args.out, f"timeseries_seed{seed}_model{model}.csv")
    df.to_csv(csv_path, index=False)

    # métricas simples
    stim_mask = (t_ms >= stim_start) & (t_ms <= (stim_start + stim_dur))
    post_mask = (t_ms > (stim_start + stim_dur))
    metrics = []
    metrics.append(("dt_ms", dt))
    metrics.append(("len_samples", int(L)))
    metrics.append(("series_duration_s", float((t_ms[-1]-t_ms[0])/1000.0)))
    metrics.append(("baseline_mean", float(B[:b0].mean())))
    metrics.append(("baseline_std",  float(B[:b0].std())))
    if stim_mask.any():
        metrics.append(("BOLD_mean_during_stim", float(B[stim_mask].mean())))
        metrics.append(("BOLD_max_during_stim", float(B[stim_mask].max())))
        metrics.append(("BOLD_min_during_stim", float(B[stim_mask].min())))
    if post_mask.any():
        # pico global después del onset (captura respuesta hemodinámica)
        Bb = B[post_mask]; Tb = t_ms[post_mask]
        imx = int(np.argmax(Bb)); imn = int(np.argmin(Bb))
        metrics.append(("BOLD_peak_val", float(Bb[imx])))
        metrics.append(("BOLD_peak_time_s", float(Tb[imx]/1000.0)))
        metrics.append(("BOLD_min_val", float(Bb[imn])))
        metrics.append(("BOLD_min_time_s", float(Tb[imn]/1000.0)))

    met = pd.DataFrame(metrics, columns=["metric","value"])
    met.to_csv(os.path.join(args.out, f"metrics_seed{seed}_model{model}.csv"), index=False)

    # Plots
    t_s = t_ms/1000.0
    stim_s = stim_start/1000.0
    stim_end_s = (stim_start + stim_dur)/1000.0

    plt.figure()
    plt.title(f"Modelo {model} — BOLD (seed {seed})")
    plt.plot(t_s, B, label="BOLD (Δrel.)")
    plt.axvspan(stim_s, stim_end_s, alpha=0.12, label="estímulo")
    plt.xlabel("Tiempo [s]"); plt.ylabel("ΔBOLD (rel.)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"plot_seed{seed}_model{model}_BOLD.png"), dpi=140)
    plt.close()

    if I is not None:
        plt.figure()
        plt.title(f"Modelo {model} — I_CBF (seed {seed})")
        plt.plot(t_s, I, label="I_CBF (norm)")
        plt.axvspan(stim_s, stim_end_s, alpha=0.12, label="estímulo")
        plt.xlabel("Tiempo [s]"); plt.ylabel("I_CBF (a.u.)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"plot_seed{seed}_model{model}_ICBF.png"), dpi=140)
        plt.close()

    if M is not None:
        plt.figure()
        plt.title(f"Modelo {model} — I_CMRO2 (seed {seed})")
        plt.plot(t_s, M, label="I_CMRO2 (norm)")
        plt.axvspan(stim_s, stim_end_s, alpha=0.12, label="estímulo")
        plt.xlabel("Tiempo [s]"); plt.ylabel("I_CMRO2 (a.u.)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"plot_seed{seed}_model{model}_ICMRO2.png"), dpi=140)
        plt.close()

    # trazabilidad
    with open(os.path.join(args.out, f"provenance_seed{seed}.json"), "w") as f:
        json.dump(dict(
            model=model, model_id=mid,
            input_factor=args.infac, stimulus=stim, seed=seed,
            rec_path=rec_path, par_path=par_path,
            init_ms=float(init_ms),
            stim_start_ms=float(stim_start), stim_duration_ms=float(stim_dur),
            baseline_ms=float(baseline_ms),
            keys_present=sorted([k for k in D.keys() if isinstance(k, str)]),
        ), f, indent=2)

    print("OK. Resultados en:", args.out)
    print("  -", os.path.basename(csv_path))
    print("  - plots: _BOLD.png", ("_ICBF.png" if I is not None else ""), ("_ICMRO2.png" if M is not None else ""))
    print("  -", f"metrics_seed{seed}_model{model}.csv")
    print("  - provenance_seed{seed}.json")

if __name__ == "__main__":
    main()

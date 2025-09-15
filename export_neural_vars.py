# export_neural_vars.py
import os, re, glob, argparse
import numpy as np, pandas as pd

MODEL2ID = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}

def if_tag(v): return str(v).replace(".", "_")

def find_seeds(ifac, stim):
    tag = if_tag(ifac)
    pats = [f"dataRaw/**/*recordingsB_{tag}_{stim}__*.npy"]
    files=[]
    for p in pats: files += glob.glob(p, recursive=True)
    files = sorted(set(files))
    rx = re.compile(rf"recordingsB_{re.escape(tag)}_{stim}__(\d+)\.npy$")
    seeds=[]
    for p in files:
        m=rx.search(os.path.basename(p))
        if m: seeds.append((int(m.group(1)), p))
    return sorted(seeds, key=lambda x:x[0])

def arr1(x):
    if isinstance(x, np.ndarray):
        return x.mean(axis=1) if x.ndim==2 else x.ravel()
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--if", dest="infac", required=True, help="1.2 ó 5, etc.")
    ap.add_argument("--stim", type=int, required=True, help="1=pulso, 3=sostenido")
    ap.add_argument("--model", default="ALL", help="A..F o ALL")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = list(MODEL2ID.keys()) if args.model.upper()=="ALL" else [args.model.upper()]
    seeds = find_seeds(args.infac, args.stim)
    if not seeds:
        raise SystemExit("No encontré seeds para esa condición en dataRaw/")

    os.makedirs(args.out, exist_ok=True)
    print(f"Exportando modelos {models} para {len(seeds)} seeds → {args.out}")

    for M in models:
        mid = MODEL2ID[M]
        # qué clave neuronal usar
        neu_key = "E" if M in ("A","B","C") else "r"
        rows_mean = []  # para mean posterior
        dfs = []
        for seed, path in seeds:
            d = np.load(path, allow_pickle=True).item()
            x = arr1(d.get(f"{mid};{neu_key}"))
            if x is None:
                print(f"⚠ seed {seed}: no hay {mid};{neu_key} en {os.path.basename(path)}")
                continue
            # tiempo (dt=1 ms; recordings suelen empezar en 2000 ms)
            t = np.arange(2000.0, 2000.0+len(x), 1.0, dtype=float)
            df = pd.DataFrame({"t_ms": t, neu_key: x})
            odir = os.path.join(args.out, f"{M}_seeds"); os.makedirs(odir, exist_ok=True)
            df.to_csv(os.path.join(odir, f"seed{seed:02d}_model{M}_neural.csv"), index=False)
            dfs.append(df[[neu_key]].rename(columns={neu_key: f"{neu_key}_seed{seed:02d}"}))

        # mean/std por modelo si hay al menos una seed válida
        if dfs:
            # alinear por longitud mínima
            min_len = min(len(df) for df in dfs)
            t = np.arange(2000.0, 2000.0+min_len, 1.0, dtype=float)
            X = np.stack([df.iloc[:min_len,0].values for df in dfs], axis=1)
            neu_mean = X.mean(axis=1); neu_std = X.std(axis=1)
            out_df = pd.DataFrame({"t_ms": t, f"{neu_key}_mean": neu_mean, f"{neu_key}_std": neu_std})
            out_df.to_csv(os.path.join(args.out, f"model{M}_neural_mean.csv"), index=False)
            print(f"✓ model {M}: {len(dfs)} seeds → model{M}_neural_mean.csv")
        else:
            print(f"✖ model {M}: no se exportó nada (faltan claves/series).")

if __name__ == "__main__":
    main()

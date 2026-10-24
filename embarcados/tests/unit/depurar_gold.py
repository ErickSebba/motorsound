"""Uso: python depurar_<x>.py <float|double|float-sosd>. Depuração (não é a verificação formal): roda p11_host caso em todos os casos do ouro e
resume o erro por etapa."""
import json, subprocess, sys
from pathlib import Path
import numpy as np

EMB = Path(__file__).resolve().parents[2]
build = sys.argv[1]
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
exe = EMB / f"build-{build}" / "p11_host.exe"
out_root = Path(sys.argv[3]) if len(sys.argv) > 3 else EMB / f"build-{build}" / "c_out_depuracao"
man = json.load(open(EMB / "gold" / "manifest.json"))
casos = man["casos"][:lim]

def ler(d, et):
    e = json.load(open(d / "etapas.json"))[et]
    dt = {"float64": np.float64, "float32": np.float32, "int32": np.int32}[e["dtype"]]
    return np.fromfile(d / e["arquivo"], dtype=dt).reshape(e["shape"]).astype(np.float64)

stats = {}
disc = {"vit_path": 0, "pred": 0}
for c in casos:
    gd = EMB / "gold" / "casos" / c["id"]
    od = out_root / c["id"]
    r = subprocess.run([str(exe), "caso", str(gd), str(EMB / "gold" / "modelos" / (c["modelo"] + ".bin")), str(od)],
                       capture_output=True, text=True)
    if r.returncode:
        print("FALHA", c["id"], r.stderr); continue
    for et in json.load(open(gd / "etapas.json")):
        if et == "audio":
            continue
        g = ler(gd, et); o = ler(od, et)
        if g.shape != o.shape:
            print("shape", c["id"], et, g.shape, o.shape); continue
        fin = np.isfinite(g)
        if not np.array_equal(fin, np.isfinite(o)):
            stats.setdefault(et, []).append(np.inf); continue
        d = np.abs(g[fin] - o[fin]); sc = np.max(np.abs(g[fin])) if fin.any() else 1
        stats.setdefault(et, []).append(float(d.max() / (sc + 1e-300)) if d.size else 0.0)
    if not np.array_equal(ler(gd, "vit_path"), ler(od, "vit_path")):
        disc["vit_path"] += 1
    if np.argmax(ler(gd, "probs")) != np.argmax(ler(od, "probs")):
        disc["pred"] += 1
print(f"{len(casos)} casos, build {build}")
for et, v in stats.items():
    v = np.array(v)
    print(f"{et:14s} max_rel_med={np.median(v):.2e} max={v.max():.2e} pior={casos[int(np.argmax(v))]['id']}")
print(disc)

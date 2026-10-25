"""Auditoria M3: queda "motor fora" comparada igual por igual no A2212.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\aud_motor_igual.py

No protocolo `motor` completo a classe Bearing (só no M4) fica sem treino no fold M4, o que
limita a acurácia balanceada a ~2/3 por construção. Para medir só o efeito do motor, este script
restringe os DOIS protocolos às mesmas janelas: classes verdadeiras Healthy/Propeller dos motores
M1–M3. No protocolo `arquivo` isso é um filtro das predições do StratifiedGroupKFold por arquivo;
no `motor` são os folds M1–M3 (= `motor_sem_M4` de diagnostico.py). Os modelos e as predições
são os de `diagnostico.py` (lidos de results/predictions/diagnostico_janelas_<ts>.csv, o mais
recente); as métricas e o bootstrap são `diagnostico.metricas` e `diagnostico.bootstrap`
(reamostragem por arquivo, estratificada pela classe, mesma semente). Nenhum classificador novo.

A queda (arquivo − motor) ganha também um IC por bootstrap PAREADO por arquivo (os mesmos
arquivos aparecem nos dois protocolos), com `diagnostico.metricas` em cada réplica.

Saída: results/csv/aud_motor_igual_<ts>.csv (+ _meta.json): uma linha por (variante, protocolo)
e uma linha de queda por variante.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import gerar_artigo_comum as C  # noqa: E402
import diagnostico as DG  # noqa: E402  (só funções; main() não roda)
from bancada_comum import carimbo, commit  # noqa: E402

CLASSES = ["Healthy", "Propeller", "Bearing"]       # = runner.classes_de("a2212")
CLASSES_HP = ["Healthy", "Propeller"]
MOTORES = ["M1", "M2", "M3"]
VARIANTES = ["p04_classicas", "p11_decifrae_harmsum", "b00_rmspico_norm", "b00_rmspico_bruto"]


def _arrays(d):
    y = np.array([CLASSES.index(c) for c in d.classe])
    pr = np.array([CLASSES.index(c) for c in d.pred])
    P = d[[f"p_{c}" for c in CLASSES]].to_numpy(float)
    return y, pr, P, d.grupo.astype(str).to_numpy()


def main():
    t_ini = time.perf_counter()
    ts = carimbo()
    fpred = C.mais_recente(C.RES / "predictions", "diagnostico_janelas_*.csv")
    J = pd.read_csv(fpred, low_memory=False)
    J = J[J.base == "a2212"]
    assert set(J.classe) == set(CLASSES)
    assert J[J.classe == "Bearing"].motor.eq("M4").all(), "Bearing fora do M4?"
    linhas, quedas = [], []
    for v in VARIANTES:
        sel = {}
        for prot in ("arquivo", "motor"):
            d = J[(J.variante == v) & (J.split == prot)]
            todo = d
            d = d[d.classe.isin(CLASSES_HP) & d.motor.isin(MOTORES)]
            if prot == "motor":
                assert set(d.fold) == {f"a2212_motor_{m}" for m in MOTORES}
            sel[prot] = d
            for escopo, dd in (("HP_M1M3", d), ("completo", todo)):
                y, pr, P, g = _arrays(dd)
                m = DG.metricas(y, pr, P, len(CLASSES))
                ic, gpc = DG.bootstrap(y, pr, P, len(CLASSES), g)
                linhas.append({
                    "ts": ts, "linha": "protocolo", "variante": v, "protocolo": prot,
                    "escopo": escopo, "n_janelas": int(len(dd)),
                    "n_arquivos": int(len(np.unique(g))),
                    "arquivos_por_classe": json.dumps({CLASSES[c]: k for c, k in gpc.items()}),
                    "acuracia_balanceada": m["acuracia_balanceada"],
                    "ic95_lo": ic["acuracia_balanceada"][0],
                    "ic95_hi": ic["acuracia_balanceada"][1],
                    "rec_Healthy": float(m["sensibilidade"][0]),
                    "rec_Propeller": float(m["sensibilidade"][1]),
                    "frac_pred_Bearing": float(np.mean(pr == 2)),
                    "n_boot": DG.N_BOOT, "seed_boot": DG.SEED_BOOT, "fonte": C.rel(fpred)})
        # queda igual por igual + bootstrap pareado por arquivo
        a, b = sel["arquivo"], sel["motor"]
        assert set(a.rec_id) == set(b.rec_id), "protocolos com arquivos diferentes"
        ya, pa, Pa, ga = _arrays(a)
        yb, pb, Pb, gb = _arrays(b)
        bal = lambda y, p, P: DG.metricas(y, p, P, len(CLASSES))["acuracia_balanceada"]  # noqa
        q = bal(ya, pa, Pa) - bal(yb, pb, Pb)
        ia = {k: np.flatnonzero(ga == k) for k in np.unique(ga)}
        ib = {k: np.flatnonzero(gb == k) for k in np.unique(gb)}
        cl = {k: int(ya[ia[k][0]]) for k in ia}
        por_c = {}
        for k, c in cl.items():
            por_c.setdefault(c, []).append(k)
        rng = np.random.default_rng(DG.SEED_BOOT)
        reps = np.empty(DG.N_BOOT)
        for r in range(DG.N_BOOT):
            s = [k for c, ks in sorted(por_c.items()) for k in rng.choice(ks, size=len(ks))]
            xa = np.concatenate([ia[k] for k in s])
            xb = np.concatenate([ib[k] for k in s])
            reps[r] = bal(ya[xa], pa[xa], Pa[xa]) - bal(yb[xb], pb[xb], Pb[xb])
        lo, hi = np.percentile(reps, [2.5, 97.5])
        full = {p: next(l_ for l_ in linhas if l_["variante"] == v and l_["protocolo"] == p
                        and l_["escopo"] == "completo")["acuracia_balanceada"]
                for p in ("arquivo", "motor")}
        quedas.append({
            "ts": ts, "linha": "queda", "variante": v, "protocolo": "arquivo-motor",
            "escopo": "HP_M1M3", "n_janelas": int(len(a)), "n_arquivos": int(len(ia)),
            "acuracia_balanceada": q, "ic95_lo": float(lo), "ic95_hi": float(hi),
            "queda_completa": full["arquivo"] - full["motor"],
            "n_boot": DG.N_BOOT, "seed_boot": DG.SEED_BOOT, "fonte": C.rel(fpred),
            "nota": "IC por bootstrap pareado por arquivo, estratificado pela classe"})
    D = pd.DataFrame(linhas + quedas)
    out = C.RES / "csv" / f"aud_motor_igual_{ts}.csv"
    meta_p = out.with_name(out.stem + "_meta.json")
    for p in (out, meta_p):
        if p.exists():
            raise FileExistsError(p)
    D.to_csv(out, index=False)
    meta = {"script": "embarcados/src/python/aud_motor_igual.py",
            "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"), "projeto": commit(),
            "fonte": {C.rel(fpred): C.sha256(fpred)},
            "fonte_meta": C.ler_json(fpred.with_name(fpred.stem + "_meta.json")).get("projeto"),
            "seed_boot": DG.SEED_BOOT, "n_boot": DG.N_BOOT,
            "restricao": "classe verdadeira em {Healthy, Propeller} e motor em {M1, M2, M3}; "
                         "Bearing continua podendo ser predito (conta como erro)",
            "metricas": "embarcados/src/python/diagnostico.py::metricas e ::bootstrap",
            "duracao_s": round(time.perf_counter() - t_ini, 1)}
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(D[["linha", "variante", "protocolo", "escopo", "n_janelas", "n_arquivos",
             "acuracia_balanceada", "ic95_lo", "ic95_hi"]].round(4).to_string(index=False))
    print(f"escrito {out}")


if __name__ == "__main__":
    main()

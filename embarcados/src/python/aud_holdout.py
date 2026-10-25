"""Auditoria M2: teste de equivalência Python × C FORA DA AMOSTRA do ouro.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\aud_holdout.py [--builds float double]

Torna reproduzível o `f_holdout.py` do auditor (reports/independent_scientific_audit.md, M2):
sorteia, com semente fixa, 12 janelas do `train/` do IDMT e 12 do A2212 que NÃO estão em
`gold/manifest.json` (chave (rec_id, idx)), gera o ouro delas com a MESMA instrumentação do ouro
oficial (`ouro_p11.etapas_p11` + `gerar_ouro.gravar`, mesmo layout de `etapas.json`), roda
`p11_host caso` dos builds float e double e compara com o comparador congelado
(`tests/equivalence/comparar.py::rodar`, importado sem modificação) e as tolerâncias congeladas
(`configs/embedded/tolerancias.json`, não alteradas).

Os modelos são os do ouro (`gold/modelos/idmt_bloco0.bin`, `a2212_arq0.bin`), reconstruídos a
partir do `.json` correspondente (mesmos float64 do `.bin`) para a referência Python.
As janelas sorteadas podem pertencer ao treino do modelo exportado (como no rascunho do auditor):
o teste é de equivalência numérica, não de desempenho.

Saídas (nunca sobrescreve):
  gold/holdout_<ts>/                                ouro + saídas do C (gitignored, regerável)
  results/csv/aud_holdout_<ts>.csv                  uma linha por (build, janela, etapa)
  results/csv/aud_holdout_<ts>_resumo.csv           resumo por (build, base) e por build
  results/csv/aud_holdout_<ts>_meta.json
Só lê o train/ do IDMT (io.py trava o test/) e o A2212.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
warnings.filterwarnings("ignore")

import ouro_p11 as O  # noqa: E402  (insere a raiz no sys.path)
import gerar_ouro as GO  # noqa: E402  (só funções; main() não roda)
from bancada_comum import carimbo, commit  # noqa: E402
from motorsound import io, runner  # noqa: E402
from motorsound.classificador import CossenoPrototipos  # noqa: E402
from motorsound.config import AVAL_A2212, AVAL_IDMT  # noqa: E402

EMB = AQUI.parents[1]
GOLD = EMB / "gold"
TOL = EMB / "configs" / "embedded" / "tolerancias.json"
CSV_DIR = EMB / "results" / "csv"
SEED = 7            # = semente do rascunho do auditor (F\f_holdout.py), registrada aqui
N_POR_BASE = 12
BASES = [("idmt", AVAL_IDMT, "idmt_bloco0"), ("a2212", AVAL_A2212, "a2212_arq0")]


def _comparar():
    spec = importlib.util.spec_from_file_location(
        "comparar_congelado", EMB / "tests" / "equivalence" / "comparar.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def modelo_do_ouro(nome):
    """(clf, C) idêntico ao exportado no ouro, a partir de gold/modelos/<nome>.json."""
    j = json.loads((GOLD / "modelos" / f"{nome}.json").read_text("utf-8"))
    clf = CossenoPrototipos(temp=float(j["temp"]))
    clf.classes = list(range(int(j["n_classes"])))
    clf.mu = np.asarray(j["mu"], float)
    clf.sd = np.asarray(j["sd"], float)
    clf.M = np.asarray(j["M"], float)
    # conferência contra o .bin (o que o C lê)
    b = _comparar().ler_modelo(GOLD / "modelos" / f"{nome}.bin")
    assert np.array_equal(b["mu"], clf.mu) and np.array_equal(b["sd"], clf.sd) \
        and np.array_equal(b["M"], clf.M) and b["temp"] == clf.temp, f"{nome}: .json != .bin"
    return clf, int(j["n_classes"])


def sortear(man):
    no_ouro = {(c.get("rec_id"), c.get("idx")) for c in man["casos"]}
    rng = np.random.default_rng(SEED)
    escolha = []
    for base, aval, fold in BASES:
        segs = runner.segmentos_de(base, aval)
        io.registrar(io.gravacoes(base, "train") if base == "idmt" else io.gravacoes(base))
        cand = [i for i, s in enumerate(segs) if (s.rec_id, s.idx) not in no_ouro]
        pick = rng.choice(cand, N_POR_BASE, replace=False)
        escolha += [(base, aval, fold, int(i), segs[int(i)], len(cand)) for i in pick]
    return escolha


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=["float", "double"])
    a = ap.parse_args(argv)
    t_ini = time.perf_counter()
    ts = carimbo()
    CP = _comparar()
    man_ouro = json.loads((GOLD / "manifest.json").read_text("utf-8"))
    H = GOLD / f"holdout_{ts}"
    if H.exists():
        raise FileExistsError(H)
    (H / "modelos").mkdir(parents=True)
    modelos = {}
    for _, _, fold in BASES:
        for ext in (".bin", ".json"):
            shutil.copy2(GOLD / "modelos" / f"{fold}{ext}", H / "modelos" / f"{fold}{ext}")
        modelos[fold] = modelo_do_ouro(fold)

    casos, falhas_chk = [], []
    for base, aval, fold, i, s, n_cand in sortear(man_ouro):
        cid = f"{base}_holdout_{i:04d}"
        pasta = H / "casos" / cid
        pasta.mkdir(parents=True)
        audio = np.ascontiguousarray(GO.audio_seg(s, 16000), dtype="<f4")
        assert np.array_equal(audio, io.audio_segmento(s, 16000))
        audio.tofile(pasta / "audio.f32")
        et, chk = O.etapas_p11(audio, modelos[fold], aval)
        desc = {"audio": {"arquivo": "audio.f32", "dtype": "float32", "shape": [len(audio)]}}
        for k, v in et.items():
            desc[k] = GO.gravar(pasta, k, v)
        (pasta / "etapas.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
        ruins = [k for k, v in chk.items() if not v and k != "vit_score==min(delta)_exato"]
        if ruins:
            falhas_chk.append(f"{cid}: {ruins}")
        casos.append({"id": cid, "tipo": "holdout", "base": base, "rec_id": s.rec_id,
                      "idx": s.idx, "t0": s.t0, "t1": s.t1, "classe": s.classe,
                      "indice_segmento": i, "n_candidatos": n_cand, "fold": fold,
                      "modelo": fold, "sha256_audio": _sha(pasta / "audio.f32"),
                      "pred_ref": int(np.argmax(et["probs"])), "checagens": chk})
    man = {"gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "gerador": "embarcados/src/python/aud_holdout.py", "seed": SEED,
           "casos": casos, "seqs": []}
    (H / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str),
                                     encoding="utf-8")

    dfs, hosts = [], {}
    for b in a.builds:
        host = CP.achar_host(b)
        if host is None:
            raise SystemExit(f"p11_host do build {b} não encontrado")
        hosts[b] = {"exe": str(host), "sha256": _sha(host),
                    "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(host.stat().st_mtime))}
        df, _ = CP.rodar(b, H, H / "c_out" / b, TOL, None, gerar=True, saida_csv=None,
                         verbose=False)
        dfs.append(df)
    D = pd.concat(dfs, ignore_index=True)
    info = pd.DataFrame(casos)[["id", "rec_id", "idx", "classe", "indice_segmento"]]
    D = D.merge(info, on="id", how="left")
    D.insert(0, "ts", ts)
    D["seed"] = SEED

    def resumo(g):
        cs = g[~g.etapa.str.contains(r"\[")]
        pv = lambda et: cs[cs.etapa == et].set_index("id")  # noqa: E731
        vp, dec, pr, ft = pv("vit_path"), pv("decisao"), pv("probs"), pv("features")
        fk = g[g.etapa.str.startswith("features[")]
        return pd.Series({
            "n_janelas": int(cs.id.nunique()),
            "caminho_identico": int(((vp.status == "PASS") & (vp.n_viol == 0)).sum()),
            "decisao_igual": int((dec.concordancia == 1).sum()),
            "n_features_fora_tol": int(fk.status.isin(["FAIL", "FAIL_JUSANTE"]).sum()),
            "n_janelas_com_feature_fora": int((ft.status != "PASS").sum()),
            "razao_tol_max_features": float(ft.razao_tol.max()),
            "max_abs_dp": float(pr.err_max.max()),
            "n_etapas_fail": int(cs.status.isin(["FAIL", "FAIL_JUSANTE"]).sum()),
            "n_etapas_pass_empate": int((cs.status == "PASS_EMPATE").sum()),
            "n_etapas_ausente": int((cs.status == "AUSENTE").sum()),
            "n_linhas_etapa": int(len(cs)),
        })
    R = pd.concat([D.groupby(["build", "base"]).apply(resumo).reset_index(),
                   D.groupby("build").apply(resumo).reset_index().assign(base="_todas")],
                  ignore_index=True)
    for k in R.columns:
        if k.startswith("n_") or k in ("caminho_identico", "decisao_igual"):
            R[k] = R[k].astype(int)
    R.insert(0, "ts", ts)
    R["seed"] = SEED

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out = CSV_DIR / f"aud_holdout_{ts}.csv"
    for p in (out, out.with_name(out.stem + "_resumo.csv"), out.with_name(out.stem + "_meta.json")):
        if p.exists():
            raise FileExistsError(p)
    D.to_csv(out, index=False)
    R.to_csv(out.with_name(out.stem + "_resumo.csv"), index=False)
    meta = {"script": "embarcados/src/python/aud_holdout.py", "gerado_em": man["gerado_em"],
            "projeto": commit(), "seed": SEED, "n_por_base": N_POR_BASE,
            "sorteio": "np.random.default_rng(seed).choice(candidatos, 12, replace=False), "
                       "IDMT primeiro e depois A2212; candidatos = runner.segmentos_de(base, aval) "
                       "cujo (rec_id, idx) não está em gold/manifest.json",
            "modelos": {f: {"bin_sha256": _sha(GOLD / "modelos" / f"{f}.bin")} for _, _, f in BASES},
            "gold_manifest_sha256": _sha(GOLD / "manifest.json"),
            "holdout_dir": str(H), "hosts": hosts,
            "comparador": "embarcados/tests/equivalence/comparar.py::rodar (importado, inalterado)",
            "sha256_comparador": _sha(EMB / "tests" / "equivalence" / "comparar.py"),
            "tolerancias": str(TOL), "sha256_tolerancias": _sha(TOL),
            "falhas_checagens_ouro": falhas_chk,
            "duracao_s": round(time.perf_counter() - t_ini, 1),
            "nota": "test/ do IDMT não lido; ouro gerado com ouro_p11.etapas_p11"}
    out.with_name(out.stem + "_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(R.to_string(index=False))
    if falhas_chk:
        print("FALHAS nas checagens do ouro:\n" + "\n".join(falhas_chk))
    print(f"escrito {out}")


if __name__ == "__main__":
    main()

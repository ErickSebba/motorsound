"""Inventário do subconjunto baixado do MIMII e do MIMII DUE.

Uso (da raiz do projeto):
    .venv/Scripts/python.exe -I scripts/inventario_mimii.py "<raiz dos dados>/extraido" \
        --csv out/indices/mimii.csv [--resumo] [--amostra 30] [--hash]

A raiz esperada contém as pastas descompactadas, uma por zip:
    mimii_6_dB_fan/fan/id_XX/{normal,abnormal}/NNNNNNNN.wav
    mimii_6_dB_pump/pump/id_XX/{normal,abnormal}/NNNNNNNN.wav
    due_dev_data_fan/fan/{train,source_test,target_test}/section_SS_....wav
    due_eval_data_fan_train/fan/train/section_SS_....wav
    due_eval_data_fan_test/fan/{source_test,target_test}/section_SS_<dom>_test_NNNN.wav
    due_ground_truth/**.csv   (rótulos do conjunto de avaliação do DCASE 2021 Task 2)

O CSV sai com `arquivo` relativo a data/mimii/ (o pai de "extraido").
Nenhum arquivo dos dados é executado; só lido com soundfile e csv.
Semente fixa da amostra: SEMENTE = 20261022.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.load_mimii import CANAL_ADOTADO, carregar  # noqa: E402

SEMENTE = 20261022
COLUNAS = ["base", "arquivo", "classe", "unidade", "condicao", "velocidade", "carga",
           "gravacao_id", "sr", "canais", "duracao_s", "f_alim_hz", "rot_esperada_hz", "obs"]

RE_MIMII = re.compile(r"^mimii_(-?\d+)_dB_(\w+)$")
RE_DUE_TREINO = re.compile(r"^section_(\d\d)_(source|target)_train_(normal|anomaly)_(\d+)_(.*)$")
RE_DUE_TESTE_DEV = re.compile(r"^section_(\d\d)_(source|target)_test_(normal|anomaly)_(\d+)$")
RE_DUE_TESTE_EVAL = re.compile(r"^section_(\d\d)_(source|target)_test_(\d+)$")


def ler_ground_truth(pasta: Path) -> dict[tuple[str, str], str]:
    """(tipo, nome do wav) -> 'normal' | 'abnormal', a partir dos CSV do DCASE 2021."""
    rot = {}
    for c in sorted(pasta.rglob("*.csv")):
        m = re.search(r"ground_truth_(\w+?)_section", c.name)
        tipo = m.group(1) if m else ""
        with open(c, newline="", encoding="utf-8") as f:
            for linha in csv.reader(f):
                if len(linha) < 2 or not linha[0].endswith(".wav"):
                    continue
                v = linha[1].strip()
                if v in ("0", "1"):
                    rot[(tipo, linha[0].strip())] = "abnormal" if v == "1" else "normal"
    return rot


def linha_mimii(p: Path, raiz_dados: Path, snr: str, info) -> dict:
    tipo, mid, classe = p.parts[-4], p.parts[-3], p.parts[-2]
    unidade = f"{tipo}_{mid}"
    return dict(base="mimii", arquivo=p.relative_to(raiz_dados).as_posix(), classe=classe,
                unidade=unidade, condicao=f"snr_{snr}dB", velocidade="", carga="",
                gravacao_id=f"mimii:{unidade}:{classe}:{p.stem}",
                obs="")


def linha_due(p: Path, raiz_dados: Path, gt: dict) -> dict | None:
    tipo, split_dir = p.parts[-3], p.parts[-2]
    nome = p.stem
    velocidade, obs, classe = "", "", None
    m = RE_DUE_TREINO.match(nome)
    if m:
        sec, dom, cl, idx, attr = m.groups()
        split = "train"
        classe = "normal" if cl == "normal" else "abnormal"
        # atributo de operação no nome do arquivo de treino; "strenght" é grafia original
        ma = re.match(r"(streng[th]{2}|strenght)_(\d+)_?(.*)$", attr)
        if ma:
            velocidade = f"strength_{ma.group(2)}"
            obs = f"atributo={attr}"
            if ma.group(1) != "strength":
                obs += ";grafia_original=strenght"
        else:
            obs = f"atributo={attr}"
    else:
        m = RE_DUE_TESTE_DEV.match(nome)
        if m:
            sec, dom, cl, idx = m.groups()
            split = "test"
            classe = "normal" if cl == "normal" else "abnormal"
        else:
            m = RE_DUE_TESTE_EVAL.match(nome)
            if not m:
                return None
            sec, dom, idx = m.groups()
            split = "test"
            classe = gt.get((tipo, p.name), "desconhecida")
            obs = "rotulo=ground_truth_dcase2021"
    unidade = f"{tipo}_sec_{sec}"
    return dict(base="mimii_due", arquivo=p.relative_to(raiz_dados).as_posix(), classe=classe,
                unidade=unidade, condicao=f"{dom}_{split}", velocidade=velocidade, carga="",
                gravacao_id=f"mimii_due:{tipo}:{nome}", obs=obs)


def inventario(raiz: Path) -> pd.DataFrame:
    raiz_dados = raiz.parent
    gt_dir = raiz / "due_ground_truth"
    gt = ler_ground_truth(gt_dir) if gt_dir.exists() else {}
    linhas, ignorados = [], []
    for pasta in sorted(d for d in raiz.iterdir() if d.is_dir()):
        mm = RE_MIMII.match(pasta.name)
        for p in sorted(pasta.rglob("*.wav")):
            info = sf.info(str(p))
            if mm:
                r = linha_mimii(p, raiz_dados, mm.group(1), info)
            elif pasta.name.startswith("due_"):
                r = linha_due(p, raiz_dados, gt)
            else:
                r = None
            if r is None:
                ignorados.append(str(p))
                continue
            r.update(sr=info.samplerate, canais=info.channels,
                     duracao_s=round(info.frames / info.samplerate, 6),
                     f_alim_hz="", rot_esperada_hz="")
            r["obs"] = ";".join(x for x in (r["obs"], f"subtype={info.subtype}") if x)
            linhas.append(r)
    if ignorados:
        print(f"[aviso] {len(ignorados)} wav sem padrão reconhecido, ex.: {ignorados[:3]}")
    df = pd.DataFrame(linhas, columns=COLUNAS)
    # MIMII DUE: o nome do arquivo de teste não traz o atributo. Quando o treino de um
    # (unidade, domínio) tem um único valor de strength, ele é atribuído ao teste desse
    # domínio e marcado como inferido (a página do DCASE 2021 diz que o domínio define a
    # condição; o rótulo por arquivo de teste não é publicado).
    d = df[df.base == "mimii_due"]
    if len(d):
        dom = d["condicao"].str.split("_").str[0]
        tr = d[d.condicao.str.endswith("train")]
        mapa = tr.groupby([tr["unidade"], dom[tr.index]])["velocidade"].unique()
        for i in d.index[d.condicao.str.endswith("test")]:
            v = mapa.get((df.at[i, "unidade"], dom[i]))
            if v is not None and len(v) == 1 and v[0]:
                df.at[i, "velocidade"] = v[0]
                df.at[i, "obs"] = ";".join(x for x in (df.at[i, "obs"], "velocidade_inferida_do_dominio") if x)
    return df


def resumo(df: pd.DataFrame) -> None:
    print("\n== Formato (sr, canais, subtype) por base ==")
    df = df.copy()
    df["subtype"] = df["obs"].str.extract(r"subtype=(\w+)")
    print(df.groupby(["base", "sr", "canais", "subtype"]).size().to_string())
    print("\n== Duração por arquivo (s) por base ==")
    print(df.groupby("base")["duracao_s"].describe().to_string())
    print("\n== Matriz classe x unidade (n arquivos | duração total em min) ==")
    t = df.pivot_table(index=["base", "unidade"], columns="classe", values="duracao_s",
                       aggfunc=["count", "sum"], fill_value=0)
    t = t.rename(columns={"count": "n", "sum": "min"}, level=0)
    t["min"] = t["min"] / 60.0
    print(t.round(1).to_string())
    print("\n== Totais por base e classe (n | horas) ==")
    g = df.groupby(["base", "classe"])["duracao_s"].agg(["count", "sum"])
    g["sum"] = (g["sum"] / 3600).round(3)
    print(g.to_string())
    d = df[df.base == "mimii_due"]
    if len(d):
        print("\n== MIMII DUE: unidade x condicao x classe ==")
        print(d.pivot_table(index="unidade", columns=["condicao", "classe"], values="arquivo",
                            aggfunc="count", fill_value=0).to_string())
        print("\n== MIMII DUE treino: unidade x condicao x atributo ==")
        tr = d[d.condicao.str.endswith("train")].copy()
        tr["attr"] = tr["obs"].str.extract(r"atributo=([^;]+)")
        print(tr.groupby(["unidade", "condicao", "attr"]).size().to_string())
    m = df[df.base == "mimii"]
    if len(m):
        print("\n== MIMII: índices dos arquivos (min, max, n, buracos) por unidade x classe ==")
        for (u, c), s in m.groupby(["unidade", "classe"]):
            idx = s["arquivo"].str.extract(r"/(\d+)\.wav$")[0].astype(int)
            print(f"{u:12s} {c:9s} min={idx.min()} max={idx.max()} n={len(idx)} "
                  f"buracos={idx.max() - idx.min() + 1 - len(idx)}")


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def amostra(df: pd.DataFrame, raiz_dados: Path, n: int) -> None:
    rng = np.random.default_rng(SEMENTE)
    print(f"\n== Amostra: {n} arquivos por base x unidade x classe x condicao, semente {SEMENTE},"
          f" canal {CANAL_ADOTADO} ==")
    print("base,unidade,condicao,classe,n,rms_dbfs_med,rms_dbfs_p10,rms_dbfs_p90,pico_dbfs_med,"
          "pico_dbfs_max,frac_clip,f_pico_30_1000hz,f_pico_1000_8000hz,corr_ch0_ch1,"
          "rms_ch_spread_db")
    df = df[df.classe.isin(["normal", "abnormal"])]
    guard = {}
    for (b, u, cond, c), s in df.groupby(["base", "unidade", "condicao", "classe"]):
        esc = s.iloc[np.sort(rng.choice(len(s), size=min(n, len(s)), replace=False))]
        rms, pk, clip, psd, corr, spread = [], [], [], None, [], []
        for a in esc["arquivo"]:
            p = raiz_dados / a
            y, sr = carregar(p)
            rms.append(_db(np.sqrt(np.mean(y.astype(np.float64) ** 2))))
            pk.append(_db(np.max(np.abs(y))))
            clip.append(np.mean(np.abs(y) >= 0.999))
            f, P = welch(y, fs=sr, nperseg=8192)
            psd = P if psd is None else psd + P
            if b == "mimii":
                Y, _ = sf.read(str(p), dtype="float32", always_2d=True)
                corr.append(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1])
                r = _db(np.sqrt(np.mean(Y.astype(np.float64) ** 2, axis=0)))
                spread.append(r.max() - r.min())
        guard[(b, u, cond.replace("_train", "_test"), c, cond)] = np.array(rms)
        b1 = (f >= 30) & (f <= 1000)
        b2 = (f > 1000) & (f <= 8000)
        fp1 = f[b1][np.argmax(psd[b1])]
        fp2 = f[b2][np.argmax(psd[b2])]
        print(f"{b},{u},{cond},{c},{len(esc)},{np.median(rms):.2f},{np.percentile(rms,10):.2f},"
              f"{np.percentile(rms,90):.2f},{np.median(pk):.2f},{np.max(pk):.2f},"
              f"{np.mean(clip):.2e},{fp1:.1f},{fp2:.1f},"
              f"{(np.median(corr) if corr else float('nan')):.3f},"
              f"{(np.median(spread) if spread else float('nan')):.2f}")
    # AUC de um detector trivial (só o RMS do arquivo) entre normal e anômalo da mesma
    # unidade e condição de teste: perto de 0,5 = o nível não denuncia a classe.
    from scipy.stats import mannwhitneyu
    print("\n== AUC do RMS (anômalo vs normal, mesma unidade e condição) ==")
    print("base,unidade,condicao,n_norm,n_anom,auc_rms")
    for (b, u, ct, c, cond), r_an in guard.items():
        if c != "abnormal":
            continue
        r_no = guard.get((b, u, ct, "normal", cond))
        if r_no is None:
            continue
        U = mannwhitneyu(r_an, r_no).statistic
        print(f"{b},{u},{cond},{len(r_no)},{len(r_an)},{U / (len(r_an) * len(r_no)):.3f}")


def duplicatas(df: pd.DataFrame, raiz_dados: Path) -> None:
    """Hash do áudio do canal adotado; procura arquivos idênticos dentro do subconjunto."""
    h = defaultdict(list)
    for a in df["arquivo"]:
        y, _ = carregar(raiz_dados / a)
        h[hashlib.sha1(y.tobytes()).hexdigest()].append(a)
    rep = [v for v in h.values() if len(v) > 1]
    print(f"\n== Duplicatas exatas (canal {CANAL_ADOTADO}): {len(rep)} grupos ==")
    for v in rep[:20]:
        print("  ", v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", type=Path, help="pasta 'extraido' com os zips descompactados")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--resumo", action="store_true")
    ap.add_argument("--amostra", type=int, default=0)
    ap.add_argument("--hash", action="store_true")
    a = ap.parse_args()
    raiz = a.raiz.resolve()
    df = inventario(raiz)
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.csv, index=False, encoding="utf-8")
    print(f"{len(df)} linhas -> {a.csv}")
    if a.resumo:
        resumo(df)
    if a.amostra:
        amostra(df, raiz.parent, a.amostra)
    if a.hash:
        duplicatas(df, raiz.parent)


if __name__ == "__main__":
    main()

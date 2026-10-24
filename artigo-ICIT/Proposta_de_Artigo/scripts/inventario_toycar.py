"""Inventário do ToyCar do DCASE 2022 Challenge Task 2 (subconjunto do ToyADMOS2).

Uso (da raiz do projeto):
    .venv/Scripts/python.exe -I scripts/inventario_toycar.py data/toycar \
        --csv out/indices/toycar.csv [--resumo] [--nivel] [--hash] [--rotacao]

Estrutura esperada em data/toycar/ (cada zip do Zenodo descompactado na sua pasta):
    extraido/dev_ToyCar/ToyCar/{train,test}/section_0[0-2]_<dom>_<split>_<cls>_NNNN_<attr>.wav
    extraido/eval_data_ToyCar_train/ToyCar/train/section_0[3-5]_<dom>_train_normal_NNNN_<attr>.wav
    extraido/eval_data_ToyCar_test/ToyCar/test/section_0[3-5]_NNNN.wav   (sem rótulo no nome)
    _rotulos_eval/gt_attributes_0[3-5].csv  (dcase2022_evaluator: nome cego -> nome com rótulos)
    _rotulos_eval/gt_data_0[3-5].csv        (nome cego -> 0 normal / 1 anomalia)
    _rotulos_eval/gt_domain_0[3-5].csv      (nome cego -> 0 source / 1 target)

`arquivo` no CSV é relativo a data/toycar/. Nenhum arquivo dos dados é executado; os wav são
só lidos com soundfile. Não há amostragem aleatória (o nível é medido em todos os arquivos de
teste), então não há semente.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.load_toycar import carregar  # noqa: E402

COLUNAS = ["base", "arquivo", "classe", "unidade", "condicao", "velocidade", "carga",
           "gravacao_id", "sr", "canais", "duracao_s", "f_alim_hz", "rot_esperada_hz", "obs"]

# Nome com rótulos: section_00_source_train_normal_0000_car_A1_spd_28V_mic_1_noise_1.wav
RE_NOME = re.compile(
    r"^section_(?P<sec>\d\d)_(?P<dom>source|target)_(?P<split>train|test)_"
    r"(?P<cls>normal|anomaly)_(?P<idx>\d{4})_(?P<attr>.+)\.wav$")
RE_CEGO = re.compile(r"^section_(?P<sec>\d\d)_(?P<idx>\d{4})\.wav$")

# Níveis de velocidade do ToyADMOS2 (ToyADMOS2_details.pdf, p. 5): tensão da fonte estabilizada
NIVEL_POR_TENSAO = {2.8: 1, 3.1: 2, 3.4: 3, 3.7: 4, 4.0: 5}


def atributos(s: str) -> dict[str, str]:
    """'car_A1_spd_28V_mic_1_noise_1' -> {'car': 'A1', 'spd': '28V', 'mic': '1', 'noise': '1'}."""
    p = s.split("_")
    if len(p) % 2:
        raise ValueError(f"atributos ímpares: {s}")
    return dict(zip(p[0::2], p[1::2]))


def ler_gt(pasta: Path) -> dict[tuple[str, str], dict]:
    """(seção, nome cego) -> {'nome_rotulado', 'gt_cls', 'gt_dom'} a partir do evaluator."""
    gt: dict[tuple[str, str], dict] = {}
    for f in sorted(pasta.glob("gt_attributes_*.csv")):
        sec = f.stem.split("_")[-1]
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cego = Path(r["file_name"]).name
                gt[(sec, cego)] = {"nome_rotulado": Path(r["file_name_with_attributes"]).name}
        for tipo in ("data", "domain"):
            with open(pasta / f"gt_{tipo}_{sec}.csv", encoding="utf-8") as fh:
                for nome, v in csv.reader(fh):
                    gt[(sec, nome)]["gt_" + tipo] = int(v)
    return gt


def linha(p: Path, raiz: Path, gt: dict) -> dict:
    nome = p.name
    obs = []
    m = RE_NOME.match(nome)
    if m is None:
        mc = RE_CEGO.match(nome)
        if mc is None:
            raise ValueError(f"nome inesperado: {p}")
        g = gt[(mc["sec"], nome)]
        # Classe e domínio: gt_data e gt_domain do dcase2022_evaluator (os mesmos do ranking
        # oficial). NÃO se usa gt_attributes: o nome rotulado de lá discorda de gt_data/gt_domain
        # em ~50 % dos arquivos e, pelo áudio (kNN), o car dele concorda só no nível do acaso
        # (ver ficha §9). Logo car/spd/mic dos 600 wav cegos ficam desconhecidos.
        m = {"sec": mc["sec"], "dom": "target" if g["gt_domain"] == 1 else "source",
             "split": "test", "cls": "anomaly" if g["gt_data"] == 1 else "normal", "attr": ""}
        obs.append("eval cego; classe/domínio de gt_data/gt_domain (dcase2022_evaluator); "
                   "atributos desconhecidos (gt_attributes desalinhado com o áudio)")
    a = atributos(m["attr"]) if m["attr"] else {}
    info = sf.info(str(p))
    spd = a.get("spd", "")
    volt = ""
    if re.fullmatch(r"\d\dV", spd):
        v = int(spd[:2]) / 10
        volt = f"{v:.1f}"
        nivel = NIVEL_POR_TENSAO.get(v)
        obs.append(f"spd={spd}: setpoint de tensão {v:.1f} V da fonte" +
                   (f" (nível {nivel} do ToyADMOS2)" if nivel else " (fora dos 5 níveis)"))
    conj = "dev" if int(m["sec"]) <= 2 else "eval"
    cond = (f"{conj}_sec{m['sec']}_{m['dom']}_{m['split']}_mic{a.get('mic', '')}"
            f"_noise{a.get('noise', '')}")
    if "car" in a:
        obs.append("unidade = atributo car (ID do carrinho; ver ficha §5)")
        unidade = "car_" + a["car"]
    else:
        unidade = f"desconhecida_sec{m['sec']}"
        obs.append("unidade desconhecida: um dos carros da seção (ficha §5)")
    return {
        "base": "toycar",
        "arquivo": p.relative_to(raiz).as_posix(),
        "classe": "anomalia" if m["cls"] == "anomaly" else "normal",
        "unidade": unidade,
        "condicao": cond,
        "velocidade": spd,
        "carga": volt,
        "gravacao_id": f"toycar/{conj}/{p.stem}",
        "sr": info.samplerate,
        "canais": info.channels,
        "duracao_s": round(info.frames / info.samplerate, 6),
        "f_alim_hz": "",
        "rot_esperada_hz": "",
        "obs": "; ".join(obs),
        # colunas auxiliares (não vão para o CSV)
        "_sec": m["sec"], "_dom": m["dom"], "_split": m["split"], "_mic": a.get("mic", ""),
        "_noise": a.get("noise", ""), "_attr_keys": "_".join(a.keys()), "_subtipo": info.subtype,
    }


def inventario(raiz: Path) -> pd.DataFrame:
    gt = ler_gt(raiz / "_rotulos_eval")
    wavs = sorted((raiz / "extraido").rglob("*.wav"))
    return pd.DataFrame([linha(p, raiz, gt) for p in wavs])


def resumo(df: pd.DataFrame) -> None:
    print(f"\n== {len(df)} wav; sr {sorted(df.sr.unique())}; canais {sorted(df.canais.unique())};"
          f" subtipo {sorted(df._subtipo.unique())}")
    print("duração (s):", df.duracao_s.describe().round(4).to_dict())
    print("chaves de atributo:", df._attr_keys.value_counts().to_dict())
    df = df.assign(conj=np.where(df._sec.astype(int) <= 2, "dev", "eval"))
    print("\n-- arquivos por conjunto x seção x domínio x split x classe")
    print(df.groupby(["conj", "_sec", "_dom", "_split", "classe"]).size().unstack(fill_value=0)
          .to_string())
    print("\n-- horas por classe:", (df.groupby("classe").duracao_s.sum() / 3600).round(3).to_dict())
    print("\n-- classe x unidade (todos)")
    print(pd.crosstab(df.unidade, df.classe, margins=True).to_string())
    print("\n-- unidade x seção x split/classe")
    print(pd.crosstab(df.unidade, [df._sec, df._split + "_" + df.classe]).to_string())
    print("\n-- unidade x domínio (por seção)")
    print(pd.crosstab([df._sec, df.unidade], df._dom).to_string())
    print("\n-- velocidade x seção x domínio")
    print(pd.crosstab([df._sec, df._dom], df.velocidade).to_string())
    print("\n-- mic x seção x domínio")
    print(pd.crosstab([df._sec, df._dom], df._mic).to_string())
    print("\n-- noise x seção x domínio")
    print(pd.crosstab([df._sec, df._dom], df._noise).to_string())
    t = df[df._split == "test"]
    print("\n-- TESTE: (seção, domínio, unidade) x classe")
    print(pd.crosstab([t._sec, t._dom, t.unidade], t.classe).to_string())
    print("\n-- TESTE: (seção, domínio, velocidade) x classe")
    print(pd.crosstab([t._sec, t._dom, t.velocidade], t.classe).to_string())
    print("\n-- TESTE: (seção, domínio, mic, noise) x classe")
    print(pd.crosstab([t._sec, t._dom, t._mic, t._noise], t.classe).to_string())


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def nivel(df: pd.DataFrame, raiz: Path) -> None:
    """RMS e pico (dBFS) de TODOS os arquivos de teste, por seção x domínio x unidade x classe."""
    t = df[df._split == "test"].copy()
    rms, pico = [], []
    for a in t.arquivo:
        y, _ = carregar(raiz / a)
        y = y - y.mean()
        rms.append(float(np.sqrt(np.mean(y ** 2))))
        pico.append(float(np.abs(y).max()))
    t["rms_db"] = _db(np.array(rms))
    t["pico_db"] = _db(np.array(pico))
    print("\n== NÍVEL (todos os arquivos de teste; dBFS, sem DC)")
    g = t.groupby(["_sec", "_dom", "unidade", "classe"]).agg(
        n=("rms_db", "size"), rms_med=("rms_db", "median"), rms_min=("rms_db", "min"),
        rms_max=("rms_db", "max"), pico_med=("pico_db", "median"))
    print(g.round(2).to_string())
    print("\n-- diferença mediana RMS anomalia - normal (dB) por seção x domínio x unidade")
    p = t.groupby(["_sec", "_dom", "unidade", "classe"]).rms_db.median().unstack()
    if {"anomalia", "normal"} <= set(p.columns):
        print((p["anomalia"] - p["normal"]).round(2).to_string())
    # AUC de um detector trivial: score = RMS (quanto maior, mais anômalo)
    from sklearn.metrics import roc_auc_score
    print("\n-- AUC do score trivial RMS (anomalia = mais alto), por seção x domínio")
    for (s, d), q in t.groupby(["_sec", "_dom"]):
        yv = (q.classe == "anomalia").astype(int)
        if yv.nunique() == 2:
            print(f"sec {s} {d:6s} n={len(q)} AUC_rms={roc_auc_score(yv, q.rms_db):.3f}"
                  f"  AUC_pico={roc_auc_score(yv, q.pico_db):.3f}")
    print("\n-- picos clipados: arquivos de teste com |y| >= 0.999:",
          int((np.array(pico) >= 0.999).sum()))


def duplicatas(df: pd.DataFrame, raiz: Path) -> None:
    """Hash do áudio decodificado de todos os wav; reporta grupos idênticos."""
    grupos = defaultdict(list)
    for a in df.arquivo:
        y, _ = carregar(raiz / a)
        grupos[hashlib.md5(y.tobytes()).hexdigest()].append(a)
    rep = [v for v in grupos.values() if len(v) > 1]
    print(f"\n== HASH: {len(df)} arquivos, {len(grupos)} áudios distintos, "
          f"{len(rep)} grupos repetidos")
    for v in rep[:10]:
        print("   ", v)


def _xcorr_max(F: np.ndarray, e: np.ndarray, G: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
    """Máximo da correlação cruzada normalizada (qualquer atraso) entre linhas de F e G."""
    out = np.empty((len(F), len(G)))
    for i in range(len(F)):
        c = np.fft.irfft(F[i][None, :] * np.conj(G), n=n, axis=1)
        out[i] = np.abs(c).max(axis=1) / (e[i] * g)
    return out


def reuso(df: pd.DataFrame, raiz: Path) -> None:
    """Procura a mesma gravação de origem (sinal limpo do carro) reaproveitada em seções
    diferentes do dev, com ruído diferente. Compara, por correlação cruzada normalizada (em
    qualquer atraso, 10 s, 16 kHz), até 40 arquivos de um grupo A com todos os do grupo B do
    mesmo carro, tensão e mic. Correlação de duas gravações independentes fica perto de 0;
    a mesma gravação com ruído somado dá valor alto (limitado pelo SNR)."""
    # (seção A, domínio A, seção B, domínio B, carro A, carro B, tensão)
    pares = [
        ("00", "source", "01", "source", "car_A1", "car_A1", "28V"),
        ("00", "source", "02", "source", "car_A1", "car_A1", "34V"),
        ("01", "source", "02", "source", "car_A1", "car_A1", "40V"),
        ("01", "source", "01", "source", "car_A1", "car_A1", "40V"),  # mesmo grupo
        ("00", "source", "00", "source", "car_E2", "car_E2", "28V"),  # mesmo grupo
        # controles com carros DIFERENTES: correlação alta aqui só pode vir do ruído
        ("01", "source", "01", "source", "car_A1", "car_A2", "28V"),
        ("00", "source", "00", "source", "car_A1", "car_E2", "28V"),
    ]
    n = 2 ** 19
    print("\n== REUSO DE GRAVAÇÃO ENTRE SEÇÕES (máx. correlação cruzada normalizada)")
    for sa, da, sb, db, u, ub, v in pares:
        qa = df[(df._sec == sa) & (df._dom == da) & (df.unidade == u) & (df.velocidade == v)]
        qb = df[(df._sec == sb) & (df._dom == db) & (df.unidade == ub) & (df.velocidade == v)]
        if qa.empty or qb.empty:
            print(f"   {sa}/{da} x {sb}/{db} {u}/{ub} {v}: grupo vazio")
            continue
        qa = qa.iloc[:40]

        def espectros(q):
            Y = np.stack([carregar(raiz / a)[0].astype(np.float64) for a in q.arquivo])
            Y -= Y.mean(axis=1, keepdims=True)
            return np.fft.rfft(Y, n=n, axis=1), np.linalg.norm(Y, axis=1)
        FA, eA = espectros(qa)
        FB, eB = espectros(qb)
        C = _xcorr_max(FA, eA, FB, eB, n)
        if sa == sb and da == db and u == ub:
            np.fill_diagonal(C[:, :len(qa)], np.nan)   # tira o próprio arquivo
        melhor = np.nanmax(C, axis=1)
        print(f"   {sa}/{da}({len(qa)}) x {sb}/{db}({len(qb)}) {u}/{ub} {v}: melhor par por arquivo "
              f"mediana {np.median(melhor):.3f}, mín {melhor.min():.3f}, máx {melhor.max():.3f};"
              f" fração >0,5: {np.mean(melhor > 0.5):.2f}; mediana geral {np.nanmedian(C):.3f}")


def rotacao(df: pd.DataFrame, raiz: Path) -> None:
    """Pico espectral dominante em 20-1000 Hz (Welch, 16 kHz, nperseg 16384 ~ 1 Hz) nos
    normais de treino source de cada seção dev, por unidade x tensão. Estimativa, não verdade."""
    from scipy.signal import welch
    t = df[(df._split == "train") & (df._dom == "source") & (df._sec.astype(int) <= 2)]
    out = []
    for (u, v), q in t.groupby(["unidade", "carga"]):
        picos = []
        for a in q.arquivo.iloc[:20]:
            y, sr = carregar(raiz / a)
            f, P = welch(y, sr, nperseg=16384)
            b = (f >= 20) & (f <= 1000)
            picos.append(f[b][np.argmax(P[b])])
        out.append((u, v, len(picos), np.median(picos), np.min(picos), np.max(picos)))
    print("\n== PICO DOMINANTE 20-1000 Hz (normais treino source dev; até 20 arquivos)")
    print(pd.DataFrame(out, columns=["unidade", "tensao_V", "n", "f_med", "f_min", "f_max"])
          .round(1).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", type=Path, help="data/toycar")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--resumo", action="store_true")
    ap.add_argument("--nivel", action="store_true")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--rotacao", action="store_true")
    ap.add_argument("--reuso", action="store_true")
    a = ap.parse_args()
    raiz = a.raiz.resolve()
    df = inventario(raiz)
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        df[COLUNAS].to_csv(a.csv, index=False, encoding="utf-8", lineterminator="\n")
        print(f"CSV: {a.csv} ({len(df)} linhas)")
    if a.resumo:
        resumo(df)
    if a.nivel:
        nivel(df, raiz)
    if a.hash:
        duplicatas(df, raiz)
    if a.rotacao:
        rotacao(df, raiz)
    if a.reuso:
        reuso(df, raiz)


if __name__ == "__main__":
    main()

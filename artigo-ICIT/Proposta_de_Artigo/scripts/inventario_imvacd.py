"""Inventário da base IM-VACD (Mendeley yc8yhg5xjd v2) baixada em data/imvacd/.

Uso (da raiz do projeto):
    .venv\\Scripts\\python.exe scripts\\inventario_imvacd.py

Gera out/indices/imvacd.csv (uma linha por CSV de áudio) e imprime em stdout as tabelas que
sustentam docs/bases/imvacd.md: formato medido, durações, matrizes classe x unidade e
classe x condição, simultaneidade entre celulares (carimbo de tempo), nível RMS/pico,
picos espectrais de 1x/2f e conferência de hashes com a API.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from lib.load_imvacd import carregar, carregar_meta, carregar_acelerometro  # noqa: E402

DADOS = RAIZ / "data" / "imvacd"
SAIDA = RAIZ / "out" / "indices" / "imvacd.csv"

# mesmos nomes de classe e unidade de out/indices/uoemd.csv (mesmos 8 motores)
CLASSES = {
    "H-H": "saudavel", "R-U": "desequilibrio_rotor", "R-M": "desalinhamento_rotor",
    "S-W": "falha_enrolamento_estator", "V-U": "desequilibrio_tensao_e_falta_fase",
    "B-R": "rotor_empenado", "K-A": "barras_rotor_quebradas", "F-B": "rolamento_defeituoso",
}
CELULAR = {"A": "iphone13", "B": "galaxyS6", "C": "galaxyA50"}
VEL_HZ = {"1": 15, "2": 20, "3": 25, "4": 30}
MONTAGEM = {"1_Hand_Held_Mounting_10_seconds": "mao", "2_Rigid_Mounting_10_seconds": "rigida"}
CARGA_PASTA = {"1_Unloaded": "0", "2_Loaded": "1"}
# Na montagem na mão com carga, os arquivos com código de velocidade 2 (20 Hz pela descrição)
# têm 1x em ~29,6 Hz no acelerômetro e o mesmo padrão de picos dos arquivos de 30 Hz; os de
# 20 Hz da montagem rígida têm 1x em ~19,5 Hz. Tratamos como 30 Hz (conferido abaixo por assert).
CORRECAO_VEL = {("mao", "1", "2"): 30}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pico(espec, freqs, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    i = np.argmax(espec[m])
    return float(freqs[m][i]), float(espec[m][i])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    api = json.load(open(DADOS / "_metadados" / "mendeley_files_v2.json", encoding="utf-8"))
    hashes_api = {f["_path"]: f["content_details"]["sha256_hash"] for f in api}
    arquivos = sorted(p for p in DADOS.rglob("*.csv") if not p.relative_to(DADOS).parts[0].startswith("_"))
    print(f"arquivos CSV em data/imvacd: {len(arquivos)} (API: {len(api)})")

    linhas, extra = [], []
    ok_hash = 0
    hashes_audio = {}
    for p in arquivos:
        rel = p.relative_to(DADOS).as_posix()
        ok_hash += sha256(p) == hashes_api.get(rel)
        partes = rel.split("/")
        mont, pasta_carga, pasta_cel, nome = MONTAGEM[partes[0]], partes[1], partes[2], partes[3]
        cel_l, c1, c2, v, carga = nome[:-4].split("-")
        assert CARGA_PASTA[pasta_carga] == carga, rel
        assert pasta_cel[0] == str("ABC".index(cel_l) + 1), rel
        meta = carregar_meta(p)
        y, sr = carregar(p)
        hashes_audio[rel] = hashlib.sha1(y.tobytes()).hexdigest()
        classe = CLASSES[f"{c1}-{c2}"]
        f_nome = VEL_HZ[v]
        f = CORRECAO_VEL.get((mont, carga, v), f_nome)
        cel = CELULAR[cel_l]
        dur = len(y) / sr
        # nível e saturação
        yac = y.astype(np.float64) - y.mean()
        rms, pk = float(np.sqrt(np.mean(yac**2))), float(np.max(np.abs(y)))
        n_sat = int(np.sum(np.abs(y) >= 32767 / 32768))
        q = np.unique(np.round(y * 32768).astype(np.int64))
        resid16 = float(np.max(np.abs(y * 32768 - np.round(y * 32768))))
        # espectro do áudio: picos em torno de f e 2f (resolução ~0,1 Hz)
        n = len(yac)
        esp = np.abs(np.fft.rfft(yac * np.hanning(n)))
        fr = np.fft.rfftfreq(n, 1 / sr)
        f2, _ = pico(esp, fr, 1.9 * f, 2.1 * f)
        # versão interpolada (zero-padding x4) para comparar celulares no mesmo evento
        esp4 = np.abs(np.fft.rfft(yac * np.hanning(n), 4 * n))
        f2_fino, _ = pico(esp4, np.fft.rfftfreq(4 * n, 1 / sr), 1.9 * f, 2.1 * f)
        del esp4
        # acelerômetro do celular: pico 1x
        try:
            a, sra = carregar_acelerometro(p)
            sra_real = meta["accelerometer_sampling_frequency_real"]
            aa = a - a.mean(0)
            ea = np.sqrt(sum(np.abs(np.fft.rfft(aa[:, k] * np.hanning(len(aa))))**2 for k in range(3)))
            fa = np.fft.rfftfreq(len(aa), 1 / sra_real)
            f1a = pico(ea, fa, 0.85 * f, 1.05 * f)[0] if fa.max() > 1.05 * f else np.nan
            f_larga = pico(ea, fa, 12, 40)[0]   # pico mais forte 12-40 Hz, p/ conferir o código
        except Exception:
            f1a, f_larga, a = np.nan, np.nan, np.zeros((0, 3))
        ts = datetime.fromisoformat(str(meta["timestamp"]))
        cond = f"{mont}_{cel}_const_{f}Hz_{'com' if carga == '1' else 'sem'}_carga"
        linhas.append(dict(
            base="imvacd", arquivo=rel, classe=classe, unidade=f"motor_{c1}_{c2}",
            condicao=cond, velocidade=f, carga="com_carga" if carga == "1" else "sem_carga",
            gravacao_id="", sr=sr, canais=1, duracao_s=round(dur, 4), f_alim_hz=f,
            rot_esperada_hz="",
            obs=(("ROTULO CORRIGIDO: nome diz codigo 2 (20 Hz), espectro indica 30 Hz; "
                  if f != f_nome else "")
                 + f"celular={cel}; montagem={mont}; f_alim=setpoint nominal do nome; "
                 f"sr_real_app={meta['acoustic_sampling_frequency_real']:.1f}; "
                 f"timestamp={meta['timestamp']}; unidade inferida (1 motor por classe)")))
        extra.append(dict(arquivo=rel, classe=classe, cel=cel, mont=mont, f=f, f_nome=f_nome,
                          codigo=v, carga=carga, f_larga=f_larga,
                          sr_set=sr, sr_real=meta["acoustic_sampling_frequency_real"],
                          dur_app=meta["acoustic_duration (s)"], n=len(y),
                          units=meta["acoustic_units"], device=meta["device"], ts=ts,
                          acc_set=meta["accelerometer_sampling_frequency_set"],
                          acc_real=meta["accelerometer_sampling_frequency_real"], n_acc=len(a),
                          rms=rms, pk=pk, n_sat=n_sat, niveis=len(q), resid16=resid16,
                          f2=f2, f2_fino=f2_fino, f1a=f1a))

    print(f"sha256 conferidos com a API: {ok_hash}/{len(arquivos)}")
    df, ex = pd.DataFrame(linhas), pd.DataFrame(extra)

    # --- gravação de origem: mesma montagem, classe, velocidade, carga e carimbo próximo
    ex["evento"] = ex.mont + "/" + ex.classe + "/" + ex.f.astype(str) + "/" + ex.carga
    print("\n== §8 simultaneidade: carimbos de tempo por evento (montagem/classe/vel/carga)")
    # Um arquivo só é a "mesma gravação" de outro se os intervalos [ts, ts + duração] se
    # sobrepõem (o carimbo pode ser início ou fim; testamos as duas leituras).
    difs = []
    sobrep = 0
    for ev, g in ex.groupby("evento"):
        g = g.sort_values("ts")
        t = g.ts.tolist()
        d = g.dur_app.tolist()
        gaps = [(t[i + 1] - t[i]).total_seconds() for i in range(len(t) - 1)]
        ov = any(gp < max(d[i], d[i + 1]) for i, gp in enumerate(gaps))
        sobrep += ov
        difs.append((ev, len(g), "".join({"iphone13": "i", "galaxyS6": "s", "galaxyA50": "a"}[c] for c in g.cel), min(gaps) if gaps else np.nan,
                     max(gaps) if gaps else np.nan, ov))
    dd = pd.DataFrame(difs, columns=["evento", "n", "ordem", "gap_min_s", "gap_max_s", "sobrepoe"])
    print("eventos (montagem/classe/vel/carga) com n celulares:", dict(Counter(dd.n)))
    print("ordem de gravação dos celulares (i=iPhone, s=S6, a=A50, por carimbo):",
          dict(Counter(dd.ordem)))
    print("intervalo entre celulares consecutivos no mesmo evento, s:")
    print(dd[["gap_min_s", "gap_max_s"]].describe().round(1).to_string())
    print("eventos com algum par de arquivos sobreposto no tempo:", sobrep, "de", len(dd))
    # sequência completa: intervalo entre arquivos consecutivos na sessão (montagem, carga)
    for (mo, ca), g in ex.groupby(["mont", "carga"]):
        g = g.sort_values("ts")
        gp = g.ts.diff().dt.total_seconds().dropna()
        print(f"  sessão {mo}/carga{ca}: {len(g)} arquivos, {g.ts.min()} → {g.ts.max()}, "
              f"gap mediano {gp.median():.1f} s, mínimo {gp.min():.1f} s")
    # Mesmo evento = mesmo motor, montagem, velocidade e carga, gravado pelos celulares em
    # sequência (mão) ou ao mesmo tempo (rígida) sem desligar o motor. Para split, os 2-3
    # arquivos do evento são UMA gravação de origem (conservador): gravacao_id por evento.
    gid = {r.arquivo: "imvacd/" + r.evento.replace("/", "_") for r in ex.itertuples()}
    df["gravacao_id"] = df.arquivo.map(gid)
    print("\nordem dos celulares e intervalo entre consecutivos, por montagem:")
    for mo, g in dd.assign(mont=dd.evento.str.split("/").str[0]).groupby("mont"):
        print(f"  {mo}: ordens {dict(Counter(g.ordem))}; gap min/mediana/max "
              f"{g.gap_min_s.min():.1f}/{pd.concat([g.gap_min_s, g.gap_max_s]).median():.1f}/"
              f"{g.gap_max_s.max():.1f} s; eventos com sobreposição {int(g.sobrepoe.sum())}/{len(g)}")
    print("datas por montagem x carga x celular x velocidade:")
    print(ex.groupby(["mont", "carga", "cel", "f"]).ts.agg(
        lambda s: f"{s.min():%Y-%m-%d %H:%M} a {s.max():%Y-%m-%d %H:%M}").to_string())

    # mesmo evento entre celulares: o pico 2x tem a mesma frequência?  Se a taxa verdadeira
    # fosse a "real" do app, as frequências medidas com a nominal difeririam pela razão das taxas.
    print("\n== §3/§8 pico 2x (zero-padding x4) entre celulares do mesmo evento")
    pares = []
    for ev, g in ex.groupby("evento"):
        g = g.set_index("cel")
        for a_, b_ in [("galaxyS6", "galaxyA50"), ("iphone13", "galaxyS6"), ("iphone13", "galaxyA50")]:
            if a_ in g.index and b_ in g.index:
                ra = g.loc[a_, "f2_fino"] / g.loc[b_, "f2_fino"]
                rr_ = (g.loc[a_, "sr_real"] / g.loc[a_, "sr_set"]) / (g.loc[b_, "sr_real"] / g.loc[b_, "sr_set"])
                pares.append(dict(mont=ev.split("/")[0], par=f"{a_}/{b_}", razao_nominal=ra,
                                  razao_se_taxa_real=ra * rr_, bate_1e3=abs(ra - 1) < 1e-3))
    pp = pd.DataFrame(pares)
    print(pp.groupby(["mont", "par"]).agg(n=("razao_nominal", "size"),
                                         med_nominal=("razao_nominal", "median"),
                                         med_taxa_real=("razao_se_taxa_real", "median"),
                                         frac_bate_0p1pct=("bate_1e3", "mean")).round(5).to_string())
    print("mesmo motor em eventos diferentes (sessão/velocidade diferente): 2x/2f varia "
          f"{ex.groupby(['classe', 'f']).f2_fino.agg(lambda s: s.max() / s.min() - 1).median()*100:.2f} % (mediana max/min-1)")

    # conferência da correção de velocidade
    print("\n== conferência do código de velocidade: pico mais forte 12-40 Hz no acelerômetro do celular")
    cc_ = ex.groupby(["mont", "carga", "codigo", "f_nome", "f"]).f_larga.median().round(2)
    print(cc_.to_string())
    m_corr = ex[(ex.mont == "mao") & (ex.carga == "1") & (ex.codigo == "2")].f_larga.median()
    assert 29 <= m_corr <= 31, m_corr
    print(f"mao/carga1/código 2: mediana {m_corr:.2f} Hz -> tratado como 30 Hz")

    # hashes de áudio repetidos?
    rep = [k for k, c in Counter(hashes_audio.values()).items() if c > 1]
    print("\náudios idênticos (sha1 do vetor):", len(rep))

    # --- formato
    print("\n== §3 formato por celular")
    for cel, g in ex.groupby("cel"):
        print(cel, "device:", dict(Counter(g.device)), "units:", dict(Counter(g.units)))
        print("   sr_set:", dict(Counter(g.sr_set)), " sr_real min/med/max: "
              f"{g.sr_real.min():.1f}/{g.sr_real.median():.1f}/{g.sr_real.max():.1f}",
              f" n amostras min/max {g.n.min()}/{g.n.max()}  dur_app {g.dur_app.min()}-{g.dur_app.max()}",
              f" residuo x32768 max {g.resid16.max():.2e}")
        print(f"   acel set {dict(Counter(g.acc_set))} real {g.acc_real.min():.1f}-{g.acc_real.max():.1f}"
              f" n_acc {g.n_acc.min()}-{g.n_acc.max()}")
        print(f"   niveis 16 bit usados min/med/max {g.niveis.min()}/{int(g.niveis.median())}/{g.niveis.max()};"
              f" amostras saturadas: total {g.n_sat.sum()} em {(g.n_sat > 0).sum()} arquivos")
    print("datas dos carimbos:", dict(Counter(ex.ts.dt.date.astype(str))))
    print("datas por montagem x carga:")
    print(ex.groupby(["mont", "carga"]).ts.agg(["min", "max"]).to_string())

    # --- durações
    print("\n== §4 duração (nominal sr_set) por classe, s")
    ex["dur"] = ex.n / ex.sr_set
    print(f"por arquivo: min {ex.dur.min():.4f} max {ex.dur.max():.4f}")
    print(ex.groupby("classe").dur.agg(["count", "sum"]).round(1).to_string())
    print(f"total {ex.dur.sum():.1f} s = {ex.dur.sum()/60:.1f} min")

    # --- matrizes
    print("\n== §5 classe x unidade")
    print(pd.crosstab(df.classe, df.unidade).to_string())
    print("\n== §6 classe x (montagem, celular, velocidade, carga): número de arquivos")
    ex["cc"] = ex.mont + "|" + ex.cel + "|" + ex.f.astype(str) + "|" + ex.carga
    m = pd.crosstab(ex.cc, ex.classe)
    print(m.to_string())

    # --- nível
    print("\n== nível acústico: RMS sem DC (x1e3, unidades pcm normalizadas), mediana por classe x celular")
    print((ex.pivot_table(index="classe", columns=["mont", "cel"], values="rms", aggfunc="median") * 1e3).round(2).to_string())
    print("\npico |y| máximo por classe x celular")
    print(ex.pivot_table(index="classe", columns="cel", values="pk", aggfunc="max").round(3).to_string())
    print("\nRMS saudável vs. falha mais silenciosa, por (montagem, celular, vel, carga): razão min/max")
    r = []
    for cc, g in ex.groupby("cc"):
        s = g[g.classe == "saudavel"].rms
        if len(s):
            r.append((cc, g[g.classe != "saudavel"].rms.min() / s.iloc[0],
                      g.rms.idxmin() == s.index[0]))
    rr = pd.DataFrame(r, columns=["cc", "razao", "saudavel_mais_baixo"])
    print(f"razão falha_mais_silenciosa/saudável: min {rr.razao.min():.2f} mediana {rr.razao.median():.2f} "
          f"max {rr.razao.max():.2f}; saudável o mais silencioso em {rr.saudavel_mais_baixo.sum()}/{len(rr)}")

    # --- rotação
    print("\n== §7 picos: 2f no áudio e 1x no acelerômetro do celular, como razão de f nominal")
    ex["r2"] = ex.f2 / (2 * ex.f)
    ex["r1a"] = ex.f1a / ex.f
    print(ex.groupby(["cel", "f"]).r2.describe()[["count", "min", "50%", "max"]].round(4).to_string())
    print(ex.groupby(["cel", "f"]).r1a.describe()[["count", "min", "50%", "max"]].round(4).to_string())
    print("\nmontagem mao, carga 1, código de velocidade 2 (20 Hz): r1a e r2 por celular")
    print(ex[(ex.mont == "mao") & (ex.carga == "1")].groupby(["cel", "f"])[["r1a", "r2"]].median().round(4).to_string())

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    cols = ["base", "arquivo", "classe", "unidade", "condicao", "velocidade", "carga", "gravacao_id",
            "sr", "canais", "duracao_s", "f_alim_hz", "rot_esperada_hz", "obs"]
    df[cols].to_csv(SAIDA, index=False)
    print(f"\nescrito {SAIDA.relative_to(RAIZ)}: {len(df)} linhas, {df.gravacao_id.nunique()} gravacao_id")
    ex.drop(columns=["ts"]).assign(ts=ex.ts.astype(str)).to_csv(
        DADOS / "_metadados" / "inventario_extra.csv", index=False)


if __name__ == "__main__":
    main()

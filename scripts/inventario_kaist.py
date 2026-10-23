"""Inventário da base KAIST (Jung et al. 2023, Mendeley ztmf3m7h5x v6), canal acústico.

Uso (da raiz do projeto):
    .venv/Scripts/python.exe -I scripts/inventario_kaist.py data/kaist out/indices/kaist.csv [--vib-remoto]

Lê SÓ arquivos baixados em <dados>/ (acoustic/*.mat, current/*.tdms, raw/*.zip, _meta/*.json);
nada de dentro do download é executado. Escreve o CSV-índice (uma linha por arquivo acústico)
e imprime em stdout todas as medidas usadas na ficha docs/bases/kaist.md.
--vib-remoto: lê, por HTTP Range, o diretório central do vibration.zip (2,66 GB, não baixado)
e o cabeçalho MAT de cada arquivo 0Nm_*, para datar as gravações de vibração.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.load_kaist import carregar, carregar_corrente, metadados  # noqa: E402

COLUNAS = ["base", "arquivo", "classe", "unidade", "condicao", "velocidade", "carga",
           "gravacao_id", "sr", "canais", "duracao_s", "f_alim_hz", "rot_esperada_hz", "obs"]
CLASSE = {"Normal": "normal", "BPFI": "rolamento_interno", "BPFO": "rolamento_externo"}
# Geometria declarada no artigo (NSK 6205 DDU): d=7,90 mm, D=38,5 mm, N=9, theta=0
D_BOLA, D_PRIM, N_BOLAS = 7.90, 38.5, 9
R_BPFO = N_BOLAS / 2 * (1 - D_BOLA / D_PRIM)
R_BPFI = N_BOLAS / 2 * (1 + D_BOLA / D_PRIM)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_nome(stem: str):
    m = re.fullmatch(r"(\d+)Nm_(Normal|BPFI|BPFO)(?:_(\d\d))?", stem)
    if not m:
        raise ValueError(stem)
    carga, tipo, sev = m.group(1), m.group(2), m.group(3)
    sev_mm = None if sev is None else int(sev) / 10.0
    return int(carga), tipo, sev_mm


def pico_banda(f, P, lo, hi):
    m = (f >= lo) & (f <= hi)
    i = np.argmax(P[m])
    return float(f[m][i]), float(10 * np.log10(P[m][i] / np.median(P[m])))


def analisa_audio(y, sr):
    y64 = y.astype(np.float64)
    rms = float(np.sqrt(np.mean(y64 ** 2)))
    pk = float(np.max(np.abs(y64)))
    u = np.unique(y64)
    passo = float(np.min(np.diff(u)))
    n_no_pico = int(np.sum(np.abs(y64) >= 0.999 * pk))
    w = y64[: len(y64) // sr * sr].reshape(-1, sr)
    rms_s = np.sqrt((w ** 2).mean(1))
    # envelope 2-10 kHz -> frequência de eixo (45-55 Hz) e linhas de defeito
    b = ss.sosfiltfilt(ss.butter(4, [2000, 10000], "bandpass", fs=sr, output="sos"), y64)
    e = np.abs(ss.hilbert(b))
    e -= e.mean()
    f, P = ss.welch(e, sr, nperseg=sr * 20)
    f_eixo, snr_eixo = pico_banda(f, P, 45, 55)
    f_motor, snr_motor = pico_banda(f, P, 22, 26)
    # espectro direto: linhas BPFO (170-195 Hz) e BPFI (255-285 Hz)
    f2, P2 = ss.welch(y64, sr, nperseg=sr * 20)
    f_bpfo, snr_bpfo = pico_banda(f2, P2, 170, 195)
    f_bpfi, snr_bpfi = pico_banda(f2, P2, 255, 285)
    f_mesh, snr_mesh = pico_banda(f2, P2, 1440, 1470)
    fb, Pb = ss.welch(y64, sr, nperseg=8192)
    df = fb[1] - fb[0]
    bandas = [20, 250, 1000, 2000, 4000, 8000, 16000, 25600]
    spl = {f"{a}-{c}": float(10 * np.log10(Pb[(fb >= a) & (fb < c)].sum() * df / 4e-10))
           for a, c in zip(bandas[:-1], bandas[1:])}
    return dict(rms=rms, pico=pk, crista=pk / rms, media=float(y64.mean()), n_valores=len(u),
                passo=passo, bits_usados=float(np.log2(2 * pk / passo)), n_no_pico=n_no_pico,
                rms_s_min=float(rms_s.min()), rms_s_max=float(rms_s.max()),
                spl_db=float(20 * np.log10(rms / 2e-5)), f_eixo=f_eixo, snr_eixo=snr_eixo,
                f_motor=f_motor, snr_motor=snr_motor, f_bpfo=f_bpfo, snr_bpfo=snr_bpfo,
                f_bpfi=f_bpfi, snr_bpfi=snr_bpfi, f_mesh=f_mesh, snr_mesh=snr_mesh, bandas=spl)


def analisa_tdms(p: Path):
    from nptdms import TdmsFile

    md = TdmsFile.read_metadata(str(p))
    fases, sr = carregar_corrente(p)
    x = next(iter(fases.values()))
    f, P = ss.welch(x.astype(np.float64), sr, nperseg=int(sr * 20))
    f_alim, _ = pico_banda(f, P, 20, 80)
    return dict(nome_interno=md.properties.get("name"), criado=str(md.properties.get("datetime")),
                run=md.properties.get("RunNumber"), sr=sr, dur=len(x) / sr, fases=len(fases),
                f_alim=f_alim)


def vib_remoto(url_mendeley: str, tamanho: int):
    import http.client
    import urllib.parse

    # o Mendeley responde 302 para uma URL S3 assinada (que só aceita GET)
    loc = url_mendeley
    for _ in range(5):
        u = urllib.parse.urlsplit(loc)
        c = http.client.HTTPSConnection(u.netloc, timeout=60)
        c.request("GET", u.path + ("?" + u.query if u.query else ""), headers={"Range": "bytes=0-0"})
        r = c.getresponse()
        r.read()
        if r.status in (301, 302, 303, 307, 308):
            loc = urllib.parse.urljoin(loc, r.getheader("Location"))
            continue
        break

    def rng(a, b):
        r = urllib.request.Request(loc, headers={"Range": f"bytes={a}-{b}"})
        return urllib.request.urlopen(r).read()

    tail = rng(tamanho - 65536, tamanho - 1)
    eocd = tail[tail.rfind(b"PK\x05\x06"):]
    cdsize, cdoff = struct.unpack("<II", eocd[12:20])
    cd = rng(cdoff, cdoff + cdsize - 1)
    p, out = 0, []
    while p < len(cd):
        (_, _, _, _, meth, mt, mdd, _, _, us, fnl, exl, cml, _, _, _, lho) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        fn = cd[p + 46:p + 46 + fnl].decode("utf8", "replace")
        data_zip = (f"{(mdd >> 9) + 1980}-{(mdd >> 5) & 15:02d}-{mdd & 31:02d} "
                    f"{mt >> 11:02d}:{(mt >> 5) & 63:02d}")
        hdr = ""
        if fn.startswith("0Nm"):
            lh = rng(lho, lho + 29)
            a, b = struct.unpack("<HH", lh[26:30])
            raw = rng(lho + 30 + a + b, lho + 30 + a + b + 4095)
            d = zlib.decompressobj(-15).decompress(raw) if meth == 8 else raw
            hdr = d[:116].decode("latin1").split("Created on:")[-1].strip()
        out.append((fn, us, data_zip, hdr))
        p += 46 + fnl + exl + cml
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dados", type=Path)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--vib-remoto", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    R = a.dados.resolve()

    print("## Downloads (sha256 local vs. Mendeley)")
    meta = {x["filename"]: x for x in json.loads((R / "_meta" / "root_files.json").read_text("utf8"))}
    for nome, x in meta.items():
        print(f"  Mendeley: {nome}  {x['size']} B  sha256={x['content_details']['sha256_hash']}")
    for z in sorted((R / "raw").glob("*.zip")):
        print(f"  local: raw/{z.name}  {z.stat().st_size} B  sha256={sha256(z)}")
    with zipfile.ZipFile(R / "raw" / "acoustic.zip") as zf:
        print("  acoustic.zip contém:", ", ".join(f"{i.filename} ({i.file_size} B, zip {i.date_time})"
                                               for i in zf.infolist()))
    with zipfile.ZipFile(R / "raw" / "current_temp.zip") as zf:
        L = zf.infolist()
        print(f"  current_temp.zip: {len(L)} arquivos .tdms; datas no zip:",
              sorted({f'{i.date_time[0]}-{i.date_time[1]:02d}-{i.date_time[2]:02d}' for i in L}))
        for i in L:
            print(f"    {i.filename} {i.file_size} B zip={i.date_time[0]}-{i.date_time[1]:02d}-"
                  f"{i.date_time[2]:02d} {i.date_time[3]:02d}:{i.date_time[4]:02d}")

    print("\n## Corrente/temperatura (TDMS extraídos, mesmo nome dos acústicos)")
    tdms = {}
    for p in sorted((R / "current").glob("*.tdms")):
        t = analisa_tdms(p)
        tdms[p.stem] = t
        print(f"  {p.name}: nome_interno={t['nome_interno']} criado={t['criado']} run={t['run']} "
              f"sr={t['sr']:.2f} dur={t['dur']:.1f}s fases_gravadas={t['fases']} "
              f"f_alim={t['f_alim']:.2f}Hz  sha256={sha256(p)}")

    print("\n## Canal acústico")
    linhas, info = [], []
    for p in sorted((R / "acoustic").glob("*.mat")):
        carga, tipo, sev = parse_nome(p.stem)
        y, sr = carregar(p)
        md = metadados(p)
        an = analisa_audio(y, sr)
        t0 = dt.datetime.strptime(md["absolute_time"].split(" ms ")[0], "%Y-%m-%d %H:%M:%S")
        info.append((t0, p.stem, md, an, len(y) / sr, tipo, sev))
        print(f"  {p.name}: sha256={sha256(p)}")
        print(f"    run_name={md['run_name']} absolute_time={md['absolute_time']} creation_time="
              f"{md['creation_time']} canal={md['canal_id']}:{md['canal_label']} grupo="
              f"{md['channel_group']} unidade={md['unidade_y']} t0={md['t0_s']:.3e}s")
        print(f"    sr={sr} n={len(y)} dur={len(y)/sr:.3f}s rms={an['rms']:.4f}Pa "
              f"({an['spl_db']:.1f} dB SPL) pico={an['pico']:.4f}Pa crista={an['crista']:.2f} "
              f"media={an['media']:.2e} n_valores={an['n_valores']} passo={an['passo']:.4e}Pa "
              f"bits_usados~{an['bits_usados']:.1f} amostras>=99.9%pico={an['n_no_pico']} "
              f"rms_1s=[{an['rms_s_min']:.4f},{an['rms_s_max']:.4f}]")
        print(f"    envelope: f_eixo={an['f_eixo']:.2f}Hz(+{an['snr_eixo']:.0f}dB) "
              f"f_motor={an['f_motor']:.2f}Hz(+{an['snr_motor']:.0f}dB) "
              f"razao={an['f_eixo']/an['f_motor']:.4f} | espectro: pico170-195={an['f_bpfo']:.2f}Hz"
              f"(+{an['snr_bpfo']:.0f}dB, BPFO teor.={R_BPFO*an['f_eixo']:.2f}) "
              f"pico255-285={an['f_bpfi']:.2f}Hz(+{an['snr_bpfi']:.0f}dB, BPFI teor.="
              f"{R_BPFI*an['f_eixo']:.2f}) pico1440-1470={an['f_mesh']:.2f}Hz(+{an['snr_mesh']:.0f}dB)")
        print("    dB SPL por banda:", " ".join(f"{k}:{v:.0f}" for k, v in an["bandas"].items()))

        sev_txt = "" if sev is None else f"; defeito {sev:.1f} mm"
        unidade = "bancada_KAIST/rolamento_A_" + (
            "normal" if sev is None else f"{tipo}_{sev:.1f}mm")
        t = tdms.get(p.stem)
        obs = (f"run {md['run_name']} inicio {md['absolute_time'].split(' ms ')[0]} (Test.Lab); "
               f"f_eixo estimada no envelope acustico {an['f_eixo']:.2f} Hz (nao e verdade direta); "
               "sem tacometro")
        if t:
            obs += (f"; corrente {p.stem}.tdms e de outra sessao ({t['criado'][:10]}), "
                    f"f_alim ali {t['f_alim']:.2f} Hz")
        linhas.append(dict(base="kaist", arquivo=f"acoustic/{p.name}", classe=CLASSE[tipo],
                           unidade=unidade, condicao=f"{carga}Nm{sev_txt}", velocidade="3010 rpm (nominal)",
                           carga=f"{carga} Nm", gravacao_id=md["run_name"], sr=sr, canais=1,
                           duracao_s=f"{len(y)/sr:.3f}", f_alim_hz="", rot_esperada_hz="", obs=obs))

    print("\n## Ordem de gravação do áudio (absolute_time do Test.Lab)")
    info.sort()
    for k, (t0, stem, md, an, dur, tipo, sev) in enumerate(info):
        gap = "" if k == 0 else f"  intervalo desde o fim do anterior: {(t0 - info[k-1][0]).total_seconds() - info[k-1][4]:.0f} s"
        print(f"  {k+1}. {t0}  {stem:12s} ({md['run_name']}){gap}")

    print("\n## Totais por classe (áudio)")
    tot = {}
    for L in linhas:
        tot.setdefault(L["classe"], [0, 0.0])
        tot[L["classe"]][0] += 1
        tot[L["classe"]][1] += float(L["duracao_s"])
    for c, (n, d) in tot.items():
        print(f"  {c}: {n} arquivos, {d:.1f} s")
    print(f"  TOTAL: {len(linhas)} arquivos, {sum(v[1] for v in tot.values()):.1f} s")

    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        w.writerows(linhas)
    print(f"\nCSV: {a.csv} ({len(linhas)} linhas)")

    if a.vib_remoto:
        print("\n## vibration.zip (remoto, HTTP Range; não baixado)")
        v = meta["vibration.zip"]
        for fn, us, dz, hdr in vib_remoto(v["content_details"]["download_url"], v["size"]):
            print(f"  {fn} {us} B zip={dz}" + (f" | MAT criado: {hdr}" if hdr else ""))


if __name__ == "__main__":
    main()

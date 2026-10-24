"""Host da bancada no ESP32 físico (processor-in-the-loop pela serial; categoria 4).

    py = .venv\\Scripts\\python.exe embarcados\\src\\python\\pil_esp32.py
    py portas                                   lista as seriais
    py info  --porta COM5 --alvo esp32s3        grava results/manifests/bancada.json
    py equiv --porta COM5 --alvo esp32s3        manda os 164 casos do ouro; grava c_out/esp32_<alvo>
    py lat   --porta COM5 --alvo esp32s3 --n 200 --aquecimento 5
    py seq-imagem --por-classe 8                monta a imagem da partição "seq" (IDMT)
    py seq   --porta COM5 --alvo esp32s3 --imagem <arquivo> --cp 2
        (antes: gravar a imagem com firmware/gravar_seq.bat COM5 <arquivo>)

Protocolo: firmware/main/main.c. Só linhas que começam com "@@" são do protocolo.
Nada aqui pode ser chamado de "medido no ESP32" sem `info` ter gravado bancada.json na
mesma sessão de medição (o id da sessão vai em todos os _meta.json).
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd

import bancada_comum as bc

BAUD = 921600
ETAPAS = ["prep", "rastreio", "viterbi", "sal_caminho", "envelope", "features", "cosseno"]


# ------------------------------------------------------------------ serial
class Placa:
    def __init__(self, porta, baud=BAUD, reset=True):
        import serial
        self.s = serial.Serial(porta, baud, timeout=60)
        if reset:
            # EN via RTS (placas DevKit com auto-reset); IO0 alto via DTR
            self.s.dtr = False
            self.s.rts = True
            time.sleep(0.1)
            self.s.rts = False
        self.espera("@@PRONTO", 20)

    def linha(self, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            b = self.s.readline()
            if not b:
                continue
            l = b.decode("utf-8", "replace").strip()
            if l.startswith("@@"):
                return l
        raise TimeoutError("sem resposta do firmware")

    def espera(self, prefixo, timeout=120):
        while True:
            l = self.linha(timeout)
            if l.startswith(prefixo):
                return l
            if l.startswith("@@ERRO"):
                raise RuntimeError(l)

    def cmd(self, texto, dados=b"", prefixo="@@", timeout=120):
        self.s.write(texto.encode() + b"\n")
        if dados:
            self.s.write(dados)
        l = self.linha(timeout)
        if l.startswith("@@ERRO"):
            raise RuntimeError(l)
        if not l.startswith(prefixo):
            raise RuntimeError(f"esperava {prefixo}, veio {l[:200]}")
        return l

    def json_de(self, l):
        return json.loads(l.split(" ", 1)[1])

    def modelo(self, caminho):
        b = Path(caminho).read_bytes()
        return self.cmd(f"MODEL {len(b)}", b, "@@OK MODEL")

    def janela(self, wid, y):
        import zlib
        y = np.ascontiguousarray(y, dtype="<f4")
        b = y.tobytes()
        r = self.json_de(self.cmd(f"WIN {wid} {y.size}", b, "@@RES", timeout=300))
        if r["crc32"] != zlib.crc32(b):
            raise RuntimeError(f"janela {wid}: CRC32 do chip {r['crc32']} != host {zlib.crc32(b)}"
                               " (bytes perdidos ou trocados na serial)")
        return r

    def fechar(self):
        self.s.close()


def sessao():
    return f"esp32-{bc.carimbo()}"


def grava_meta(caminho: Path, meta: dict):
    caminho.with_name(caminho.stem + "_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def bancada(alvo):
    p = bc.EMB / "results" / "manifests" / "bancada.json"
    if not p.exists():
        raise SystemExit("rode `info` antes: bancada.json não existe")
    b = json.loads(p.read_text(encoding="utf-8"))
    if b.get("alvo") != alvo:
        raise SystemExit(f"bancada.json é de {b.get('alvo')}, não de {alvo}")
    return b


# ------------------------------------------------------------------ comandos
def cmd_portas(a):
    from serial.tools import list_ports
    for p in list_ports.comports():
        print(p.device, "|", p.description, "|", p.hwid)


def cmd_info(a):
    pl = Placa(a.porta)
    info = pl.json_de(pl.cmd("INFO", prefixo="@@INFO"))
    pl.fechar()
    build = Path(f"C:/Espressif/build_p11_{a.alvo}")
    sdk = {}
    for l in (build / "sdkconfig").read_text(encoding="utf-8").splitlines():
        if l.startswith("CONFIG_") and any(k in l for k in (
                "CPU_FREQ", "SPIRAM", "ESPTOOLPY_FLASHSIZE", "COMPILER_OPTIMIZATION", "BT_ENABLED",
                "PM_ENABLE", "CONSOLE_UART_BAUDRATE", "FREERTOS_HZ", "IDF_TARGET=")):
            k, _, v = l.partition("=")
            sdk[k] = v
    nomes = {1: "ESP32", 2: "ESP32-S2", 9: "ESP32-S3", 5: "ESP32-C3", 13: "ESP32-C6"}
    b = {"categoria": "manifesto_da_bancada", "sessao": sessao(), "alvo": a.alvo, "porta": a.porta,
         "chip_modelo": nomes.get(info["chip_model"], f"model={info['chip_model']}"),
         "esp_chip_info": info, "cpu_mhz": info["cpu_hz"] / 1e6,
         "flash_mb": info["flash_bytes"] / 2**20, "psram_mb": info["psram_bytes"] / 2**20,
         "idf": info["idf"], "flags_nucleo": info["flags"], "variante": info["variante"],
         "sdkconfig": sdk, "wifi_bt": "nunca iniciados pelo firmware; CONFIG_BT_ENABLED=n",
         "projeto": bc.commit(), "gravado_em": bc.carimbo(),
         "size_txt": (build / "size.txt").read_text(encoding="utf-8", errors="replace")}
    p = bc.EMB / "results" / "manifests" / "bancada.json"
    if p.exists():
        p.rename(p.with_name(f"bancada_{json.loads(p.read_text(encoding='utf-8'))['gravado_em']}.json"))
    p.write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: b[k] for k in ("chip_modelo", "cpu_mhz", "flash_mb", "psram_mb", "idf")},
                     ensure_ascii=False))


def _grava_saida_caso(dir_out: Path, r: dict, n_classes: int):
    dir_out.mkdir(parents=True, exist_ok=True)
    etapas = {}
    def put(nome, v, dtype):
        v = np.asarray(v, dtype={"float32": "<f4", "int32": "<i4"}[dtype])
        ext = "f32" if dtype == "float32" else "i32"
        v.tofile(dir_out / f"{nome}.{ext}")
        etapas[nome] = {"arquivo": f"{nome}.{ext}", "dtype": dtype, "shape": list(v.shape)}
    put("vit_path", r["vit_path"], "int32")
    put("vit_score", [r["vit_score"]], "float32")
    put("vit_min_delta", [r["vit_min_delta"]], "float32")
    put("env_mod", r["env_mod"], "float32")
    put("features", r["features"], "float32")
    put("z", r["z"], "float32")
    put("cos", r["cos"][:n_classes], "float32")
    put("probs", r["probs"][:n_classes], "float32")
    (dir_out / "etapas.json").write_text(json.dumps(etapas, indent=1), encoding="utf-8")


def cmd_equiv(a):
    b = bancada(a.alvo)
    man = bc.manifesto()
    pl = Placa(a.porta)
    cout = bc.EMB / "results" / "c_out" / f"esp32_{a.alvo}" / "casos"
    linhas, mod_atual = [], None
    for c in man["casos"]:
        if c["modelo"] != mod_atual:
            pl.modelo(bc.GOLD / "modelos" / f"{c['modelo']}.bin")
            mod_atual = c["modelo"]
        r = pl.janela(len(linhas), bc.audio(c["id"]))
        n_cl = json.loads((bc.GOLD / "modelos" / f"{c['modelo']}.json").read_text(encoding="utf-8"))["n_classes"]
        _grava_saida_caso(cout / c["id"], r, n_cl)
        linhas.append({"id": c["id"], **{f"us_{k}": v for k, v in r["us"].items()},
                       "ws_bytes": r["ws_bytes"], "ws_na_psram": r["ws_na_psram"],
                       "heap_min_livre_interna": r["heap_min_livre_interna"]})
        print(c["id"], "ok", flush=True)
    pl.fechar()
    out = bc.EMB / "results" / "csv" / f"esp32_equiv_{a.alvo}_{bc.carimbo()}.csv"
    pd.DataFrame(linhas).to_csv(out, index=False)
    grava_meta(out, {"categoria": "medicao_no_ESP32", "sessao_bancada": b["sessao"],
                     "chip": b["chip_modelo"], "c_out": str(cout), "projeto": bc.commit()})
    print(f"saídas do chip em {cout}; compare com:\n  .venv\\Scripts\\python.exe "
          f"embarcados\\tests\\equivalence\\comparar.py --build esp32_{a.alvo} --tol-build float "
          f"--c-out {cout.parent}")


def cmd_lat(a):
    b = bancada(a.alvo)
    pl = Placa(a.porta)
    linhas = []
    for base in ("idmt", "a2212"):
        pl.modelo(bc.modelo_bin(base))
        ids, n_dist = bc.casos_latencia(base, a.n, a.aquecimento)
        for k, cid in enumerate(ids):
            r = pl.janela(k, bc.audio(cid))
            us = r["us"]
            tot = sum(us[e] for e in ETAPAS)
            linhas.append({"impl": f"c_{b['variante']}_esp32", "base": base, "janela": cid, "k": k,
                           "aquecimento": k < a.aquecimento, "n_janelas_distintas": n_dist,
                           **{f"{e}_ms": us[e] / 1e3 for e in ETAPAS}, "init_ms": us["init"] / 1e3,
                           "total_ms": tot / 1e3, "pred": int(np.argmax(r["probs"])),
                           "ws_bytes": r["ws_bytes"], "ws_na_psram": r["ws_na_psram"],
                           "heap_min_livre_interna": r["heap_min_livre_interna"],
                           "heap_min_livre_psram": r["heap_min_livre_psram"],
                           "pilha_folga_bytes": r["pilha_folga_bytes"]})
            if k % 20 == 0:
                print(base, k, f"{tot / 1e3:.1f} ms", flush=True)
    pl.fechar()
    df = pd.DataFrame(linhas)
    ts = bc.carimbo()
    out = bc.EMB / "results" / "csv" / f"esp32_lat_{a.alvo}_{ts}.csv"
    df.to_csv(out, index=False)
    meta = {"categoria": "medicao_no_ESP32", "sessao_bancada": b["sessao"], "chip": b["chip_modelo"],
            "cpu_mhz": b["cpu_mhz"], "idf": b["idf"], "flags": b["flags_nucleo"], "argv": vars(a),
            "projeto": bc.commit(), "relogio": "esp_timer_get_time em torno de cada chamada do nucleo",
            "nota": "janelas = casos nao-borda do ouro em ciclo; aquecimento excluido do resumo"}
    grava_meta(out, meta)
    res = {"meta": meta, "resumo": {}}
    for base, g in df[~df.aquecimento].groupby("base"):
        com_init = g.total_ms + g.init_ms
        livre_boot_int = b["esp_chip_info"]["heap_livre_interna"]
        livre_boot_ps = b["esp_chip_info"]["heap_livre_psram"]
        d = {"total_ms": bc.resumo(g.total_ms),
             "init_ms": bc.resumo(g.init_ms),
             "total_com_init_ms": bc.resumo(com_init),
             "fator_tempo_real_mediana": float(np.median(g.total_ms) / 1e3 / bc.DUR_S[base]),
             "fator_tempo_real_max": float(g.total_ms.max() / 1e3 / bc.DUR_S[base]),
             "fator_tempo_real_com_init_mediana": float(np.median(com_init) / 1e3 / bc.DUR_S[base]),
             # pico de uso = heap livre no boot (INFO, antes de alocar) - mínimo livre na sessão
             "pico_heap_interna_bytes": int(livre_boot_int - g.heap_min_livre_interna.min()),
             "pico_heap_psram_bytes": int(livre_boot_ps - g.heap_min_livre_psram.min()),
             "ws_bytes": int(g.ws_bytes.iloc[0]), "ws_na_psram": int(g.ws_na_psram.iloc[0]),
             "heap_min_livre_interna": int(g.heap_min_livre_interna.min()),
             "heap_min_livre_psram": int(g.heap_min_livre_psram.min()),
             "pilha_folga_bytes_min": int(g.pilha_folga_bytes.min())}
        for e in ETAPAS:
            d[f"{e}_ms"] = bc.resumo(g[f"{e}_ms"])
        res["resumo"][base] = d
    (bc.EMB / "results" / "metrics" / f"esp32_lat_{a.alvo}_{ts}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for base, d in res["resumo"].items():
        print(f"{base}: mediana {d['total_ms']['mediana']:.1f} ms, p95 {d['total_ms']['p95']:.1f}, "
              f"máx {d['total_ms']['max']:.1f}, RTF {d['fator_tempo_real_mediana']:.3f}")


def cmd_seq_imagem(a):
    """Sequência good -> broken -> heavyload com janelas LIMPAS de validação do IDMT (ouro)."""
    man = bc.manifesto()
    ordem = ["good", "broken", "heavyload"]
    jan, rot = [], []
    for ci, cl in enumerate(ordem):
        ids = [c["id"] for c in man["casos"] if c["tipo"] == "limpo" and c["base"] == "idmt"
               and c["classe"] == cl][:a.por_classe]
        for i in ids:
            jan.append(bc.audio(i))
            rot.append(ci)
    x = np.stack(jan)
    escala = float(np.abs(x).max() / 32767.0)
    q = np.round(x / escala).astype("<i2")
    n_jan, n_am = q.shape
    cab = b"P11S" + struct.pack("<IIf", n_jan, n_am, escala)
    img = cab + np.asarray(rot, dtype="<u4").tobytes() + q.tobytes()
    if len(img) > 0x2F0000:
        raise SystemExit(f"imagem de {len(img)} bytes não cabe na partição seq (0x2F0000)")
    ts = bc.carimbo()
    pasta = bc.EMB / "gold" / "seq_esp32"
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / f"seq_idmt_{ts}.bin"
    arq.write_bytes(img)
    # o mesmo áudio desquantizado, para a referência Python e o C do PC processarem igual
    (q.astype(np.float32) * np.float32(escala)).astype("<f4").tofile(pasta / f"seq_idmt_{ts}_audio.f32")
    (pasta / f"seq_idmt_{ts}.json").write_text(json.dumps(
        {"n_jan": n_jan, "n_amostras": n_am, "escala": escala, "rotulos": rot, "classes": ordem,
         "janelas": [c for c in []], "por_classe": a.por_classe, "bytes": len(img)}, indent=1),
        encoding="utf-8")
    print(arq, len(img), "bytes")


def deteccao(verdade, decisao):
    """Atraso (em janelas) de cada troca: primeira janela >= troca com decisão = novo estado."""
    out = []
    for t in range(1, len(verdade)):
        if verdade[t] != verdade[t - 1]:
            fim = next((u for u in range(t + 1, len(verdade)) if verdade[u] != verdade[t]), len(verdade))
            d = next((u - t for u in range(t, fim) if decisao[u] == verdade[t]), None)
            out.append({"troca_em": t, "de": verdade[t - 1], "para": verdade[t], "atraso_janelas": d})
    return out


def cmd_seq(a):
    b = bancada(a.alvo)
    pl = Placa(a.porta)
    pl.modelo(bc.modelo_bin("idmt"))
    pl.s.write(f"SEQ {a.cp} {a.n_logprob} {a.temp}\n".encode())
    linhas = []
    while True:
        l = pl.linha(600)
        if l.startswith("@@SEQFIM"):
            break
        if l.startswith("@@ERRO"):
            raise RuntimeError(l)
        r = pl.json_de(l)
        linhas.append({"jan": r["jan"], "verdade": r["verdade"], "decisao_online": r["decisao"],
                       "us_total": sum(r["us"][e] for e in ETAPAS), "us_hmm": r["us_hmm"],
                       "us_leitura_flash": r["us_leitura_flash"],
                       **{f"p{k}": v for k, v in enumerate(r["probs"])}})
    pl.fechar()
    df = pd.DataFrame(linhas)
    det = deteccao(df.verdade.tolist(), df.decisao_online.tolist())
    ts = bc.carimbo()
    out = bc.EMB / "results" / "csv" / f"esp32_seq_{a.alvo}_{ts}.csv"
    df.to_csv(out, index=False)
    grava_meta(out, {"categoria": "medicao_no_ESP32", "sessao_bancada": b["sessao"],
                     "chip": b["chip_modelo"], "imagem": a.imagem, "cp": a.cp,
                     "n_logprob": a.n_logprob, "temp": a.temp, "deteccao": det,
                     "passo_s": 3.0, "projeto": bc.commit()})
    print(json.dumps(det))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("portas")
    for nome in ("info", "equiv", "lat", "seq"):
        p = sp.add_parser(nome)
        p.add_argument("--porta", required=True)
        # esp32s3oct = ESP32-S3 com PSRAM octal (build_fw.bat esp32s3oct)
        p.add_argument("--alvo", required=True, choices=["esp32", "esp32s3", "esp32s3oct"])
        if nome == "lat":
            p.add_argument("--n", type=int, default=200)
            p.add_argument("--aquecimento", type=int, default=5)
        if nome == "seq":
            p.add_argument("--imagem", required=True)
            # mesmos valores do HMM dos experimentos (sequencias.avaliar: temp=1, n_logprob=-50)
            p.add_argument("--cp", type=float, default=2.0)
            p.add_argument("--n-logprob", type=float, default=-50.0)
            p.add_argument("--temp", type=float, default=1.0)
    p = sp.add_parser("seq-imagem")
    p.add_argument("--por-classe", type=int, default=8)
    a = ap.parse_args(argv)
    {"portas": cmd_portas, "info": cmd_info, "equiv": cmd_equiv, "lat": cmd_lat,
     "seq-imagem": cmd_seq_imagem, "seq": cmd_seq}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Gravação para placas cujo reset automático não puxa o GPIO0 (a placa CH9102 usada em 24-25/10).

    (ambiente do IDF) python -u gravar_boot_espera.py COM3 C:\\Espressif\\build_p11_esp32_lowmem 600

Segure o BOOT enquanto o script roda: ele pulsa o EN pela RTS, vê o modo de boot e, ao
entrar em download, grava só o app (0x10000) comprimido. A imagem de sequência e o bootloader
são gravados uma vez com gravar_tudo.bat.

Pulsa EN pela RTS a cada ~0,6 s e mostra o modo de boot; quando vir download, grava o app.
Se a porta cair (USB instável), reabre e continua; se a gravação cair, volta a esperar."""
import re
import subprocess
import sys
import time

import serial

porta, build, fim = sys.argv[1], sys.argv[2], time.time() + float(sys.argv[3])


def abre():
    while time.time() < fim:
        try:
            s = serial.Serial()
            s.port, s.baudrate, s.timeout = porta, 115200, 0.1
            s.dtr = False
            s.rts = False
            s.open()
            return s
        except serial.SerialException:
            time.sleep(0.5)
    return None


def espera_download():
    s = abre()
    while s and time.time() < fim:
        try:
            s.rts = True
            time.sleep(0.1)
            s.rts = False
            t = time.time()
            buf = b""
            while time.time() - t < 0.5:
                buf += s.read(4096)
        except serial.SerialException:
            print(time.strftime("%H:%M:%S"), "porta caiu; reabrindo", flush=True)
            try:
                s.close()
            except Exception:
                pass
            time.sleep(1)
            s = abre()
            continue
        m = re.findall(rb"boot:0x([0-9a-f]+)", buf)
        modo = m[-1].decode() if m else "?"
        print(time.strftime("%H:%M:%S"), "boot:0x" + modo, flush=True)
        if b"DOWNLOAD" in buf or (m and int(modo, 16) & 0x10 == 0):
            s.close()
            return True
    if s:
        s.close()
    return False


while time.time() < fim:
    if not espera_download():
        break
    print("ENTROU EM DOWNLOAD; gravando", flush=True)
    r = subprocess.run([sys.executable, "-u", "-m", "esptool", "--chip", "esp32", "--port", porta, "-b", "115200",
                        "--before", "no_reset", "--after", "hard_reset", "write_flash", "-z", "0x10000",
                        build + r"\p11_bancada.bin"])
    if r.returncode == 0:
        print("GRAVADO_OK", flush=True)
        sys.exit(0)
    print("gravação falhou; segure o BOOT de novo", flush=True)
    time.sleep(1)
print("SEM_DOWNLOAD")
sys.exit(3)

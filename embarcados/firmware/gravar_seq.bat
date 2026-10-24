@echo off
rem Grava a imagem da sequência (pil_esp32.py seq-imagem) na partição "seq" (0x110000,
rem ver partitions.csv).
rem   gravar_seq.bat COM5 <imagem.bin>
setlocal
if "%2"=="" (echo uso: gravar_seq.bat ^<COMx^> ^<imagem.bin^> & exit /b 1)
call "%~dp0idf_env.bat"
python -m esptool --port %1 --baud 921600 write_flash 0x110000 "%~2" || exit /b 1
echo GRAVADA SEQ %2

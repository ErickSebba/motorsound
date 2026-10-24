@echo off
rem Grava firmware (bootloader, partições, app) e a imagem da sequência numa só conexão.
rem Para placas sem reset automático para o modo de gravação: faça BOOT + EN antes e use
rem "no_reset" (a conexão não reinicia o chip).
rem   gravar_tudo.bat COM3 esp32_lowmem <imagem_seq.bin> [no_reset|default_reset]
setlocal
if "%3"=="" (echo uso: gravar_tudo.bat ^<COMx^> ^<build: esp32, esp32_lowmem...^> ^<imagem.bin^> [antes] & exit /b 1)
set ANTES=%4
if "%ANTES%"=="" set ANTES=no_reset
call "%~dp0idf_env.bat"
set B=C:\Espressif\build_p11_%2
cd /d "%B%"
python -m esptool --chip esp32 --port %1 -b 460800 --before %ANTES% --after hard_reset write_flash @flash_args 0x110000 "%~3" || exit /b 1
echo GRAVADO_TUDO %2

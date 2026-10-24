@echo off
rem Compila o firmware para um alvo e uma variante, e grava o relatório de tamanho.
rem   build_fw.bat esp32s3            ESP32-S3, PSRAM quad (ou sem PSRAM), variante ref
rem   build_fw.bat esp32s3oct         ESP32-S3 com PSRAM octal (N8R8, N16R8...), variante ref
rem   build_fw.bat esp32              ESP32 (WROOM/WROVER), variante ref
rem   build_fw.bat esp32s3 espdsp     variante p11_c_espdsp
rem O build sai em C:\Espressif\build_p11_<alvo>[_<variante>] (caminho sem parênteses: o IDF
rem não lida bem com "Artigo3(MotorSound)").
setlocal
set ALVO=%1
if "%ALVO%"=="" set ALVO=esp32s3
set P11_VARIANTE=%2
if "%P11_VARIANTE%"=="" set P11_VARIANTE=ref
set CHIP=%ALVO%
set DEFS=%~dp0sdkconfig.defaults
if "%ALVO%"=="esp32s3oct" (
  set CHIP=esp32s3
  set "DEFS=%~dp0sdkconfig.defaults;%~dp0sdkconfig.defaults.s3oct"
)
call "%~dp0idf_env.bat"
set B=C:\Espressif\build_p11_%ALVO%
if not "%P11_VARIANTE%"=="ref" set B=%B%_%P11_VARIANTE%
idf.py -C "%~dp0." -B "%B%" -DIDF_TARGET=%CHIP% -DSDKCONFIG="%B%\sdkconfig" "-DSDKCONFIG_DEFAULTS=%DEFS%" build || exit /b 1
idf.py -C "%~dp0." -B "%B%" -DSDKCONFIG="%B%\sdkconfig" size > "%B%\size.txt" || exit /b 1
idf.py -C "%~dp0." -B "%B%" -DSDKCONFIG="%B%\sdkconfig" size-components > "%B%\size_components.txt" || exit /b 1
echo FW OK %ALVO% %P11_VARIANTE%

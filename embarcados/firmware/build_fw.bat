@echo off
rem Compila o firmware para um alvo (esp32 ou esp32s3) e uma variante, e grava o relatório
rem de tamanho.
rem   build_fw.bat esp32s3            (variante ref = p11_c_ref)
rem   build_fw.bat esp32s3 espdsp     (variante p11_c_espdsp)
rem O build sai em C:\Espressif\build_p11_<alvo>[_<variante>] (caminho sem parênteses: o IDF
rem não lida bem com "Artigo3(MotorSound)").
setlocal
set ALVO=%1
if "%ALVO%"=="" set ALVO=esp32s3
set P11_VARIANTE=%2
if "%P11_VARIANTE%"=="" set P11_VARIANTE=ref
call "%~dp0idf_env.bat"
set B=C:\Espressif\build_p11_%ALVO%
if not "%P11_VARIANTE%"=="ref" set B=%B%_%P11_VARIANTE%
idf.py -C "%~dp0." -B "%B%" -DIDF_TARGET=%ALVO% -DSDKCONFIG="%B%\sdkconfig" build || exit /b 1
idf.py -C "%~dp0." -B "%B%" -DSDKCONFIG="%B%\sdkconfig" size > "%B%\size.txt" || exit /b 1
idf.py -C "%~dp0." -B "%B%" -DSDKCONFIG="%B%\sdkconfig" size-components > "%B%\size_components.txt" || exit /b 1
echo FW OK %ALVO% %P11_VARIANTE%

@echo off
rem Compila o firmware para um alvo (esp32 ou esp32s3) e grava o relatório de tamanho.
rem   build_fw.bat esp32s3
rem O build sai em C:\Espressif\build_p11_<alvo> (caminho sem parênteses: o IDF não lida
rem bem com "Artigo3(MotorSound)").
setlocal
set ALVO=%1
if "%ALVO%"=="" set ALVO=esp32s3
call "%~dp0idf_env.bat"
set B=C:\Espressif\build_p11_%ALVO%
idf.py -C "%~dp0." -B "%B%" -DIDF_TARGET=%ALVO% -DSDKCONFIG="%B%\sdkconfig" build || exit /b 1
idf.py -C "%~dp0." -B "%B%" -DSDKCONFIG="%B%\sdkconfig" size > "%B%\size.txt" || exit /b 1
idf.py -C "%~dp0." -B "%B%" -DSDKCONFIG="%B%\sdkconfig" size-components > "%B%\size_components.txt" || exit /b 1
echo FW OK %ALVO%

@echo off
rem Grava o firmware já compilado (build_fw.bat) na placa.
rem   gravar_fw.bat esp32s3 COM5
setlocal
if "%2"=="" (echo uso: gravar_fw.bat ^<esp32^|esp32s3^> ^<COMx^> & exit /b 1)
call "%~dp0idf_env.bat"
set B=C:\Espressif\build_p11_%1
idf.py -C "%~dp0." -B "%B%" -DSDKCONFIG="%B%\sdkconfig" -p %2 flash || exit /b 1
echo GRAVADO %1 %2

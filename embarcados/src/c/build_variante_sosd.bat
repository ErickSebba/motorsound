@echo off
rem build_variante_sosd.bat -- build float com P11_SOS_DOUBLE=ON (sosfilt do envelope em
rem double) em embarcados\build-float-sosd. Opção de precisão, desligada no build padrão.
setlocal
set "AQUI=%~dp0"
pushd "%AQUI%..\.."
set "EMB=%CD%"
popd
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cmake -S "%EMB%" -B "%EMB%\build-float-sosd" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DP11_REAL=float -DP11_SOS_DOUBLE=ON
if errorlevel 1 exit /b 1
cmake --build "%EMB%\build-float-sosd"
if errorlevel 1 exit /b 1
echo BUILD OK
endlocal

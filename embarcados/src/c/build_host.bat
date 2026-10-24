@echo off
rem build_host.bat -- compila o núcleo p11 no PC com MSVC 2022 Build Tools + CMake (NMake).
rem Gera embarcados\build-float (P11_REAL=float, o do MCU) e embarcados\build-double (só PC).
rem Uso (de qualquer pasta):  cmd /c "<caminho>\build_host.bat"
setlocal
set "AQUI=%~dp0"
pushd "%AQUI%..\.."
set "EMB=%CD%"
popd
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64 falhou & exit /b 1)
for %%R in (float double) do (
  cmake -S "%EMB%" -B "%EMB%\build-%%R" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DP11_REAL=%%R
  if errorlevel 1 exit /b 1
  cmake --build "%EMB%\build-%%R"
  if errorlevel 1 exit /b 1
)
echo BUILD OK
endlocal

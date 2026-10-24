@echo off
rem Ambiente do ESP-IDF v5.5.6 instalado em C:\Espressif (fora do repositório).
rem Tira o .venv do projeto do PATH: o export do IDF recusa rodar dentro de outro venv.
set VIRTUAL_ENV=
set "PATH=%PATH:.venv\Scripts;=%"
set IDF_TOOLS_PATH=C:\Espressif\tools
set IDF_PATH=C:\Espressif\esp-idf-v5.5.6
set IDF_PYTHON_BASE=C:\Users\erick\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none
set "PATH=%IDF_PYTHON_BASE%;%PATH%"
call "%IDF_PATH%\export.bat" >nul

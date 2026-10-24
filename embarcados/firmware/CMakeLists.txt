# Firmware de medição da p11 (ESP-IDF v5.5.6). Processor-in-the-loop pela serial e fluxo
# contínuo a partir de uma partição de dados na flash. Ver firmware/README.md.
#
# Variante: -DP11_VARIANTE=ref (padrão: mesmo núcleo C e mesma FFT portável do PC).
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(p11_bancada)

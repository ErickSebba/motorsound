# O núcleo C é compilado direto de embarcados/src/c (mesmo fonte do PC), sem cópia.
#
# Variantes (variável de ambiente P11_VARIANTE, lida por build_fw.bat):
#   ref    (padrão) p11_c_ref    : mesmo núcleo e mesma FFT portável do PC
#   espdsp          p11_c_espdsp : FFT de N potência de 2 pela ESP-DSP (dsps_fft2r_fc32);
#                                  Hilbert (16000/48000) e Welch de 1000 seguem no C portável
#   lowmem          p11_c_lowmem : a mesma conta em pool de blocos pequenos (ESP32 sem PSRAM;
#                                  ver src/c/p11_lowmem.h); só janelas de 1 s cabem
set(P11_SRC "${CMAKE_CURRENT_LIST_DIR}/../../src/c")
set(P11_VAR "$ENV{P11_VARIANTE}")
if(P11_VAR STREQUAL "")
  set(P11_VAR "ref")
endif()

set(REQ esp_timer esp_partition driver esp_driver_uart esp_psram spi_flash)
if(P11_VAR STREQUAL "espdsp")
  list(APPEND REQ espressif__esp-dsp)
endif()

idf_component_register(
    SRCS "main.c"
         "${P11_SRC}/p11_fft.c"
         "${P11_SRC}/p11_core.c"
         "${P11_SRC}/p11_decisao.c"
         "${P11_SRC}/p11_lowmem.c"
    INCLUDE_DIRS "." "${P11_SRC}"
    REQUIRES ${REQ})

# Mesmas definições do build float do PC. -ffp-contract=off: o Xtensa tem madd.s (FMA);
# fundir multiplicação e soma mudaria o arredondamento em relação à referência.
# -Wno-error=maybe-uninitialized: falso positivo em p11_decisao.c (a[] em hmm_online_passo,
# preenchido para j < S com S >= 1); mantido como aviso para não mexer no fonte verificado.
target_compile_definitions(${COMPONENT_LIB} PRIVATE P11_REAL=float P11_REAL_EH_DOUBLE=0)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-variable
                       -Wno-unused-function -Wno-error=maybe-uninitialized)

if(P11_VAR STREQUAL "espdsp")
  target_compile_definitions(${COMPONENT_LIB} PRIVATE P11_USAR_ESP_DSP=1
      P11_VARIANTE="p11_c_espdsp" P11_FLAGS="-O2 -ffp-contract=off P11_REAL=float P11_USAR_ESP_DSP")
elseif(P11_VAR STREQUAL "lowmem")
  target_compile_definitions(${COMPONENT_LIB} PRIVATE P11_LOWMEM=1
      P11_VARIANTE="p11_c_lowmem" P11_FLAGS="-O2 -ffp-contract=off P11_REAL=float P11_LOWMEM")
else()
  target_compile_definitions(${COMPONENT_LIB} PRIVATE
      P11_VARIANTE="p11_c_ref" P11_FLAGS="-O2 -ffp-contract=off P11_REAL=float")
endif()

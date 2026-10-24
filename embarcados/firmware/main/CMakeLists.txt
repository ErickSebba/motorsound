# O núcleo C é compilado direto de embarcados/src/c (mesmo fonte do PC), sem cópia.
set(P11_SRC "${CMAKE_CURRENT_LIST_DIR}/../../src/c")

idf_component_register(
    SRCS "main.c"
         "${P11_SRC}/p11_fft.c"
         "${P11_SRC}/p11_core.c"
         "${P11_SRC}/p11_decisao.c"
    INCLUDE_DIRS "." "${P11_SRC}"
    REQUIRES esp_timer esp_partition driver esp_driver_uart esp_psram spi_flash)

# Mesmas definições do build float do PC. -ffp-contract=off: o Xtensa tem madd.s (FMA);
# fundir multiplicação e soma mudaria o arredondamento em relação à referência.
target_compile_definitions(${COMPONENT_LIB} PRIVATE P11_REAL=float P11_REAL_EH_DOUBLE=0)
# -Wno-error=maybe-uninitialized: falso positivo em p11_decisao.c (a[] em hmm_online_passo,
# preenchido para j < S com S >= 1); mantido como aviso para não mexer no fonte verificado.
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-variable
                       -Wno-unused-function -Wno-error=maybe-uninitialized)

# Registro das flags no binário (vão para o INFO e para bancada.json)
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    P11_FLAGS="-O2 -ffp-contract=off P11_REAL=float"
    P11_VARIANTE="p11_c_ref")

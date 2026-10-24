# CMake do host (PC) para o núcleo p11. O firmware ESP-IDF terá o próprio CMakeLists em
# firmware/ (ainda não existe: BLOQUEIOS.md B2).
#   cmake -S embarcados -B embarcados/build-float  -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DP11_REAL=float
#   cmake -S embarcados -B embarcados/build-double -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DP11_REAL=double
cmake_minimum_required(VERSION 3.16)
project(p11 C)

set(P11_REAL "float" CACHE STRING "tipo real do núcleo: float (MCU) ou double (só PC)")
set_property(CACHE P11_REAL PROPERTY STRINGS float double)
option(P11_SOS_DOUBLE "build float: sosfilt do envelope em double (opção de precisão, não de algoritmo)" OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(P11_REAL STREQUAL "double")
  set(P11_EH_DOUBLE 1)
else()
  set(P11_EH_DOUBLE 0)
endif()

add_library(p11 STATIC
  src/c/p11_fft.c
  src/c/p11_core.c
  src/c/p11_decisao.c)
target_include_directories(p11 PUBLIC src/c)
target_compile_definitions(p11 PUBLIC P11_REAL=${P11_REAL} P11_REAL_EH_DOUBLE=${P11_EH_DOUBLE}
                           _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
if(P11_SOS_DOUBLE)
  target_compile_definitions(p11 PUBLIC P11_SOS_DOUBLE=1)
endif()

if(MSVC)
  # /O2 (equivalente ao -O2) e /fp:precise (sem contração em FMA, sem reassociação)
  target_compile_options(p11 PUBLIC /O2 /fp:precise /W3 /utf-8)
else()
  target_compile_options(p11 PUBLIC -O2 -ffp-contract=off -Wall -Wextra)
  target_link_libraries(p11 PUBLIC m)
endif()

add_executable(p11_host src/c/p11_host.c)
target_link_libraries(p11_host PRIVATE p11)

add_executable(p11_testes tests/unit/p11_testes.c)
target_link_libraries(p11_testes PRIVATE p11)

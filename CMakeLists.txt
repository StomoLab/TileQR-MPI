cmake_minimum_required(VERSION 3.16)
project(TileQR-MPI LANGUAGES CXX)

# 詳細出力 (icecream-cpp による IC マクロ) の有効/無効。
option(COUT "Enable detailed trace output via icecream-cpp" OFF)

# 残差チェック (||A - QR||_F / ||A||_F) の有効/無効。
option(CHECK "Enable residual check after factorization" OFF)

# 単一構成ジェネレータ向けに既定ビルドタイプを設定。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 依存: MPI と PLASMA (局所タイルカーネル GEQRT/TSQRT/ORMQR/TSMQR の core ルーチン)。
find_package(MPI REQUIRED)

# PLASMA。インストールされた plasmaConfig.cmake は空でターゲットを定義しないため、
# 生成済みの plasmaTargets.cmake を直接取り込む (plasma::plasma_core_blas を定義)。
set(plasma_DIR /opt/PLASMA/lib/cmake/plasma CACHE PATH "PLASMA CMake package dir")
if(NOT TARGET plasma::plasma_core_blas)
  include(${plasma_DIR}/plasmaTargets.cmake)
endif()

add_executable(tileqr main.cpp)
target_include_directories(tileqr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tileqr PRIVATE MPI::MPI_CXX plasma::plasma_core_blas)

if(NOT MSVC)
  target_compile_options(tileqr PRIVATE -Wall -Wextra)
endif()

# COUT=ON のとき icecream-cpp(単一ヘッダ)を取得し、詳細出力を有効化する。
# サードパーティヘッダなので SYSTEM 扱いにして警告を抑制する。
if(COUT)
  include(FetchContent)
  FetchContent_Declare(
    icecream_cpp
    URL https://raw.githubusercontent.com/renatoGarcia/icecream-cpp/v1.0.0/icecream.hpp
    DOWNLOAD_NO_EXTRACT TRUE
  )
  FetchContent_MakeAvailable(icecream_cpp)
  target_include_directories(tileqr SYSTEM PRIVATE ${icecream_cpp_SOURCE_DIR})
  target_compile_definitions(tileqr PRIVATE COUT)
endif()

if(CHECK)
  target_compile_definitions(tileqr PRIVATE CHECK)
endif()

# cross compile settings
set(CMAKE_SYSTEM_NAME Windows)

# 设置 Boost 库的根目录
set(BOOST_ROOT "/usr")

# 添加 Boost 库的搜索路径
list(APPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")

message(STATUS "cross compile for windows_64")
set(TOOLCHAIN_PATH "/usr")
set(CMAKE_C_COMPILER "${TOOLCHAIN_PATH}/bin/x86_64-w64-mingw32-gcc")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PATH}/bin/x86_64-w64-mingw32-g++")

set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

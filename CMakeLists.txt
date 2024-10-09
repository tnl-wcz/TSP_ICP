cmake_minimum_required(VERSION 2.8)
project(TSP)

#--- CMake configuration
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#--- Compiler configuration
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

#--- OpenMP (cannot work in clang)
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

#--- Eigen3
include_directories("/home/wcz/TSP/include/eigen")

#--- Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
set(Boost_DEBUG ON)
# 设置 Boost 库的根目录
set(BOOST_ROOT "/usr/include/")

# 设置 Boost 库的组件路径
set(Boost_INCLUDE_DIRS "${BOOST_ROOT}")
set(BOOST_LIBRARYDIR /usr/lib/x86_64-linux-gnu)
# 查找 Boost 库
find_package(Boost REQUIRED COMPONENTS system filesystem iostreams serialization)

#--- PCL
find_package(PCL REQUIRED COMPONENTS common io sample_consensus segmentation)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 查找 OpenCV 库
# find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED)

#--- Build example
include_directories(.)
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")
add_executable(FRICP ${SOURCES} ${HEADERS})
# target_link_libraries(FRICP ${CERES_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_SAMPLE_CONSENSUS_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES} Eigen3::Eigen lz4)
target_link_libraries(FRICP
    ${CERES_LIBRARIES}
    ${PCL_COMMON_LIBRARIES}
    ${PCL_IO_LIBRARIES}
    ${PCL_SAMPLE_CONSENSUS_LIBRARIES}
    ${PCL_SEGMENTATION_LIBRARIES}
    Eigen3::Eigen
    lz4
    ${OpenCV_LIBRARIES}  # 添加 OpenCV 的链接库
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SERIALIZATION_LIBRARY}
)


if(APPLE OR UNIX)
    #--- Deploy data folder link
    execute_process(COMMAND ln -f -s ${CMAKE_SOURCE_DIR}/data WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    #--- "Dummy" target to have data appear in QtCreator
    add_custom_target(data SOURCES ${CMAKE_SOURCE_DIR}/data)
else()
    file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${PROJECT_BINARY_DIR})
endif()


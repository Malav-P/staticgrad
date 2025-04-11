# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(StaticGrad)

message(STATUS "This is the source dir: ${CMAKE_SOURCE_DIR}")
option(BUILD_TESTING "Build unit tests" OFF)


# Compiler settings
# Find clang++ and g++ in the system PATH
find_program(CLANGXX_EXECUTABLE NAMES clang++)
find_program(GCCXX_EXECUTABLE NAMES g++)

# Choose clang++ if found, otherwise fall back to g++
if (CLANGXX_EXECUTABLE)
    set(CMAKE_CXX_COMPILER "${CLANGXX_EXECUTABLE}")
    message(STATUS "Using Clang++: ${CLANGXX_EXECUTABLE}")
elseif (GCCXX_EXECUTABLE)
    set(CMAKE_CXX_COMPILER "${GCCXX_EXECUTABLE}")
    message(STATUS "Using G++: ${GCCXX_EXECUTABLE}")
else()
    message(FATAL_ERROR "No suitable C++ compiler found")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall -std=c++14 -fPIC")

# Directory locations
set(REPO_PREFIX "${CMAKE_SOURCE_DIR}/gpt2_python/")
set(SRCDIR "StaticGrad/src")
set(INCDIR "StaticGrad/include")
set(LIB_DIR "lib")


# Find BLAS routines
find_library(ACCELERATE_LIBRARIES Accelerate)
if (ACCELERATE_LIBRARIES)
    message(STATUS "Accelerate framework found: ${ACCELERATE_LIBRARIES}")
    set(BLAS_LIBRARIES ${ACCELERATE_LIBRARIES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DACCELERATE_NEW_LAPACK")
    add_definitions(-DAPPLE)
else()
    message(STATUS "Accelerate framework not found. Checking for BLAS...")
    
    # Find BLAS
    find_package(BLAS REQUIRED)
    if (BLAS_FOUND)
        message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
        set(BLAS_LIBRARIES ${BLAS_LIBRARIES})
    else()
        message(FATAL_ERROR "Neither Accelerate framework nor BLAS found.")
    endif()
endif()


# Add include directories
include_directories(${INCDIR} )

# Add source files
file(GLOB SRCS ${SRCDIR}/*.cpp)

# Define library
add_library(StaticGrad STATIC ${SRCS})

# Define it as a preprocessor definition
target_compile_definitions(StaticGrad PRIVATE REPO_PREFIX="${REPO_PREFIX}")

# Link StaticGrad with the BLAS libraries
target_link_libraries(StaticGrad PUBLIC ${BLAS_LIBRARIES})
set_target_properties(StaticGrad PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${LIB_DIR})

if (BUILD_TESTING)
    set(TEST_SRCDIR "test/src")
    set(TEST_INCDIR "test/include")
    set(BIN_DIR "bin")
    file(GLOB TEST_SRCS ${TEST_SRCDIR}/*.cpp)
    include_directories(${TEST_INCDIR} ${GTEST_INCLUDE_DIRS})

    # Find GoogleTest package
    find_package(GTest REQUIRED)
    if (GTest_FOUND)
        message(STATUS "GTest package found: ${GTEST_LIBRARIES}")
    else()
        message(FATAL_ERROR "GTest not found")
    endif()

    # Add source files for test common code
    add_library(test_common OBJECT ${TEST_SRCDIR}/test_common.cpp)

    # Define and add test executables
    set(TESTS
        att add layernorm matmul embedding transformerblock gpt2 utils datastream tokenizer interface inference train
    )

    foreach(test ${TESTS})
        set(test_src "../${TEST_SRCDIR}/${test}_test.cpp")
        if (EXISTS ${test_src})
            add_executable(${test} ${test_src} $<TARGET_OBJECTS:test_common>)
            target_link_libraries(${test} StaticGrad ${GTEST_LIBRARIES} ${LIBS} ${BLAS_LIBRARIES})
            set_target_properties(${test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})
            target_compile_definitions(${test} PRIVATE REPO_PREFIX="${REPO_PREFIX}")
        else()
            message(WARNING "Test source file ${test_src} does not exist. Skipping ${test}.")
        endif()
    endforeach()

    add_custom_target(tests DEPENDS ${TESTS})
    add_custom_target(run DEPENDS ${TESTS})

    foreach(test_exec ${TESTS})
        add_custom_command(TARGET run
            POST_BUILD
            COMMAND ${BIN_DIR}/${test_exec}
            COMMENT "Running ${test_exec}"
        )
    endforeach()
endif()


# Install the library and headers
install(TARGETS StaticGrad
    EXPORT StaticGradTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# set install prefix
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

# Install headers
install(FILES
    ${INCDIR}/interface.hpp
    DESTINATION include
)


# Export targets for downstream use
install(EXPORT StaticGradTargets
    FILE StaticGradTargets.cmake
    NAMESPACE StaticGrad::
    DESTINATION lib/cmake/StaticGrad
)

# Generate a Config file for find_package
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/StaticGradConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StaticGradConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/StaticGradConfig.cmake"
    INSTALL_DESTINATION lib/cmake/StaticGrad
)

# Install the config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/StaticGradConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/StaticGradConfigVersion.cmake"
    DESTINATION lib/cmake/StaticGrad
)

# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(StaticGrad)

# Compiler settings
set(CMAKE_CXX_COMPILER "clang++")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -std=c++14 -fPIC -Wall")

# Directory locations
set(SRCDIR "StaticGrad/src")
set(INCDIR "StaticGrad/include")
set(LIB_DIR "lib")
set(TEST_SRCDIR "test/src")
set(TEST_INCDIR "test/include")
set(TEST_BUILD_DIR "test/build")
set(BIN_DIR "bin")


# Find GoogleTest package
find_package(GTest REQUIRED)
if (GTest_FOUND)
    message(STATUS "GTest package found: ${GTEST_LIBRARIES}")
else()
    message(FATAL_ERROR "GTest not found")
endif()

# Find the Accelerate framework
find_library(ACCELERATE_LIBRARIES Accelerate)
if (ACCELERATE_LIBRARIES)
    message(STATUS "Accelerate framework found: ${ACCELERATE_LIBRARIES}")
    set(BLAS_LIBRARIES ${ACCELERATE_LIBRARIES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DACCELERATE_NEW_LAPACK")
else()
    message(STATUS "Accelerate framework not found. Checking for OpenBLAS...")
    
    # Find OpenBLAS
    find_package(OpenBLAS REQUIRED)
    if (OpenBLAS_FOUND)
        message(STATUS "OpenBLAS found: ${OpenBLAS_LIBRARIES}")
        set(BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
    else()
        message(FATAL_ERROR "Neither Accelerate framework nor OpenBLAS found.")
    endif()
endif()


# Add include directories
include_directories(${INCDIR} ${TEST_INCDIR} ${GTEST_INCLUDE_DIRS} "/opt/homebrew/include")

# # Link directories if needed (usually not necessary with `find_package`)
# link_directories(${GTEST_LIBRARY_DIRS})


# Add source files
file(GLOB SRCS ${SRCDIR}/*.cpp)
file(GLOB TEST_SRCS ${TEST_SRCDIR}/*.cpp)

# Define library
add_library(StaticGrad STATIC ${SRCS})
# Link StaticGrad with the BLAS libraries
target_link_libraries(StaticGrad PRIVATE ${BLAS_LIBRARIES})

set_target_properties(StaticGrad PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${LIB_DIR})

# Add source files for test common code
add_library(test_common OBJECT ${TEST_SRCDIR}/test_common.cpp)
set_target_properties(test_common PROPERTIES OBJECT_OUTPUT_DIRECTORY ${TEST_BUILD_DIR})

# Define and add test executables
set(TESTS
    att add layernorm matmul softmax encoder transformerblock gpt2 utils datastream tokenizer interface inference train
)

# Add your test executables
foreach(test ${TESTS})
    set(test_src "../${TEST_SRCDIR}/${test}_test.cpp")
    if (EXISTS ${test_src})
        add_executable(${test} ${test_src} $<TARGET_OBJECTS:test_common>)
        target_link_libraries(${test} StaticGrad ${GTEST_LIBRARIES} ${LIBS} ${BLAS_LIBRARIES})
        set_target_properties(${test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})
    else()
        message(WARNING "Test source file ${test_src} does not exist. Skipping ${test}.")
    endif()
endforeach()


# Define a custom target to build all test binaries
add_custom_target(tests
    DEPENDS ${TESTS}
)


# # Add custom target for running tests TODO
# add_custom_target(run
#     COMMAND ${CMAKE_COMMAND} -E echo "Running tests..."
#     COMMAND ${CMAKE_COMMAND} -E env zsh -c 'for test in ${BIN_DIR}/*; do ./$test; done'
#     DEPENDS tests
# )

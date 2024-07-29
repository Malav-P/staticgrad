# Compiler (g++ or clang++ both work)
CC = g++

# Compiler flags
CXXFLAGS := -Wextra -std=c++14 -fPIC -framework Accelerate -DACCELERATE_NEW_LAPACK -Wall

# Directory locations
SRCDIR := StaticGrad/src
INCDIR := StaticGrad/include
SRC_BUILD_DIR := StaticGrad/build
TEST_SRCDIR := test/src
TEST_INCDIR := test/include
TEST_BUILD_DIR := test/build

# Library name
LIBNAME := ${SRC_BUILD_DIR}/libStaticGrad.a

# List of source files
SRCS := $(wildcard ${SRCDIR}/*.cpp)

# List of object files in the new directory
OBJS := $(patsubst ${SRCDIR}/%.cpp, ${SRC_BUILD_DIR}/%.o, $(SRCS))

# List of test source files
TEST_SRCS := $(wildcard ${TEST_SRCDIR}/*.cpp)

# List of test object files
TEST_OBJS := $(TEST_SRCS:.cpp=.o)

# Library directories
LIB_DIRS = -L/opt/homebrew/lib

# Libraries to link against
LIBS = -lgtest -lgtest_main -pthread

# Include dirs
INCLUDE_DIRS = -I/opt/homebrew/include/ -I./StaticGrad/include/ -I ${TEST_INCDIR}

# Binary directory
BIN_DIR = test/bin

# Source files
SOURCES = $(SRC_BUILD_DIR)/classes.o $(TEST_BUILD_DIR)/test_common.o

# List of Tests
TESTS = att add layernorm matmul softmax encoder transformerblock gpt2 utils datastream tokenizer interface train

# Phony target to clean up
.PHONY: clean test run

# Default target: build the library
all: ${LIBNAME}

# Rule to build the library
${LIBNAME}: ${OBJS}
	ar rcs $@ $^
	ranlib $@

# Rule to compile source files to object files
${SRC_BUILD_DIR}/%.o: ${SRCDIR}/%.cpp
	g++ $(CXXFLAGS) -I ${INCDIR} -c $< -o $@

# Rule to compile test source file into object files
${TEST_BUILD_DIR}/%.o: ${TEST_SRCDIR}/%.cpp
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Clean rule
clean:
	rm -f ${SRC_BUILD_DIR}/*.o ${LIBNAME} ${TEST_BUILD_DIR}/*.o
	rm -f $(BIN_DIR)/*
	clear

run: test
	@for test in $(BIN_DIR)/*; do \
		./$$test; \
	done

# Make tests for internal library functions
test: $(TESTS)


att: $(TEST_BUILD_DIR)/att_test.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

add: $(TEST_BUILD_DIR)/add_test.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

layernorm: $(TEST_BUILD_DIR)/layernorm_test.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

matmul: $(TEST_BUILD_DIR)/matmul_test.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

softmax: $(TEST_BUILD_DIR)/softmax_test.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

encoder: $(TEST_BUILD_DIR)/encoder_test.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

transformerblock: $(TEST_BUILD_DIR)/transformerblock_test.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

gpt2: $(TEST_BUILD_DIR)/gpt2_test.o $(SRC_BUILD_DIR)/gpt2.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

utils: $(TEST_BUILD_DIR)/utils_test.o $(SRC_BUILD_DIR)/utils.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

datastream: $(TEST_BUILD_DIR)/datastream_test.o $(SRC_BUILD_DIR)/datastream.o
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

tokenizer: $(TEST_BUILD_DIR)/tokenizer_test.o $(SRC_BUILD_DIR)/tokenizer.o
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

interface: $(TEST_BUILD_DIR)/interface_test.o $(SRC_BUILD_DIR)/interface.o $(SRC_BUILD_DIR)/gpt2.o $(SRC_BUILD_DIR)/datastream.o $(SRC_BUILD_DIR)/tokenizer.o $(SRC_BUILD_DIR)/utils.o $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS)

train: $(TEST_BUILD_DIR)/train_test.o $(LIBNAME)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/$@ $^ $(LIBS) $(LIB_DIRS) 


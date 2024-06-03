# Compiler (g++ or clang++ both work)
CC = g++

# Compiler flags
CXXFLAGS = -std=c++17 -framework Accelerate -DACCELERATE_NEW_LAPACK #-Wall

# address sanitizer
ADDRESS_SANITIZER = -fsanitize=address -fno-omit-frame-pointer

ifeq ($(SANITIZE), 1)
  CXXFLAGS += $(ADDRESS_SANITIZER)
endif

# Library directories
LIB_DIRS = -L/opt/homebrew/lib

# Libraries to link against
LIBS = -lgtest -lgtest_main -pthread

# Include dirs
INCLUDE_DIRS = -I./StaticGrad/include/ -I./test/include/ -I/opt/homebrew/include/

# Binary directory
BIN_DIR = test/bin

# Source files
SOURCES = StaticGrad/src/classes.cpp test/src/test_common.cpp

.PHONY: all clean run

all: att add layernorm matmul softmax encoder transformerblock gpt2 utils datastream

att: test/src/att_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/att_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

add: test/src/add_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/add_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

layernorm: test/src/layernorm_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/layernorm_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

matmul: test/src/matmul_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/matmul_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

softmax: test/src/softmax_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/softmax_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

encoder: test/src/encoder_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/encoder_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

transformerblock: test/src/transformerblock_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/transformerblock_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

gpt2: test/src/gpt2_test.cpp StaticGrad/src/gpt2.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/gpt2_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

utils: test/src/utils_test.cpp StaticGrad/src/utils.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/utils_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

datastream: test/src/datastream_test.cpp StaticGrad/src/datastream.cpp
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/datastream_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)


clean:
	rm -f $(BIN_DIR)/*
	clear

run: all
	@for test in $(BIN_DIR)/*_test; do \
		./$$test; \
	done

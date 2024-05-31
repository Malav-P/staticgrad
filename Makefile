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
INCLUDE_DIRS = -I./ -I/opt/homebrew/include/

# Binary directory
BIN_DIR = bin

# Source files
SOURCES = src/classes.cpp

.PHONY: all clean run

all: att add layernorm matmul softmax encoder transformerblock gpt2 utils

att: test/att_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/att_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

add: test/add_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/add_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

layernorm: test/layernorm_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/layernorm_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

matmul: test/matmul_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/matmul_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

softmax: test/softmax_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/softmax_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

encoder: test/encoder_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/encoder_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

transformerblock: test/transformerblock_test.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/transformerblock_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

gpt2: test/gpt2_test.cpp src/gpt2.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/gpt2_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)

utils: test/utils_test.cpp src/utils.cpp $(SOURCES)
	$(CC) $(CXXFLAGS) -o $(BIN_DIR)/utils_test $^ $(INCLUDE_DIRS) $(LIBS) $(LIB_DIRS)


clean:
	rm -f $(BIN_DIR)/*
	clear

run: all
	./$(BIN_DIR)/att_test
	./$(BIN_DIR)/add_test
	./$(BIN_DIR)/layernorm_test
	./$(BIN_DIR)/matmul_test
	./$(BIN_DIR)/softmax_test
	./$(BIN_DIR)/encoder_test
	./$(BIN_DIR)/transformerblock_test
	./$(BIN_DIR)/gpt2_test
	./$(BIN_DIR)/utils_test

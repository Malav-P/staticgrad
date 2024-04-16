.PHONY: all clean
all: matmul columnadd transmatmul softmax

matmul:
	g++ -std=c++17 -o bin/testmatmul test/test.cpp src/classes.cpp -framework Accelerate -DACCELERATE_NEW_LAPACK -I./

columnadd:
	g++ -std=c++17 -o bin/testcolumnadd test/testcolumnadd.cpp src/classes.cpp -framework Accelerate -DACCELERATE_NEW_LAPACK -I./

transmatmul:
	g++ -std=c++17 -o bin/testransmatmul test/testransmatmul.cpp src/classes.cpp -framework Accelerate -DACCELERATE_NEW_LAPACK -I./

softmax:
	g++ -std=c++17 -o bin/testsoftmax test/testsoftmax.cpp src/classes.cpp -framework Accelerate -DACCELERATE_NEW_LAPACK -I./


clean:
	rm -f bin/*
	clear

run: all
	./bin/testmatmul
	./bin/testcolumnadd
	./bin/testransmatmul
	./bin/testsoftmax


leaks: all
	leaks --atExit -- ./bin/testmatmul
	leaks --atExit -- ./bin/testcolumnadd
	leaks --atExit -- ./bin/testransmatmul
	leaks --atExit -- ./bin/testsoftmax

#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

#include <vector>
#include <cstring>
#include <cmath>

using namespace std;
class Node;

void fillArrayWithRandom(float* arr, int size);
void fillArrayWithOnes(float* arr, int size);

void setup_node(Node* node, std::vector<size_t> shape_);
void teardown_node(Node* node);




#endif // TEST_COMMON_HPP
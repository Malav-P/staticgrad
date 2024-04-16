#include <iostream>
#include <memory>
#include "src/classes.hpp"

using namespace std;

int main() {

        vector<size_t> va{3,2};

        Node* A = new Node(va);
        Node* B = new Node(va);
        Operation* op = new SoftMax(B,A);



        Tensor* A_data = A->data.get();
        A_data->values[0] = 1.0;
        A_data->values[1] = 7.0;
        A_data->values[2] = 3.0;
        A_data->values[3] = -2.0;
        A_data->values[4] = 5.0;
        A_data->values[5] = 1.0;

        Tensor* B_data = B->data.get();



        op->forward();

        cout << A_data->values[0]<< ", " << A_data->values[1] << "\n" << A_data->values[2] << ", " << A_data->values[3] << "\n" << A_data->values[4] << ", " << A_data->values[5]  << "\n";
        cout << B_data->values[0]<< ", " << B_data->values[1] << "\n" << B_data->values[2] << ", " << B_data->values[3] << "\n" << B_data->values[4] << ", " << B_data->values[5]  << "\n";


        delete A;
        delete B;
        delete op;


        return 0;
    
    
}

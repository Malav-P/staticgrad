#include <iostream>
#include <memory>
#include "src/classes.hpp"

using namespace std;

int main() {

        vector<size_t> va{3,2};
        vector<size_t> vb{3,1};

        Node* A = new Node(va);
        Node* B = new Node(vb);

        Node* C = new Node(va);

        Operation* op = new ColumnAdd(C,A,B);



        Tensor* A_data = A->data.get();
        A_data->values[0] = 1.0f;
        A_data->values[1] = 2.0f;
        A_data->values[2] = 3.0f;
        A_data->values[3] = 4.0f;
        A_data->values[4] = 5.0f;
        A_data->values[5] = 6.0f;

        Tensor* B_data = B->data.get();
        B_data->values[0] = 1.0f;
        B_data->values[1] = 2.0f;
        B_data->values[2] = 3.0f;

        Tensor* C_data = C->data.get();
        C->data->grad[0] = 1.0f;     
        C->data->grad[1] = 1.0f;       
        C->data->grad[2] = 1.0f;       
        C->data->grad[3] = 1.0f;       
        C->data->grad[4] = 1.0f;       
        C->data->grad[5] = 1.0f;       


        op->forward();
        op->backward();

        cout << A_data->grad[0]<< ", " << A_data->grad[1] << "\n" << A_data->grad[2] << ", " << A_data->grad[3] << "\n" << A_data->grad[4] << ", " << A_data->grad[5]  << "\n\n";
        cout << B_data->grad[0]<< ", " << B_data->grad[1] << ", " << B_data->grad[2] << "\n\n";
        cout << C_data->values[0]<< ", " << C_data->values[1]<< "\n" << C_data->values[2] << ", " << C_data->values[3]<< "\n" << C_data->values[4]<< ", " << C_data->values[5] << "\n";

        delete A;
        delete B;
        delete C;
        delete op;

        return 0;
    
    
}

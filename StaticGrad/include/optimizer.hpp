#include "node.hpp"

class Optimizer {
    public :
    size_t num_params;
    float* params;
    float* grad;
    float lr;
    
    Optimizer(size_t num_params_, float* params_, float* grad_, float lr_ = 0.0001):
        num_params(num_params_),
        params(params_),
        grad(grad_),
        lr(lr_) {}



        virtual ~Optimizer() {}

        virtual void update() = 0;
        virtual void reset() = 0;

        void zero_grad(){
            memset(grad, 0, num_params * sizeof(float));
        }
};

class Adam : public Optimizer {
    public :
        Adam(size_t num_params_, float* params_, float* grad_, float lr_=0.0001, float beta1_ = 0.9 , float beta2_ =0.999):
        Optimizer(num_params_, params_, grad_, lr_),
        beta1(beta1_),
        beta2(beta2_),
        t(1)
        {
            m = new float[num_params]();
            v = new float[num_params]();
        }

        float* m;
        float* v;
        float beta1;
        float beta2;
        int t;

        ~Adam(){
            delete[] m;
            delete[] v;
        }

        void update(){
            for (size_t i = 0; i < num_params; i++){
                float g_ = grad[i];
        
                m[i] = beta1 * m[i] + (1-beta1)*g_;
                v[i] = beta2 * v[i] + (1-beta2)*g_*g_;
        
                float mhat = m[i] / (1 - powf(beta1, t));
                float vhat = v[i] / (1 - powf(beta2, t));
        
        
                params[i] -= lr * mhat / (sqrtf(vhat) + 1e-8);
        
            }
            t+=1;
        }

        void reset(){
            t = 1;
            memset(v, 0, num_params * sizeof(float));
            memset(m, 0, num_params * sizeof(float));
        }
};

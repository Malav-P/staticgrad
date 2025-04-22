#include "node.hpp"

/**
 * 
 * @brief Base class for all Optimizers
 *
 */
class Optimizer {
    public :
        size_t num_params;
        void** params_ptr;
        void** grad_ptr;
        float lr;
        
        /**
         * 
         * @brief Base class constructor
         * 
         * @param num_params_ the number of parameters this optimizer is responsible for
         * @param params_ address of pointer to contiguos array of parameters
         * @param grad_ address of pointer to contiguous array of gradients of parameters
         * @param lr_ learning rate
         *
         */
        Optimizer(size_t num_params_, void** params_, void** grad_, float lr_ = 0.0001):
            num_params(num_params_),
            params_ptr(params_),
            grad_ptr(grad_),
            lr(lr_) {}

        /**
         * 
         * @brief Base class destructor
         * 
         */
        virtual ~Optimizer() {}

        /**
         * 
         * @brief Update method
         * 
         * @note responsible for taking a gradient step
         * 
         */
        virtual void update() = 0;

        /**
         * 
         * @brief Reset method
         * 
         * @note responsible for resetting first and second moments, time step, etc back to initial values
         * 
         */
        virtual void reset() = 0;

        /**
         * 
         * @brief Zero gradients of the parameters
         * 
         * 
         */
        void zero_grad(){
            memset(*grad_ptr, 0, num_params * sizeof(float));
        }
};

/**
 * 
 * @brief Class implementing vanilla Adam optimizer
 *
 */
class Adam : public Optimizer {
    public :

        float* m; ///< second moments
        float* v; ///< first moments
        float beta1;
        float beta2;
        int t;

        /**
         * 
         * @brief Adam class constructor
         * 
         * @param num_params_ the number of parameters this optimizer is responsible for
         * @param params_ address of pointer to contiguos array of parameters
         * @param grad_ address of pointer to contiguous array of gradients of parameters
         * @param lr_ learning rate
         * @param beta1_ first moment hyperparameter
         * @param beta2_ second moment hyperparameter
         *
         */
        Adam(size_t num_params_, void** params_, void** grad_, float lr_=0.0001, float beta1_ = 0.9 , float beta2_ =0.999):
        Optimizer(num_params_, params_, grad_, lr_),
        beta1(beta1_),
        beta2(beta2_),
        t(1)
        {
            m = new float[num_params]();
            v = new float[num_params]();
        }

        /**
         * 
         * @brief Adam destructor, deallocates memory for first and second moments
         * 
         */
        ~Adam(){
            delete[] m;
            delete[] v;
        }

        /**
         * 
         * @brief Update method
         * 
         * @note responsible for taking a gradient step using Adam update rule
         * 
         */
        void update(){
            float* g = (float*)(*grad_ptr);
            float* p = (float*)(*params_ptr);
            for (size_t i = 0; i < num_params; i++){
                float g_ = g[i];
        
                m[i] = beta1 * m[i] + (1-beta1)*g_;
                v[i] = beta2 * v[i] + (1-beta2)*g_*g_;
        
                float mhat = m[i] / (1 - powf(beta1, t));
                float vhat = v[i] / (1 - powf(beta2, t));
        
        
                p[i] -= lr * mhat / (sqrtf(vhat) + 1e-8);
        
            }
            t+=1;
        }

        /**
         * 
         * @brief Reset method
         * 
         * @note responsible for resetting first and second moments, time step, back to initial values
         * 
         */
        void reset(){
            t = 1;
            memset(v, 0, num_params * sizeof(float));
            memset(m, 0, num_params * sizeof(float));
        }
};

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "macro.h"


float MaxAbsClamp(float* input, int size){
    // Calculate the max absolute value of an array
    // Clamp the value to a small number if output is 0
    float max = fabs(input[0]);
    float a;

    for(int i = 1; i < size; i++){
        a = fabs(input[i]);
        if(a > max) max = a;
    }
    if(max == 0) max = 1e-5;
    return max; 
}

void SimpleRMSNorm(float* layer_norm, float* input){
    float x;
    float x_norm = 0;
    for(int i = 0; i < BATCH_SIZE*SEQ_LEN; i++){
        for(int i_emb = 0; i_emb < EMB_LEN; i_emb++){
            x = input[i_emb + i*EMB_LEN];
            x_norm = x_norm + x*x;
        }
        x_norm = sqrtf(x_norm * EMB_LEN);
        for(int i_emb = 0; i_emb < EMB_LEN; i_emb++){
            x = input[i_emb + i*EMB_LEN];
            x = x/x_norm;
            layer_norm[i_emb + i*EMB_LEN] = x;
        }
        x_norm = 0;
    }
}

void rmsnorm(float* out, float* x){
    float x_norm = 0;
    for(int i = 0; i < EMB_LEN; i++){
        x_norm += x[i]*x[i];
    }
    x_norm = sqrtf(x_norm*EMB_LEN);
    for(int i = 0; i < EMB_LEN; i++){
        out[i] = x[i]/x_norm;
    }
}


void q_vmmul(float* xout, float* x, const int32_t* w, const float w_scale, int linear_in, int linear_out){
    int32_t weight;
    float wx;
    int idx_w;
    int b;
    for(int w_col = 0; w_col < linear_out; w_col ++){
        wx = 0;
        for(int x_col = 0; x_col < linear_in; x_col ++){
            idx_w = 2*(x_col + w_col*linear_in) / MACHINE_WIDTH;
            b = 2*(x_col + w_col*linear_in) % MACHINE_WIDTH;
            weight = (w[idx_w] << b) >> (MACHINE_WIDTH-2);
            wx += (float) weight * x[x_col];
        }
        xout[w_col] = wx * w_scale;
    }
}

void q_vmmul_bin(float* xout, float* x, const int32_t* w, const float w_scale, int linear_in, int linear_out){
    int32_t weight;
    float wx;
    int idx_w;
    int b;
    for(int w_col = 0; w_col < linear_out; w_col ++){
        wx = 0;
        for(int x_col = 0; x_col < linear_in; x_col ++){
            idx_w = (x_col + w_col*linear_in) / MACHINE_WIDTH;
            b = (x_col + w_col*linear_in) % MACHINE_WIDTH;
            weight = w[idx_w] << b;
            if(weight < 0){
                wx -= x[x_col];
            }else{
                wx += x[x_col];
            }
        }
        xout[w_col] = wx * w_scale;
    }
}

void softmax(float* x, int size){
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// RoPE relative positional encoding: complex-valued rotate q and k in each head
void RoPE(float* Q, float* K, int pos){
    int head_dim;
    float freq, val, fcr, fci ,q0, q1, k0, k1;

    //Encode Q and K for the first KV_DIM elements
    for(int i = 0; i < KV_DIM; i+=2){
        head_dim = i % HEAD_SIZE;
        freq = 1.0f / powf(10000.0f, head_dim / (float) HEAD_SIZE);
        val = pos * freq;
        fcr = cosf(val);
        fci = sinf(val);
        q0 = Q[i];
        q1 = Q[i+1];
        Q[i] = q0 * fcr - q1 * fci;
        Q[i+1] = q0 * fci + q1 * fcr;
        k0 = K[i];
        k1 = K[i+1];
        K[i] = k0 * fcr - k1 * fci;
        K[i+1] = k0 * fci + k1 * fcr;
    }

    //Encode the rest of elements in Q
    for(int i = KV_DIM; i < EMB_LEN; i+=2){
        head_dim = i % HEAD_SIZE;
        freq = 1.0f / powf(10000.0f, head_dim / (float) HEAD_SIZE);
        val = pos * freq;
        fcr = cosf(val);
        fci = sinf(val);
        q0 = Q[i];
        q1 = Q[i+1];
        Q[i] = q0 * fcr - q1 * fci;
        Q[i+1] = q0 * fci + q1 * fcr;
    }

}

void GQA(float* xb, float* score, float* Q, float* k_cache, float* v_cache, int pos, int layer_off){
    for(int h = 0; h < N_HEAD; h++){
        // compute dot prodcut: Q*K'
        float* q = Q + h*HEAD_SIZE;
        // iterate over all timesteps, including the current one.
        for(int t = 0; t <= pos; t++){
            float* k = k_cache + layer_off + t*KV_DIM + (h/KV_MUL)*HEAD_SIZE;
            float dpt = 0;
            for (int i = 0; i < HEAD_SIZE; i++){
                dpt += q[i] * k[i];
            }
            dpt /= ROOT_SQUARE_HEAD_SIZE;
            score[t] = dpt;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(score, pos + 1);
        
        float* attn = xb + h*HEAD_SIZE;
        memset(attn, 0, HEAD_SIZE * sizeof(float));
        // weighted sum of the values
        for(int t = 0; t <= pos; t ++){
            float* v = v_cache + layer_off + t*KV_DIM + (h/KV_MUL)*HEAD_SIZE;
            for(int i = 0; i < HEAD_SIZE; i++){
                attn[i] += score[t] * v[i];
            }
        }
    }
}

void SwiGLU(float* ffn1, float* ffn2){
    float val;
    for(int i = 0; i < HIDDEN_DIM; i++){
        val = ffn1[i];
        val *= (1.0f / (1.0f +expf(-val)));
        val *= ffn2[i];
        ffn1[i] = val;
    }
}

int itos(int num,char *str)//radix-10
{
    int i = 0;
    int len;
    if(num<0)
    {
        num = -num;
        str[i++] = '-';
    } 

    do
    {
        str[i++] = num%10+48; 
        num /= 10;
    }while(num);
    str[i++] = '\r';
    str[i++] = '\n';
    str[i++] = '\0';
    len = i;
    i = i - 3;
    int j = 0;
    if(str[0]=='-')
    {
        j = 1;
        ++i;
    }

    for(;j<i/2;j++)
    {
        str[j] = str[j] + str[i-1-j];
        str[i-1-j] = str[j] - str[i-1-j];
        str[j] = str[j] - str[i-1-j];
    } 
    return len;
}

// void WeightMUL_M2(float* input, const unsigned int* bl_w1, const int* bl_w2, float* scale_list, const float w_scale, float* layer_mult){
//     unsigned int w1;
//     int w2;
//     unsigned int *p_layer_quant_base = (unsigned int*) input;
//     unsigned int *p_layer_quant = p_layer_quant_base;
//     unsigned int wx;
//     float f_wx;
//     float *p_wx = (float*) &wx;
//     int index_w;
//     int b;
//     for(int x_row = 0; x_row < BATCH_SIZE*SEQ_LEN; x_row++){
//         p_layer_quant = p_layer_quant_base + x_row*EMB_LEN;
//         for(int w_col = 0; w_col < LINEAR_OUT_COL; w_col ++){
//             f_wx = 0;        
//             for(int x_col = 0; x_col < EMB_LEN; x_col ++){
//                 index_w = (x_col + w_col*EMB_LEN)/MACHINE_WIDTH;
//                 b = (x_col + w_col*EMB_LEN)%MACHINE_WIDTH;
//                 w2 = (bl_w2[index_w] << b) >> (MACHINE_WIDTH-1);
//                 switch (w2)
//                 {
//                 case 0:
//                     break;            
//                 default:
//                     w1 = (bl_w1[index_w] >> b) << (MACHINE_WIDTH-1);                
//                     wx = (w1 ^ (*p_layer_quant));
//                     f_wx = f_wx + (*p_wx);
//                     break;
//                 }
//                 p_layer_quant = p_layer_quant + 1;
//             }
//             layer_mult[w_col + x_row*LINEAR_OUT_COL] = f_wx*w_scale/scale_list[x_row];
//             p_layer_quant = p_layer_quant - EMB_LEN;
//         }
//     }
// }

// void WeightMUL_M3(float* input, unsigned int* bl_w1, int* bl_w2, float* scale_list, float w_scale, float* layer_mult){
//     unsigned int w1;
//     int w2;
//     unsigned int *p_layer_quant_base = (unsigned int*) input;
//     unsigned int *p_layer_quant = p_layer_quant_base;
//     unsigned int wx;
//     float f_wx;
//     float *p_wx = (float*) &wx;
//     int index_w;
//     int b;
//     for(int x_row = 0; x_row < BATCH_SIZE*SEQ_LEN; x_row++){
//         p_layer_quant = p_layer_quant_base + x_row*EMB_LEN;
//         for(int w_col = 0; w_col < LINEAR_OUT_COL; w_col ++){
//             f_wx = 0;        
//             for(int x_col = 0; x_col < EMB_LEN; x_col ++){
//                 index_w = (x_col + w_col*EMB_LEN)/MACHINE_WIDTH;
//                 b = (x_col + w_col*EMB_LEN)%MACHINE_WIDTH;
//                 w1 = (bl_w1[index_w] >> b) << (MACHINE_WIDTH-1);
//                 w2 = (bl_w2[index_w] << b) >> (MACHINE_WIDTH-1);
//                 wx = (w1 ^ (*p_layer_quant)) & w2;
//                 f_wx = f_wx + (*p_wx);
//                 //printf("%dth -> w: 0x%X; x: 0x%X; wx: %X; fwx: %f\n",x_col,w,(*p_layer_quant),wx,f_wx);
//                 p_layer_quant = p_layer_quant + 1;
//             }
//             layer_mult[w_col + x_row*LINEAR_OUT_COL] = f_wx*w_scale/scale_list[x_row];
//             p_layer_quant = p_layer_quant - EMB_LEN;
//         }
//     }
// }

// void ActQuant(float* input, float* layer_quant, float* scale_list){
//     float *p_layer_norm = input;
//     float x_quant;
//     float x_scale;
//     for(int i = 0; i < SEQ_LEN; i++){
//         //printf("p_layer_norm: %p\n", p_layer_norm);
//         //printf("maxabs: %f\n", MaxAbsClamp(p_layer_norm, EMB_LEN));
//         x_scale = Q_UP_BOUND/MaxAbsClamp(p_layer_norm, EMB_LEN);
//         scale_list[i] = x_scale;
//         for(int i_emb = 0; i_emb < EMB_LEN; i_emb ++){
//             x_quant = round(input[i_emb + i*EMB_LEN]*x_scale);
//             x_quant = (x_quant > Q_UP_BOUND) ? Q_UP_BOUND : x_quant;
//             x_quant = (x_quant < Q_LOW_BOUND) ? Q_LOW_BOUND : x_quant;
//             layer_quant[i_emb + i*EMB_LEN] = x_quant;
//         }
//         p_layer_norm = p_layer_norm + EMB_LEN;
//     }
// }
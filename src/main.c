//BitLinear Implementation Version 1.0
//Author: Jiawen Qi
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "macro.h"

extern const int32_t wq[N_LAYER*N_QUERY_WEIGHT];
extern const int32_t wk[N_LAYER*N_KV_WEIGHT];
extern const int32_t wv[N_LAYER*N_KV_WEIGHT];
extern const int32_t wo[N_LAYER*N_QUERY_WEIGHT];
extern const int32_t w1[N_LAYER*N_FFN_WEIGHT];
extern const int32_t w2[N_LAYER*N_FFN_WEIGHT];
extern const int32_t w3[N_LAYER*N_FFN_WEIGHT];
extern const float wq_scale[N_LAYER];
extern const float wk_scale[N_LAYER];
extern const float wv_scale[N_LAYER];
extern const float wo_scale[N_LAYER];
extern const float w1_scale[N_LAYER];
extern const float w2_scale[N_LAYER];
extern const float w3_scale[N_LAYER];

extern float input[SEQ_LEN*EMB_LEN];
extern const float ref[EMB_LEN];

//extern void SimpleRMSNorm(float* layer_norm, float* input);
//extern void ActQuant(float* input, float* layer_quant, float* scale_list);
//extern void WeightMUL_M2(float* input, const unsigned int* bl_w1, const int* bl_w2, float* scale_list, const float w_scale, float* layer_mult);
//extern void WeightMUL_M3(float* input, const unsigned int* bl_w1, const int* bl_w2, float* scale_list, const float w_scale, float* layer_mult);
extern void softmax(float* x, int size);
extern void GQA(float* xb, float* score, float* Q, float* k_cache, float* v_cache, int pos, int layer_off);
extern void q_vmmul(float* xout, float* x, const int32_t* w, const float w_scale,  int linear_in, int linear_out);
extern void q_vmmul_bin(float* xout, float* x, const int32_t* w, const float w_scale,  int linear_in, int linear_out);
extern float MaxAbsClamp(float* input, int size);
extern void rmsnorm(float* out, float* x);
extern void RoPE(float* Q, float* K, int pos);
extern void SwiGLU(float* ffn1, float* ffn2);
extern uint64_t itos(int num, char* str);

int main(void)
{
    float x[EMB_LEN];
    float xb[EMB_LEN];
    float xb2[EMB_LEN];
    float hb[HIDDEN_DIM];
    float hb2[HIDDEN_DIM];
    float q[EMB_LEN];
    float k[KV_DIM];
    float v[KV_DIM];
    float score[SEQ_LEN];
    float key_cache[N_LAYER*SEQ_LEN*KV_DIM];
    float value_cache[N_LAYER*SEQ_LEN*KV_DIM];

    int loff;
    const int32_t* p_wq;
    const int32_t* p_wk;
    const int32_t* p_wv;
    const int32_t* p_wo;
    const int32_t* p_w1;
    const int32_t* p_w2;
    const int32_t* p_w3;


    for(int pos = 0; pos < SEQ_LEN; pos++){
        float* p_in = input + pos * EMB_LEN;
        for(int l = 0; l < N_LAYER; l ++){
            p_wq = wq + l*N_QUERY_WEIGHT;
            p_wk = wk + l*N_KV_WEIGHT;
            p_wv = wv + l*N_KV_WEIGHT;
            p_wo = wo + l*N_QUERY_WEIGHT;
            p_w1 = w1 + l*N_FFN_WEIGHT;
            p_w2 = w2 + l*N_FFN_WEIGHT;
            p_w3 = w3 + l*N_FFN_WEIGHT;
            loff = l * SEQ_LEN * KV_DIM;

            // attention rmsnorm
            rmsnorm(xb, p_in);

            #if TERNARY
            q_vmmul(q, xb, p_wq, wq_scale[l], EMB_LEN, EMB_LEN);
            q_vmmul(k, xb, p_wk, wk_scale[l], EMB_LEN, KV_DIM);
            q_vmmul(v, xb, p_wv, wv_scale[l], EMB_LEN, KV_DIM);
            #else
            q_vmmul_bin(q, xb, p_wq, wq_scale[l], EMB_LEN, EMB_LEN);
            q_vmmul_bin(k, xb, p_wk, wk_scale[l], EMB_LEN, KV_DIM);
            q_vmmul_bin(v, xb, p_wv, wv_scale[l], EMB_LEN, KV_DIM);
            #endif
            // for(int i = 0; i < EMB_LEN; i++) printf("%f, ", q[i]);
            // printf("\n");

            RoPE(q, k, pos);
            float* key_cache_row = key_cache + loff + pos * KV_DIM;
            float* value_cache_row = value_cache + loff + pos * KV_DIM;
            memcpy(key_cache_row, k, KV_DIM * sizeof(*key_cache_row));
            memcpy(value_cache_row, v, KV_DIM * sizeof(*value_cache_row));
            GQA(xb, score, q, key_cache, value_cache, pos, loff);
            
            #if TERNARY
            q_vmmul(xb2, xb, p_wo, wo_scale[l], EMB_LEN, EMB_LEN);
            #else
            q_vmmul_bin(xb2, xb, p_wo, wo_scale[l], EMB_LEN, EMB_LEN);
            #endif
            // residual connection
            for (int i = 0; i < EMB_LEN; i++){
                xb[i] = xb2[i] + p_in[i];
            }

            // ffn rmsnorm
            rmsnorm(xb2, xb);
            #if TERNARY
            q_vmmul(hb, xb2, p_w1, w1_scale[l], EMB_LEN, HIDDEN_DIM);
            q_vmmul(hb2, xb2, p_w2, w2_scale[l], EMB_LEN, HIDDEN_DIM);
            #else
            q_vmmul_bin(hb, xb2, p_w1, w1_scale[l], EMB_LEN, HIDDEN_DIM);
            q_vmmul_bin(hb2, xb2, p_w2, w2_scale[l], EMB_LEN, HIDDEN_DIM);
            #endif
            SwiGLU(hb, hb2);
            // for(int i = 0; i < HIDDEN_DIM; i++) printf("%f, ", s->hb[i]);
            // printf("\n");
            #if TERNARY
            q_vmmul(xb2, hb, p_w3, w3_scale[l], HIDDEN_DIM, EMB_LEN);
            #else
            q_vmmul_bin(xb2, hb, p_w3, w3_scale[l], HIDDEN_DIM, EMB_LEN);
            #endif
            // for(int i = 0; i < EMB_LEN; i++) printf("%f, ", s->xb[i]);
            // printf("\n");
            // residual conncection
            for (int i = 0; i < EMB_LEN; i++){
                x[i] = xb[i] + xb2[i];
            }
            p_in = x;
        }
    }

    float diff;
    float error = 0;
    for(int i = 0; i < EMB_LEN; i++){
        diff = x[i] - ref[i];
        error = error + diff*diff;
    }
    //printf("Squared Error of Attention: %f\n", error);

    char str[16];
    int len = itos((int) error, str);
    //printf("num to string %s, len: %d\n",str,len);

    return 0;
}

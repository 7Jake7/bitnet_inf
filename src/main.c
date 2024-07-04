//BitLinear Implementation Version 1.0
//Author: Jiawen Qi
#include <stdio.h>
#include <math.h>
#include "macro.h"

extern unsigned int layer_bn_w[NUM_BL_WEIGHT]; 
extern float input[BATCH_SIZE*SEQ_LEN*EMB_LEN];

extern float MaxAbsClamp(float* input, int size);

int main(int argc, char const *argv[])
{
    printf("Hello World!\n");
    float x;
    float x_norm = 0;
    const float w_scale = 0.4703;
    int quant_bit = 8;  // Bit width of input quantization.
    float x_scale;

    // Layer Normalization (SimpleRMSNorm)
    float layer_norm[BATCH_SIZE*SEQ_LEN*EMB_LEN] = {0};
    
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

    
    printf("After layer normalization:[");
    for(int i = 0; i < BATCH_SIZE*SEQ_LEN; i++){
        for(int i_emb = 0; i_emb < EMB_LEN; i_emb++){
            x = layer_norm[i_emb + i*EMB_LEN];
            printf("%f, ",x);
        }
    }
    printf("]\n");    


    // Input Quantization
    float layer_quant[BATCH_SIZE*SEQ_LEN*EMB_LEN] = {0};
    float scale_list[BATCH_SIZE*SEQ_LEN];
    float *p_layer_norm = layer_norm;
    float x_quant;
    for(int i = 0; i < BATCH_SIZE*SEQ_LEN; i++){
        p_layer_norm = p_layer_norm + i*EMB_LEN;
        printf("p_layer_norm: %p\n", p_layer_norm);
        printf("maxabs: %f\n", MaxAbsClamp(p_layer_norm, EMB_LEN));
        x_scale = ((1 << (quant_bit - 1)) - 1)/MaxAbsClamp(p_layer_norm, EMB_LEN);
        scale_list[i] = x_scale;
        for(int i_emb = 0; i_emb < EMB_LEN; i_emb ++){
            x_quant = round(layer_norm[i_emb + i*EMB_LEN]*x_scale);
            x_quant = (x_quant > 127) ? 127 : x_quant;
            x_quant = (x_quant < -128) ? -128 : x_quant;
            layer_quant[i_emb + i*EMB_LEN] = x_quant;
        }
    }

    printf("After layer quantization:[");
    for(int i = 0; i < BATCH_SIZE*SEQ_LEN; i++){
        for(int i_emb = 0; i_emb < EMB_LEN; i_emb++){
            x = layer_quant[i_emb + i*EMB_LEN];
            printf("%f, ",x);
        }
    }
    printf("]\n");
    
    // Weight Decompression and Multiplication
    
    unsigned int w;
    unsigned int *p_layer_quant_base = (unsigned int*) layer_quant;
    unsigned int *p_layer_quant = p_layer_quant_base;
    unsigned int wx;
    float f_wx;
    float layer_mult[BATCH_SIZE*SEQ_LEN*EMB_LEN];
    float *p_wx = (float*) &wx;
    int index_w;
    int b;
    
    printf("layer_quant[0]: 0x%X, p_layer_quant: 0x%X\n",layer_quant[0],*p_layer_quant);
    printf("layer_quant[1]: 0x%X, p_layer_quant: 0x%X\n",layer_quant[1],*(p_layer_quant + 1));
    
    
    for(int x_row = 0; x_row < BATCH_SIZE*SEQ_LEN; x_row++){
        p_layer_quant = p_layer_quant_base + x_row*EMB_LEN;
        for(int w_col = 0; w_col < EMB_LEN; w_col ++){
            f_wx = 0;        
            for(int x_col = 0; x_col < EMB_LEN; x_col ++){
                index_w = (x_col + w_col*EMB_LEN)/MACHINE_WIDTH;
                b = (x_col + w_col*EMB_LEN)%MACHINE_WIDTH;
                w = (layer_bn_w[index_w] >> (MACHINE_WIDTH-1-b)) << (MACHINE_WIDTH-1);
                wx = w ^ (*p_layer_quant);
                f_wx = f_wx + (*p_wx);
                //printf("%dth -> w: 0x%X; x: 0x%X; wx: %X; fwx: %f\n",x_col,w,(*p_layer_quant),wx,f_wx);
                p_layer_quant = p_layer_quant + 1;
            }
            layer_mult[w_col + x_row*EMB_LEN] = f_wx*w_scale/scale_list[x_row];
            p_layer_quant = p_layer_quant - EMB_LEN;
        }
    }
    
    printf("After Matrix Multiplication:[");
    for(int i = 0; i < BATCH_SIZE*SEQ_LEN; i++){
        for(int i_emb = 0; i_emb < EMB_LEN; i_emb++){
            x = layer_mult[i_emb + i*EMB_LEN];
            printf("%f, ",x);
        }
    }
    printf("]\n");
    
    return 0;
}

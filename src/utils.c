#include <stdio.h>
#include <math.h>
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
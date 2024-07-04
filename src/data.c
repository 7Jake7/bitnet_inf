//Prestore compressed weights of the model.
//Now only for one BitLinear layer
#include "macro.h"

const unsigned int layer_bn_w[NUM_BL_WEIGHT] = {951988894, 4166812018};
//float input[BATCH_SIZE*SEQ_LEN*EMB_LEN] = {0.6977, 0.8000, 0.1610, 0.2823, 0.6816, 0.9152, 0.3971, 0.8742};
float input[BATCH_SIZE*SEQ_LEN*EMB_LEN] = {0.7745, 0.4369, 0.5191, 0.6159, 0.8102, 0.9801, 0.1147, 0.3168, 0.6965, 0.9143, 0.9351, 0.9412, 0.5995, 0.0652, 0.5460, 0.1872};
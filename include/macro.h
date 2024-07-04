#ifndef __MODEL__
#define __MODEL__

#define NUM_BL_WEIGHT 2  // from len() of the compressed weight vector
#define MACHINE_WIDTH 32
#define BATCH_SIZE 1
#define SEQ_LEN 2
#define EMB_LEN_BASE 3
#define EMB_LEN (1 << EMB_LEN_BASE) 
#endif
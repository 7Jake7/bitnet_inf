#include <stdint.h>
#include "macro.h"
const int32_t wq[N_LAYER*N_QUERY_WEIGHT] = {3190988536, 1556181571, };
const int32_t wk[N_LAYER*N_KV_WEIGHT] = {93799219, };
const int32_t wv[N_LAYER*N_KV_WEIGHT] = {2996318184, };
const int32_t wo[N_LAYER*N_QUERY_WEIGHT] = {1104123898, 3537123514, };
const int32_t w1[N_LAYER*N_FFN_WEIGHT] = {974444732, 371550136, 1672897255, 1514620960, };
const int32_t w2[N_LAYER*N_FFN_WEIGHT] = {1913424522, 3790537737, 3861166030, 2054548162, };
const int32_t w3[N_LAYER*N_FFN_WEIGHT] = {1025268456, 1707625161, 1565453099, 1558908622, };
const float wq_scale[N_LAYER] = {0.2511039078235626, };
const float wk_scale[N_LAYER] = {0.2853246033191681, };
const float wv_scale[N_LAYER] = {0.22257691621780396, };
const float wo_scale[N_LAYER] = {0.25234177708625793, };
const float w1_scale[N_LAYER] = {0.22404898703098297, };
const float w2_scale[N_LAYER] = {0.2890251576900482, };
const float w3_scale[N_LAYER] = {0.25585904717445374, };
float input[SEQ_LEN*EMB_LEN] = {-0.003743410110473633, 0.26822179555892944, -0.4115225672721863, -0.3679695129394531, -0.19257718324661255, 0.13407868146896362, -0.009906589984893799, 0.39644473791122437, };
float const ref[EMB_LEN] = {0.047524936497211456, 0.21894389390945435, -0.434444785118103, -0.4894697964191437, -0.26079893112182617, 0.08744435757398605, 0.04091312363743782, 0.34599220752716064, };

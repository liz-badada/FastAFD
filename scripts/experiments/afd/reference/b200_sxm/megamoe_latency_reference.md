# MegaMoE measured latency reference

Only exact measured points are listed. An AFD row is eligible only when it is stable and has a correctness-passing colocated MegaMoE-versus-DeepEP result at the same model, precision, system, logical source-rank batch, MTP nextN, and layer count.

The speedup columns compare complete colocated MoE-stage backend paths, including production quantization, dispatch/combine, expert alignment, scatter/gather, and GEMMs. They are not GEMM-only or end-to-end serving speedups. The AIC profile consumes the absolute MegaMoE latency.

## Model contracts

| model | hidden | expert intermediate | routed experts | routed top-k | shared experts | activation | precision |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| deepseek_v4_flash_fp4 | 4096 | 2048 | 256 | 6 | 1 | silu | w4a8_mxfp4_mxfp8 |
| deepseek_v4_pro_fp4 | 7168 | 3072 | 384 | 6 | 1 | silu | w4a8_mxfp4_mxfp8 |
| minimax_m25_fp4 | 3072 | 1536 | 256 | 8 | 0 | silu | w4a8_mxfp4_mxfp8 |
| minimax_m3_fp4 | 6144 | 3072 | 128 | 4 | 1 | swigluoai | w4a8_mxfp4_mxfp8 |
| qwen3_235b_fp4 | 4096 | 1536 | 128 | 8 | 0 | silu | w4a8_mxfp4_mxfp8 |

## Backend summary

| model | precision | system | colocated points | split points | colocated MegaMoE p50 ms | split MegaMoE p50 ms | DeepEP / MegaMoE median | conservative lower bound | max MegaMoE CV % | reference stack |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |
| deepseek_v4_flash_fp4 | w4a8_mxfp4_mxfp8 | b200_sxm | 8 | 24 | 5.7239-7.3525 | 8.4790-34.9837 | 2.6263-2.9594 | 2.5978-2.8943 | 1.6797 | DeepEP 1.2.1 / SGLang 0.5.7 |
| deepseek_v4_pro_fp4 | w4a8_mxfp4_mxfp8 | b200_sxm | 8 | 24 | 21.5855-25.7013 | 25.8363-103.1615 | 1.8239-1.9206 | 1.7421-1.8973 | 1.2515 | DeepEP 1.2.1 / SGLang 0.5.7 |
| minimax_m25_fp4 | w4a8_mxfp4_mxfp8 | b200_sxm | 8 | 24 | 5.7087-8.3483 | 7.8110-30.7495 | 2.5206-3.4755 | 2.4785-3.4498 | 1.7245 | DeepEP 1.2.1 / SGLang 0.5.7 |
| minimax_m3_fp4 | w4a8_mxfp4_mxfp8 | b200_sxm | 8 | 24 | 8.0642-12.3029 | 12.5697-49.6180 | 2.2105-2.7277 | 2.1896-2.7080 | 0.5758 | DeepEP 1.2.1 / SGLang 0.5.7 |
| qwen3_235b_fp4 | w4a8_mxfp4_mxfp8 | b200_sxm | 8 | 24 | 7.5416-13.5846 | 9.5629-35.9506 | 2.4070-3.7204 | 2.3870-3.6900 | 1.7508 | DeepEP 1.2.1 / SGLang 0.5.7 |

## Measurement environments

| system | GPU | driver / power / clocks / memory | Torch | CUDA | container | partition | jobs | nodes | source commits | source tree SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b200_sxm | NVIDIA B200 | 595.58.03, 1000.00, 1965, 3996, 183359 | 2.9.1+cu129 | 12.9 | /home/scratch.jinyanc_wwfo/enroot_images/sglang_deepseek_v4_blackwell.sqsh | b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb | 3437123, 3437432, 3437512, 3438056, 3438076 | umbriel-b200-028, umbriel-b200-040 | a48593b5180b33ab77a3b114938d9025bcf77181, cd0bd85b34a93a24f53bfd1fa814693f5ab77b88, e507eacf858d2046bdc2cca02ed86c0e58bd6c60 | 8b11c7947e5914aeecce94000c52fae7bbb76581eb67bce1bb9d9032362e570e, c860a6f67727c18fde5cf5f6895915dccf4e7ea3d6bd07a94852d23ce1ce4053, fce9cfe9888adc31974e31434f16d3f1b87585a692a9adbbfe3fb481c4a564da |

## Exact measured points

| model | stage | precision | system | topology | logical_batch | physical_batch | mtp_nextn | microbatches | layers | mega_p50_ms | reference_p50_ms | reference_stack | speedup | speedup_lower_bound | mega_cv_percent | reference_cv_percent | correctness | eligible | source_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek_v4_flash_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 48 | 0 | 1 | 43 | 5.7239 | 16.9394 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.9594 | 2.8864 | 0.3814 | 0.3830 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_flash_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 96 | 1 | 1 | 43 | 5.9325 | 17.1925 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.8980 | 2.8619 | 0.1669 | 4.0654 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_flash_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 144 | 2 | 1 | 43 | 6.1244 | 17.4640 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.8515 | 2.8197 | 0.2000 | 3.6608 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_flash_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 192 | 3 | 1 | 43 | 6.3494 | 17.6699 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.7829 | 2.7545 | 0.2280 | 2.4750 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_flash_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 96 | 0 | 1 | 43 | 5.9502 | 17.4854 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.9386 | 2.8943 | 0.2383 | 2.7183 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_flash_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 192 | 1 | 1 | 43 | 6.3750 | 17.6432 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.7676 | 2.7264 | 0.2329 | 1.2287 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_flash_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 288 | 2 | 1 | 43 | 6.7658 | 18.8668 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.7886 | 2.7553 | 0.2069 | 2.1025 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_flash_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 384 | 3 | 1 | 43 | 7.3525 | 19.3096 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.6263 | 2.5978 | 0.2241 | 2.0508 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_pro_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 48 | 0 | 1 | 61 | 21.5855 | 41.4564 | DeepEP 1.2.1 / SGLang 0.5.7 | 1.9206 | 1.8973 | 0.0598 | 1.3690 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_pro_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 96 | 1 | 1 | 61 | 22.3223 | 41.8256 | DeepEP 1.2.1 / SGLang 0.5.7 | 1.8737 | 1.8414 | 0.0728 | 1.4575 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_pro_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 144 | 2 | 1 | 61 | 23.1868 | 42.9227 | DeepEP 1.2.1 / SGLang 0.5.7 | 1.8512 | 1.8327 | 0.0744 | 1.1771 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_pro_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 192 | 3 | 1 | 61 | 23.6005 | 43.0588 | DeepEP 1.2.1 / SGLang 0.5.7 | 1.8245 | 1.8184 | 0.0770 | 1.0835 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_pro_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 96 | 0 | 1 | 61 | 22.3054 | 41.8989 | DeepEP 1.2.1 / SGLang 0.5.7 | 1.8784 | 1.8612 | 0.0851 | 1.1847 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_pro_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 192 | 1 | 1 | 61 | 23.6201 | 43.0796 | DeepEP 1.2.1 / SGLang 0.5.7 | 1.8239 | 1.8158 | 0.0642 | 1.8581 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_pro_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 288 | 2 | 1 | 61 | 24.8435 | 45.3906 | DeepEP 1.2.1 / SGLang 0.5.7 | 1.8271 | 1.7421 | 1.2515 | 1.1103 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_pro_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 384 | 3 | 1 | 61 | 25.7013 | 47.2575 | DeepEP 1.2.1 / SGLang 0.5.7 | 1.8387 | 1.8023 | 0.1561 | 1.6015 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m25_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 48 | 0 | 1 | 62 | 5.7087 | 19.8405 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.4755 | 3.4498 | 0.1555 | 2.3708 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m25_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 96 | 1 | 1 | 62 | 6.0114 | 19.9892 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.3252 | 3.2865 | 0.2151 | 3.7005 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m25_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 144 | 2 | 1 | 62 | 6.3262 | 20.3223 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.2124 | 3.1326 | 0.2659 | 3.4436 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m25_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 192 | 3 | 1 | 62 | 6.5631 | 20.4655 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.1182 | 3.0692 | 0.2116 | 2.1400 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m25_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 96 | 0 | 1 | 62 | 6.0267 | 20.4007 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.3851 | 3.3456 | 0.1725 | 2.2803 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m25_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 192 | 1 | 1 | 62 | 6.5848 | 20.2761 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.0792 | 3.0293 | 0.2287 | 1.5918 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m25_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 288 | 2 | 1 | 62 | 7.4203 | 20.7125 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.7913 | 2.7715 | 0.1509 | 0.3179 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m25_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 384 | 3 | 1 | 62 | 8.3483 | 21.0423 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.5206 | 2.4785 | 0.2087 | 1.3971 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m3_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 48 | 0 | 1 | 57 | 8.0642 | 21.9965 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.7277 | 2.7080 | 0.1611 | 0.2616 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m3_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 96 | 1 | 1 | 57 | 8.3920 | 22.4724 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.6778 | 2.6508 | 0.1661 | 1.5485 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m3_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 144 | 2 | 1 | 57 | 8.9129 | 22.9271 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.5723 | 2.5585 | 0.1337 | 0.7072 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m3_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 192 | 3 | 1 | 57 | 9.2549 | 24.1059 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.6047 | 2.5774 | 0.1800 | 1.6377 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m3_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 96 | 0 | 1 | 57 | 8.3945 | 22.4636 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.6760 | 2.6605 | 0.1335 | 1.0418 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m3_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 192 | 1 | 1 | 57 | 9.2330 | 23.6650 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.5631 | 2.5441 | 0.1611 | 1.1558 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m3_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 288 | 2 | 1 | 57 | 10.6952 | 26.1643 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.4464 | 2.4238 | 0.1478 | 1.1343 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| minimax_m3_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 384 | 3 | 1 | 57 | 12.3029 | 27.1956 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.2105 | 2.1896 | 0.1539 | 1.1997 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| qwen3_235b_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 48 | 0 | 1 | 94 | 7.5416 | 28.0580 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.7204 | 3.6900 | 0.1637 | 1.7091 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| qwen3_235b_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 96 | 1 | 1 | 94 | 8.1441 | 29.0696 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.5694 | 3.4195 | 0.4277 | 1.3823 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| qwen3_235b_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 144 | 2 | 1 | 94 | 9.1408 | 29.0945 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.1829 | 3.1392 | 0.1178 | 1.2334 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| qwen3_235b_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 48 | 192 | 3 | 1 | 94 | 9.6080 | 28.7047 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.9876 | 2.9549 | 0.1307 | 1.7022 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| qwen3_235b_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 96 | 0 | 1 | 94 | 8.1414 | 28.3529 | DeepEP 1.2.1 / SGLang 0.5.7 | 3.4825 | 3.4188 | 0.2663 | 0.6955 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| qwen3_235b_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 192 | 1 | 1 | 94 | 9.6093 | 28.4733 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.9631 | 2.9301 | 0.1716 | 1.3917 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| qwen3_235b_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 288 | 2 | 1 | 94 | 11.8007 | 32.0507 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.7160 | 2.6125 | 0.5057 | 1.4031 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| qwen3_235b_fp4 | agg | w4a8_mxfp4_mxfp8 | b200_sxm | ep8 | 96 | 384 | 3 | 1 | 94 | 13.5846 | 32.6977 | DeepEP 1.2.1 / SGLang 0.5.7 | 2.4070 | 2.3870 | 0.1136 | 2.1673 | yes | yes | a48593b5180b33ab77a3b114938d9025bcf77181 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 1 | 43 | 8.4790 | - | - | 2.9594 | 2.8864 | 0.3004 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 2 | 43 | 16.7226 | - | - | 2.9594 | 2.8864 | 0.1305 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 4 | 43 | 33.1604 | - | - | 2.9594 | 2.8864 | 0.2979 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 1 | 43 | 8.8532 | - | - | 2.8980 | 2.8619 | 0.2593 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 2 | 43 | 17.0357 | - | - | 2.8980 | 2.8619 | 1.6797 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 4 | 43 | 33.3698 | - | - | 2.8980 | 2.8619 | 0.1390 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 1 | 43 | 9.0479 | - | - | 2.8515 | 2.8197 | 0.2307 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 2 | 43 | 17.4581 | - | - | 2.8515 | 2.8197 | 0.1405 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 4 | 43 | 33.9651 | - | - | 2.8515 | 2.8197 | 0.0500 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 1 | 43 | 9.2480 | - | - | 2.7829 | 2.7545 | 0.1748 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 2 | 43 | 17.5525 | - | - | 2.7829 | 2.7545 | 0.2976 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 4 | 43 | 34.0436 | - | - | 2.7829 | 2.7545 | 0.0719 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 1 | 43 | 8.8042 | - | - | 2.9386 | 2.8943 | 0.8157 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 2 | 43 | 17.0645 | - | - | 2.9386 | 2.8943 | 0.1748 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 4 | 43 | 33.3812 | - | - | 2.9386 | 2.8943 | 0.1918 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 1 | 43 | 9.2761 | - | - | 2.7676 | 2.7264 | 0.3756 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 2 | 43 | 17.5692 | - | - | 2.7676 | 2.7264 | 0.2154 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 4 | 43 | 34.0487 | - | - | 2.7676 | 2.7264 | 0.1704 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 1 | 43 | 9.7244 | - | - | 2.7886 | 2.7553 | 0.2434 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 2 | 43 | 18.1088 | - | - | 2.7886 | 2.7553 | 0.2607 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 4 | 43 | 34.9446 | - | - | 2.7886 | 2.7553 | 0.1768 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 1 | 43 | 10.5825 | - | - | 2.6263 | 2.5978 | 0.2570 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 2 | 43 | 18.4185 | - | - | 2.6263 | 2.5978 | 0.1590 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_flash_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 4 | 43 | 34.9837 | - | - | 2.6263 | 2.5978 | 0.1298 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 48 | 0 | 1 | 61 | 25.8363 | - | - | 1.9206 | 1.8973 | 0.3349 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 48 | 0 | 2 | 61 | 48.3385 | - | - | 1.9206 | 1.8973 | 0.1044 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 48 | 0 | 4 | 61 | 89.5472 | - | - | 1.9206 | 1.8973 | 0.0948 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 96 | 1 | 1 | 61 | 27.0129 | - | - | 1.8737 | 1.8414 | 0.2135 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 96 | 1 | 2 | 61 | 49.2765 | - | - | 1.8737 | 1.8414 | 0.1667 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 96 | 1 | 4 | 61 | 95.8346 | - | - | 1.8737 | 1.8414 | 0.1300 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 144 | 2 | 1 | 61 | 28.0627 | - | - | 1.8512 | 1.8327 | 0.1150 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 144 | 2 | 2 | 61 | 50.4733 | - | - | 1.8512 | 1.8327 | 0.1989 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 144 | 2 | 4 | 61 | 98.3190 | - | - | 1.8512 | 1.8327 | 0.0640 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 192 | 3 | 1 | 61 | 29.1157 | - | - | 1.8245 | 1.8184 | 0.1039 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 192 | 3 | 2 | 61 | 51.6171 | - | - | 1.8245 | 1.8184 | 0.1145 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 48 | 192 | 3 | 4 | 61 | 97.6774 | - | - | 1.8245 | 1.8184 | 0.0955 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 96 | 0 | 1 | 61 | 27.0109 | - | - | 1.8784 | 1.8612 | 0.0934 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 96 | 0 | 2 | 61 | 49.3411 | - | - | 1.8784 | 1.8612 | 0.1439 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 96 | 0 | 4 | 61 | 95.9790 | - | - | 1.8784 | 1.8612 | 0.1328 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 192 | 1 | 1 | 61 | 29.1141 | - | - | 1.8239 | 1.8158 | 0.1984 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 192 | 1 | 2 | 61 | 51.6137 | - | - | 1.8239 | 1.8158 | 0.0795 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 192 | 1 | 4 | 61 | 97.6898 | - | - | 1.8239 | 1.8158 | 0.0653 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 288 | 2 | 1 | 61 | 30.2978 | - | - | 1.8271 | 1.7421 | 0.1257 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 288 | 2 | 2 | 61 | 53.7012 | - | - | 1.8271 | 1.7421 | 0.0681 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 288 | 2 | 4 | 61 | 101.0496 | - | - | 1.8271 | 1.7421 | 0.0440 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 384 | 3 | 1 | 61 | 31.8137 | - | - | 1.8387 | 1.8023 | 0.0913 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 384 | 3 | 2 | 61 | 55.7910 | - | - | 1.8387 | 1.8023 | 0.0915 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| deepseek_v4_pro_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 2A6F | 96 | 384 | 3 | 4 | 61 | 103.1615 | - | - | 1.8387 | 1.8023 | 0.0602 | - | yes | yes | e507eacf858d2046bdc2cca02ed86c0e58bd6c60 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 1 | 62 | 7.8110 | - | - | 3.4755 | 3.4498 | 0.3451 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 2 | 62 | 14.8610 | - | - | 3.4755 | 3.4498 | 0.4218 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 4 | 62 | 29.1733 | - | - | 3.4755 | 3.4498 | 0.5180 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 1 | 62 | 8.1422 | - | - | 3.3252 | 3.2865 | 0.3252 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 2 | 62 | 15.1050 | - | - | 3.3252 | 3.2865 | 0.2764 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 4 | 62 | 29.7929 | - | - | 3.3252 | 3.2865 | 0.2093 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 1 | 62 | 8.3466 | - | - | 3.2124 | 3.1326 | 1.3171 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 2 | 62 | 15.3587 | - | - | 3.2124 | 3.1326 | 1.7245 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 4 | 62 | 29.9589 | - | - | 3.2124 | 3.1326 | 0.2695 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 1 | 62 | 8.3603 | - | - | 3.1182 | 3.0692 | 0.1717 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 2 | 62 | 15.4785 | - | - | 3.1182 | 3.0692 | 0.4408 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 4 | 62 | 30.0967 | - | - | 3.1182 | 3.0692 | 0.2908 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 1 | 62 | 8.1873 | - | - | 3.3851 | 3.3456 | 0.1684 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 2 | 62 | 15.1100 | - | - | 3.3851 | 3.3456 | 0.8578 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 4 | 62 | 29.8236 | - | - | 3.3851 | 3.3456 | 0.1304 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 1 | 62 | 8.3933 | - | - | 3.0792 | 3.0293 | 0.1956 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 2 | 62 | 15.4722 | - | - | 3.0792 | 3.0293 | 0.2838 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 4 | 62 | 30.1372 | - | - | 3.0792 | 3.0293 | 0.2470 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 1 | 62 | 9.4391 | - | - | 2.7913 | 2.7715 | 0.3727 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 2 | 62 | 15.3304 | - | - | 2.7913 | 2.7715 | 0.2284 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 4 | 62 | 30.5395 | - | - | 2.7913 | 2.7715 | 0.2692 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 1 | 62 | 9.8629 | - | - | 2.5206 | 2.4785 | 0.3234 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 2 | 62 | 15.5446 | - | - | 2.5206 | 2.4785 | 0.5358 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m25_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 4 | 62 | 30.7495 | - | - | 2.5206 | 2.4785 | 0.3138 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 1 | 57 | 12.5697 | - | - | 2.7277 | 2.7080 | 0.2508 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 2 | 57 | 22.8323 | - | - | 2.7277 | 2.7080 | 0.0848 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 4 | 57 | 45.4499 | - | - | 2.7277 | 2.7080 | 0.1470 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 1 | 57 | 13.1493 | - | - | 2.6778 | 2.6508 | 0.2882 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 2 | 57 | 23.9845 | - | - | 2.6778 | 2.6508 | 0.1485 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 4 | 57 | 45.6604 | - | - | 2.6778 | 2.6508 | 0.1289 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 1 | 57 | 13.6075 | - | - | 2.5723 | 2.5585 | 0.1566 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 2 | 57 | 24.6672 | - | - | 2.5723 | 2.5585 | 0.1429 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 4 | 57 | 47.8820 | - | - | 2.5723 | 2.5585 | 0.0827 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 1 | 57 | 13.7619 | - | - | 2.6047 | 2.5774 | 0.2532 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 2 | 57 | 24.8115 | - | - | 2.6047 | 2.5774 | 0.1646 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 4 | 57 | 48.0967 | - | - | 2.6047 | 2.5774 | 0.0651 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 1 | 57 | 13.1131 | - | - | 2.6760 | 2.6605 | 0.1292 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 2 | 57 | 23.9865 | - | - | 2.6760 | 2.6605 | 0.1484 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 4 | 57 | 45.6451 | - | - | 2.6760 | 2.6605 | 0.1652 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 1 | 57 | 13.7581 | - | - | 2.5631 | 2.5441 | 0.2724 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 2 | 57 | 24.8386 | - | - | 2.5631 | 2.5441 | 0.1154 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 4 | 57 | 48.0768 | - | - | 2.5631 | 2.5441 | 0.1248 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 1 | 57 | 15.6805 | - | - | 2.4464 | 2.4238 | 0.5758 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 2 | 57 | 25.1874 | - | - | 2.4464 | 2.4238 | 0.1026 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 4 | 57 | 49.2977 | - | - | 2.4464 | 2.4238 | 0.0877 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 1 | 57 | 16.3408 | - | - | 2.2105 | 2.1896 | 0.1709 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 2 | 57 | 25.5526 | - | - | 2.2105 | 2.1896 | 0.0663 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| minimax_m3_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 4 | 57 | 49.6180 | - | - | 2.2105 | 2.1896 | 0.2397 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 1 | 94 | 9.5629 | - | - | 3.7204 | 3.6900 | 0.2382 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 2 | 94 | 16.1205 | - | - | 3.7204 | 3.6900 | 0.3506 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 48 | 0 | 4 | 94 | 32.1254 | - | - | 3.7204 | 3.6900 | 0.3840 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 1 | 94 | 10.0842 | - | - | 3.5694 | 3.4195 | 0.1257 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 2 | 94 | 16.6877 | - | - | 3.5694 | 3.4195 | 1.7508 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 96 | 1 | 4 | 94 | 32.4270 | - | - | 3.5694 | 3.4195 | 0.1861 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 1 | 94 | 10.4962 | - | - | 3.1829 | 3.1392 | 0.2046 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 2 | 94 | 17.4904 | - | - | 3.1829 | 3.1392 | 1.3106 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 144 | 2 | 4 | 94 | 33.0899 | - | - | 3.1829 | 3.1392 | 0.2515 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 1 | 94 | 10.8752 | - | - | 2.9876 | 2.9549 | 0.1517 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 2 | 94 | 17.5346 | - | - | 2.9876 | 2.9549 | 0.4964 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 48 | 192 | 3 | 4 | 94 | 33.3450 | - | - | 2.9876 | 2.9549 | 0.2277 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 1 | 94 | 10.1211 | - | - | 3.4825 | 3.4188 | 0.2375 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 2 | 94 | 16.6703 | - | - | 3.4825 | 3.4188 | 0.3233 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 96 | 0 | 4 | 94 | 32.3444 | - | - | 3.4825 | 3.4188 | 0.2822 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 1 | 94 | 10.8942 | - | - | 2.9631 | 2.9301 | 0.3034 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 2 | 94 | 17.6008 | - | - | 2.9631 | 2.9301 | 0.3145 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 192 | 1 | 4 | 94 | 33.3163 | - | - | 2.9631 | 2.9301 | 0.2968 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 1 | 94 | 12.6773 | - | - | 2.7160 | 2.6125 | 0.2100 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 2 | 94 | 18.5100 | - | - | 2.7160 | 2.6125 | 0.1796 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 288 | 2 | 4 | 94 | 35.7184 | - | - | 2.7160 | 2.6125 | 0.1249 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 1 | 94 | 13.8410 | - | - | 2.4070 | 2.3870 | 0.2046 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 2 | 94 | 19.3511 | - | - | 2.4070 | 2.3870 | 0.4395 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |
| qwen3_235b_fp4 | afd | w4a8_mxfp4_mxfp8 | b200_sxm | 4A4F | 96 | 384 | 3 | 4 | 94 | 35.9506 | - | - | 2.4070 | 2.3870 | 0.2711 | - | yes | yes | cd0bd85b34a93a24f53bfd1fa814693f5ab77b88 |

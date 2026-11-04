# Benchmarks on NVIDIA GeForce GTX 1650

| kernel | shape | impl | ms | rate |
|---|---|---|---|---|
| act_quant | 16x7168 float32 | torch eager | 0.0472 | 12.23 GB/s |
| act_quant | 16x7168 float32 | triton int8 | 0.0132 | 43.69 GB/s |
| act_quant | 16x7168 float32 | triton int8 K-major (v2) | 0.0134 | 42.93 GB/s |
| act_quant | 16x7168 bfloat16 | torch eager | 0.0466 | 7.47 GB/s |
| act_quant | 16x7168 bfloat16 | triton int8 | 0.0116 | 30.03 GB/s |
| act_quant | 16x7168 bfloat16 | triton int8 K-major (v2) | 0.0124 | 27.93 GB/s |
| act_quant | 256x7168 float32 | torch eager | 0.5046 | 18.3 GB/s |
| act_quant | 256x7168 float32 | triton int8 | 0.0632 | 146.16 GB/s |
| act_quant | 256x7168 float32 | triton int8 K-major (v2) | 0.1059 | 87.21 GB/s |
| act_quant | 256x7168 bfloat16 | torch eager | 0.5916 | 9.4 GB/s |
| act_quant | 256x7168 bfloat16 | triton int8 | 0.0405 | 137.3 GB/s |
| act_quant | 256x7168 bfloat16 | triton int8 K-major (v2) | 0.0949 | 58.61 GB/s |
| act_quant | 2048x7168 float32 | torch eager | 3.8279 | 19.29 GB/s |
| act_quant | 2048x7168 float32 | triton int8 | 0.5528 | 133.61 GB/s |
| act_quant | 2048x7168 float32 | triton int8 K-major (v2) | 0.7144 | 103.39 GB/s |
| act_quant | 2048x7168 bfloat16 | torch eager | 4.4713 | 9.95 GB/s |
| act_quant | 2048x7168 bfloat16 | triton int8 | 0.3871 | 114.95 GB/s |
| act_quant | 2048x7168 bfloat16 | triton int8 K-major (v2) | 0.6738 | 66.04 GB/s |
| act_quant | 8192x7168 float32 | torch eager | 15.1296 | 19.53 GB/s |
| act_quant | 8192x7168 float32 | triton int8 | 2.3115 | 127.81 GB/s |
| act_quant | 8192x7168 float32 | triton int8 K-major (v2) | 2.8114 | 105.09 GB/s |
| act_quant | 8192x7168 bfloat16 | torch eager | 17.7991 | 10.0 GB/s |
| act_quant | 8192x7168 bfloat16 | triton int8 | 1.8461 | 96.42 GB/s |
| act_quant | 8192x7168 bfloat16 | triton int8 K-major (v2) | 2.661 | 66.89 GB/s |
| gemm | q_a 7168->1536 M=32 | torch fp32 (unquantized) | 0.9208 | 0.77 TFLOP/s |
| gemm | q_a 7168->1536 M=32 | torch fp16 (unquantized) | 2.0902 | 0.34 TFLOP/s |
| gemm | q_a 7168->1536 M=32 | torch dequant + fp32 | 0.9471 | 0.74 TFLOP/s |
| gemm | q_a 7168->1536 M=32 | torch _int_mm (no scales) | 0.2694 | 2.62 TOP/s |
| gemm | q_a 7168->1536 M=32 | triton int8 v0 | 0.7006 | 1.01 TOP/s |
| gemm | q_a 7168->1536 M=32 | triton int8 v1 | 0.6647 | 1.06 TOP/s |
| gemm | q_a 7168->1536 M=32 | triton int8 v2 (dp4a) | 0.5818 | 1.21 TOP/s |
| gemm | q_a 7168->1536 M=128 | torch fp32 (unquantized) | 1.2775 | 2.21 TFLOP/s |
| gemm | q_a 7168->1536 M=128 | torch fp16 (unquantized) | 8.361 | 0.34 TFLOP/s |
| gemm | q_a 7168->1536 M=128 | torch dequant + fp32 | 1.3642 | 2.07 TFLOP/s |
| gemm | q_a 7168->1536 M=128 | torch _int_mm (no scales) | 0.2706 | 10.42 TOP/s |
| gemm | q_a 7168->1536 M=128 | triton int8 v0 | 2.6001 | 1.08 TOP/s |
| gemm | q_a 7168->1536 M=128 | triton int8 v1 | 2.2643 | 1.24 TOP/s |
| gemm | q_a 7168->1536 M=128 | triton int8 v2 (dp4a) | 0.6329 | 4.45 TOP/s |
| gemm | q_a 7168->1536 M=1024 | torch fp32 (unquantized) | 8.0682 | 2.79 TFLOP/s |
| gemm | q_a 7168->1536 M=1024 | torch fp16 (unquantized) | 66.5879 | 0.34 TFLOP/s |
| gemm | q_a 7168->1536 M=1024 | torch dequant + fp32 | 8.6805 | 2.6 TFLOP/s |
| gemm | q_a 7168->1536 M=1024 | torch _int_mm (no scales) | 1.8627 | 12.11 TOP/s |
| gemm | q_a 7168->1536 M=1024 | triton int8 v0 | 17.9426 | 1.26 TOP/s |
| gemm | q_a 7168->1536 M=1024 | triton int8 v1 | 17.6018 | 1.28 TOP/s |
| gemm | q_a 7168->1536 M=1024 | triton int8 v2 (dp4a) | 4.6715 | 4.83 TOP/s |
| gemm | q_a 7168->1536 M=4096 | torch fp32 (unquantized) | 31.804 | 2.84 TFLOP/s |
| gemm | q_a 7168->1536 M=4096 | torch fp16 (unquantized) | 266.3829 | 0.34 TFLOP/s |
| gemm | q_a 7168->1536 M=4096 | torch dequant + fp32 | 34.2695 | 2.63 TFLOP/s |
| gemm | q_a 7168->1536 M=4096 | torch _int_mm (no scales) | 7.6313 | 11.82 TOP/s |
| gemm | q_a 7168->1536 M=4096 | triton int8 v0 | 70.3896 | 1.28 TOP/s |
| gemm | q_a 7168->1536 M=4096 | triton int8 v1 | 70.3117 | 1.28 TOP/s |
| gemm | q_a 7168->1536 M=4096 | triton int8 v2 (dp4a) | 18.5427 | 4.86 TOP/s |
| gemm | expert up 7168->2048 M=32 | torch fp32 (unquantized) | 1.4141 | 0.66 TFLOP/s |
| gemm | expert up 7168->2048 M=32 | torch fp16 (unquantized) | 2.7885 | 0.34 TFLOP/s |
| gemm | expert up 7168->2048 M=32 | torch dequant + fp32 | 1.4054 | 0.67 TFLOP/s |
| gemm | expert up 7168->2048 M=32 | torch _int_mm (no scales) | 0.5287 | 1.78 TOP/s |
| gemm | expert up 7168->2048 M=32 | triton int8 v0 | 1.3125 | 0.72 TOP/s |
| gemm | expert up 7168->2048 M=32 | triton int8 v1 | 0.988 | 0.95 TOP/s |
| gemm | expert up 7168->2048 M=32 | triton int8 v2 (dp4a) | 0.5921 | 1.59 TOP/s |
| gemm | expert up 7168->2048 M=128 | torch fp32 (unquantized) | 1.6179 | 2.32 TFLOP/s |
| gemm | expert up 7168->2048 M=128 | torch fp16 (unquantized) | 11.1048 | 0.34 TFLOP/s |
| gemm | expert up 7168->2048 M=128 | torch dequant + fp32 | 1.6895 | 2.22 TFLOP/s |
| gemm | expert up 7168->2048 M=128 | torch _int_mm (no scales) | 0.5314 | 7.07 TOP/s |
| gemm | expert up 7168->2048 M=128 | triton int8 v0 | 3.2363 | 1.16 TOP/s |
| gemm | expert up 7168->2048 M=128 | triton int8 v1 | 3.2211 | 1.17 TOP/s |
| gemm | expert up 7168->2048 M=128 | triton int8 v2 (dp4a) | 1.1238 | 3.34 TOP/s |
| gemm | expert up 7168->2048 M=1024 | torch fp32 (unquantized) | 23.2304 | 1.29 TFLOP/s |
| gemm | expert up 7168->2048 M=1024 | torch fp16 (unquantized) | 90.352 | 0.33 TFLOP/s |
| gemm | expert up 7168->2048 M=1024 | torch dequant + fp32 | 26.432 | 1.14 TFLOP/s |
| gemm | expert up 7168->2048 M=1024 | torch _int_mm (no scales) | 2.6544 | 11.33 TOP/s |
| gemm | expert up 7168->2048 M=1024 | triton int8 v0 | 23.7005 | 1.27 TOP/s |
| gemm | expert up 7168->2048 M=1024 | triton int8 v1 | 23.6659 | 1.27 TOP/s |
| gemm | expert up 7168->2048 M=1024 | triton int8 v2 (dp4a) | 6.5397 | 4.6 TOP/s |
| gemm | expert up 7168->2048 M=4096 | torch fp32 (unquantized) | 42.6224 | 2.82 TFLOP/s |
| gemm | expert up 7168->2048 M=4096 | torch fp16 (unquantized) | 351.9963 | 0.34 TFLOP/s |
| gemm | expert up 7168->2048 M=4096 | torch dequant + fp32 | 45.0884 | 2.67 TFLOP/s |
| gemm | expert up 7168->2048 M=4096 | torch _int_mm (no scales) | 10.1974 | 11.79 TOP/s |
| gemm | expert up 7168->2048 M=4096 | triton int8 v0 | 94.0066 | 1.28 TOP/s |
| gemm | expert up 7168->2048 M=4096 | triton int8 v1 | 93.6286 | 1.28 TOP/s |
| gemm | expert up 7168->2048 M=4096 | triton int8 v2 (dp4a) | 24.5898 | 4.89 TOP/s |
| gemm | expert down 2048->7168 M=32 | torch fp32 (unquantized) | 1.0475 | 0.9 TFLOP/s |
| gemm | expert down 2048->7168 M=32 | torch fp16 (unquantized) | 2.7249 | 0.34 TFLOP/s |
| gemm | expert down 2048->7168 M=32 | torch dequant + fp32 | 1.0593 | 0.89 TFLOP/s |
| gemm | expert down 2048->7168 M=32 | torch _int_mm (no scales) | 0.3131 | 3.0 TOP/s |
| gemm | expert down 2048->7168 M=32 | triton int8 v0 | 0.7503 | 1.25 TOP/s |
| gemm | expert down 2048->7168 M=32 | triton int8 v1 | 0.746 | 1.26 TOP/s |
| gemm | expert down 2048->7168 M=32 | triton int8 v2 (dp4a) | 0.224 | 4.19 TOP/s |
| gemm | expert down 2048->7168 M=128 | torch fp32 (unquantized) | 2.1713 | 1.73 TFLOP/s |
| gemm | expert down 2048->7168 M=128 | torch fp16 (unquantized) | 10.8934 | 0.34 TFLOP/s |
| gemm | expert down 2048->7168 M=128 | torch dequant + fp32 | 2.2016 | 1.71 TFLOP/s |
| gemm | expert down 2048->7168 M=128 | torch _int_mm (no scales) | 0.3181 | 11.81 TOP/s |
| gemm | expert down 2048->7168 M=128 | triton int8 v0 | 2.9427 | 1.28 TOP/s |
| gemm | expert down 2048->7168 M=128 | triton int8 v1 | 2.9339 | 1.28 TOP/s |
| gemm | expert down 2048->7168 M=128 | triton int8 v2 (dp4a) | 0.784 | 4.79 TOP/s |
| gemm | expert down 2048->7168 M=1024 | torch fp32 (unquantized) | 17.5715 | 1.71 TFLOP/s |
| gemm | expert down 2048->7168 M=1024 | torch fp16 (unquantized) | 86.9604 | 0.35 TFLOP/s |
| gemm | expert down 2048->7168 M=1024 | torch dequant + fp32 | 17.6586 | 1.7 TFLOP/s |
| gemm | expert down 2048->7168 M=1024 | torch _int_mm (no scales) | 2.5089 | 11.98 TOP/s |
| gemm | expert down 2048->7168 M=1024 | triton int8 v0 | 23.4086 | 1.28 TOP/s |
| gemm | expert down 2048->7168 M=1024 | triton int8 v1 | 23.3865 | 1.29 TOP/s |
| gemm | expert down 2048->7168 M=1024 | triton int8 v2 (dp4a) | 6.3613 | 4.73 TOP/s |
| gemm | expert down 2048->7168 M=4096 | torch fp32 (unquantized) | 76.1565 | 1.58 TFLOP/s |
| gemm | expert down 2048->7168 M=4096 | torch fp16 (unquantized) | 347.7975 | 0.35 TFLOP/s |
| gemm | expert down 2048->7168 M=4096 | torch dequant + fp32 | 71.5841 | 1.68 TFLOP/s |
| gemm | expert down 2048->7168 M=4096 | torch _int_mm (no scales) | 10.4104 | 11.55 TOP/s |
| gemm | expert down 2048->7168 M=4096 | triton int8 v0 | 93.5732 | 1.29 TOP/s |
| gemm | expert down 2048->7168 M=4096 | triton int8 v1 | 93.5222 | 1.29 TOP/s |
| gemm | expert down 2048->7168 M=4096 | triton int8 v2 (dp4a) | 24.8394 | 4.84 TOP/s |
| index | decode B=4 M=1 N=4096 | torch fp32 (chunked) | 0.3866 | 0.69 TOP/s |
| index | decode B=4 M=1 N=4096 | torch int8 _int_mm | 0.5659 | 0.47 TOP/s |
| index | decode B=4 M=1 N=4096 | triton v0 (DeepSeek tile) | 0.2373 | 1.13 TOP/s |
| index | decode B=4 M=1 N=4096 | triton v1 tuned q1 n32 w8 | 0.2182 | 1.23 TOP/s |
| index | decode B=4 M=1 N=4096 | triton v2 dp4a q1 n128 w2 u8 | 0.0556 | 4.83 TOP/s |
| index | decode B=4 M=1 N=32768 | torch fp32 (chunked) | 2.7553 | 0.78 TOP/s |
| index | decode B=4 M=1 N=32768 | torch int8 _int_mm | 8.208 | 0.26 TOP/s |
| index | decode B=4 M=1 N=32768 | triton v0 (DeepSeek tile) | 1.6985 | 1.26 TOP/s |
| index | decode B=4 M=1 N=32768 | triton v1 tuned q1 n32 w4 | 1.6805 | 1.28 TOP/s |
| index | decode B=4 M=1 N=32768 | triton v2 dp4a q1 n128 w2 u8 | 0.3419 | 6.28 TOP/s |
| index | decode B=1 M=1 N=131072 | torch fp32 (chunked) | 2.6575 | 0.81 TOP/s |
| index | decode B=1 M=1 N=131072 | torch int8 _int_mm | 8.119 | 0.26 TOP/s |
| index | decode B=1 M=1 N=131072 | triton v0 (DeepSeek tile) | 1.6984 | 1.26 TOP/s |
| index | decode B=1 M=1 N=131072 | triton v1 tuned q1 n32 w4 | 1.6801 | 1.28 TOP/s |
| index | decode B=1 M=1 N=131072 | triton v2 dp4a q1 n128 w2 u8 | 0.3417 | 6.28 TOP/s |
| index | prefill B=1 M=1024 N=1024 | torch fp32 (chunked) | 15.5811 | 1.1 TOP/s |
| index | prefill B=1 M=1024 N=1024 | torch int8 _int_mm | 13.8703 | 1.24 TOP/s |
| index | prefill B=1 M=1024 N=1024 | triton v0 (DeepSeek tile) | 13.3831 | 1.28 TOP/s |
| index | prefill B=1 M=1024 N=1024 | triton v1 tuned q1 n32 w8 | 13.3707 | 1.28 TOP/s |
| index | prefill B=1 M=1024 N=1024 | triton v1 + causal skip | 6.9017 | 2.49 TOP/s |
| index | prefill B=1 M=1024 N=1024 | triton v2 dp4a q1 n128 w2 u8 | 2.5513 | 6.73 TOP/s |
| index | prefill B=1 M=1024 N=1024 | triton v2 dp4a + causal skip | 1.561 | 11.01 TOP/s |
| index | prefill B=1 M=4096 N=4096 | torch fp32 (chunked) | 304.1537 | 0.9 TOP/s |
| index | prefill B=1 M=4096 N=4096 | torch int8 _int_mm | 220.1642 | 1.25 TOP/s |
| index | prefill B=1 M=4096 N=4096 | triton v0 (DeepSeek tile) | 213.676 | 1.29 TOP/s |
| index | prefill B=1 M=4096 N=4096 | triton v1 tuned q1 n128 w4 | 213.6909 | 1.29 TOP/s |
| index | prefill B=1 M=4096 N=4096 | triton v1 + causal skip | 110.2003 | 2.49 TOP/s |
| index | prefill B=1 M=4096 N=4096 | triton v2 dp4a q1 n128 w2 u8 | 39.9194 | 6.89 TOP/s |
| index | prefill B=1 M=4096 N=4096 | triton v2 dp4a + causal skip | 21.1507 | 13.0 TOP/s |

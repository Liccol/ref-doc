#!/usr/bin/env bash
set -euo pipefail
# Inside lch-vllm-030. Start rank 0 on node25, rank 1 on node26.
rank=${1:?usage: bash serve.sh 0|1 [baseline|sleep|sp|pp]}
case "$rank" in 0) ip=172.27.4.105;; 1) ip=172.27.4.106;; *) exit 2;; esac
mode=${2:-baseline}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export GLOO_SOCKET_IFNAME=data0.172 IP_SOCKET_IFNAME=data0.172 HCCL_SOCKET_IFNAME=data0.172
export HCCL_IF_IP="$ip" VLLM_HOST_IP="$ip" HCCL_OP_EXPANSION_MODE=CCU_SCHED
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENGINE_READY_TIMEOUT_S=3600
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
extra=()
if [[ "$mode" == pp ]]; then
  topology=(--data-parallel-size 1 --pipeline-parallel-size 4 --nnodes 2 --node-rank "$rank" --master-addr 172.27.4.105 --master-port 29502)
  [[ "$rank" == 0 ]] || topology+=(--headless)
else
  topology=(--data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-address 177.11.25.1 --data-parallel-rpc-port 31071)
  [[ "$rank" == 0 ]] || topology+=(--data-parallel-start-rank 2 --headless)
fi
if [[ "$mode" == sleep ]]; then
  export VLLM_SERVER_DEV_MODE=1
  extra+=(--enable-sleep-mode --additional-config '{"weight_nz_mode":0}' --no-enable-prefix-caching)
elif [[ "$mode" == sp ]]; then
  extra+=(--additional-config '{"enable_flashcomm1":true}')
elif [[ "$mode" != baseline && "$mode" != pp ]]; then exit 2; fi
exec vllm serve /mnt/weight/MiniMax-M3.1-Flash-BF16/ \
  --host 0.0.0.0 --port 31070 --served-model-name MiniMax-M3.1-Flash-BF16 \
  --tensor-parallel-size 4 "${topology[@]}" --enable-expert-parallel \
  --max-model-len 131072 --dtype bfloat16 --trust-remote-code \
  --generation-config vllm --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":1,"video":0}' --distributed-executor-backend mp \
  --reasoning-parser minimax_m3 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' "${extra[@]}"

#!/usr/bin/env bash
# 逐个配置验证能否消掉 GoogLeNet 的段错误。
# 每个配置跑 N 次，因为崩溃是间发性的，单次通过不能说明问题。
# 用法：bash fix_matrix.sh [iters] [repeats]

ITERS=${1:-500}
REPEATS=${2:-3}

run() {
  local label="$1"; shift
  local ok=0 crash=0
  for i in $(seq 1 "$REPEATS"); do
    if env "$@" python -u probe_crash.py googlenet "$ITERS" 2>&1 | grep -q '^OK'; then
      ok=$((ok + 1))
    else
      crash=$((crash + 1))
    fi
  done
  printf '%-40s 通过 %d / 崩溃 %d\n' "$label" "$ok" "$crash"
}

echo "iters=$ITERS  repeats=$REPEATS"
echo "----------------------------------------------------------"
run "baseline (4 线程, oneDNN 开)"       DUMMY=1
run "OPENBLAS_NUM_THREADS=1"             OPENBLAS_NUM_THREADS=1
run "KMP_BLOCKTIME=0 + PASSIVE"          KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE
run "OMP_NUM_THREADS=2"                  OMP_NUM_THREADS=2
run "OMP_NUM_THREADS=1"                  OMP_NUM_THREADS=1
run "oneDNN 关闭"                         MKLDNN=off

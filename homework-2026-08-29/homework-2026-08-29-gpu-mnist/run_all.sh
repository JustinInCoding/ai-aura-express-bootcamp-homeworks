#!/usr/bin/env bash
# 四网络对比（GPU 版）：每个模型一个独立进程，崩溃后自动重试续训。
#
# 为什么要分进程：单个模型失败不影响其余三个的结果落盘；GPU 上 OOM 或
# 驱动超时等问题也是分进程更健壮。
#
# 与 CPU 版的区别：
#   - CPU 版需要单独降 GoogLeNet 的线程来规避 OpenMP 段错误，GPU 上不需要
#   - CPU 版有 os._exit(0) 规避退出阶段段错误，GPU 版没有
#   - 重试循环保留，但触发原因不同：GPU 主要防 SSH 断连或实例按量被回收
#
# 用法：bash run_all.sh [epochs] [tag] [每个模型最大尝试次数]
# 环境变量：
#   FORCE=1       重跑已有结果的模型（默认跳过）
#   MODEL=resnet  只跑指定模型（默认 four）
#
# 注意：--num-workers 与本脚本无关，如有需要可在调用前 export 或在
# train.py 改默认值。shell 脚本只传 epochs/tag，不传运行环境参数。

set -u

EPOCHS=${1:-5}
TAG=${2:-baseline}
MAX_TRY=${3:-3}
FORCE=${FORCE:-0}
mkdir -p logs results

# 默认跑全部四个，也可通过 MODEL=resnet bash run_all.sh 指定
MODELS=${MODEL:-"alexnet vgg googlenet resnet"}

for m in $MODELS; do
  result="results/${TAG}_${m}.json"
  if [ -f "$result" ] && [ "$FORCE" != "1" ]; then
    echo "######## 跳过 ${m}：$result 已存在（FORCE=1 可强制重跑）"
    continue
  fi

  echo ""
  echo "################ $m (epochs=$EPOCHS, tag=$TAG) ################"
  for try in $(seq 1 "$MAX_TRY"); do
    [ "$try" -gt 1 ] && echo ">>> 第 $try 次尝试（从断点续训）"
    # -u 关掉 stdout 块缓冲：输出接到管道时 Python 默认攒满 4KB 才写，
    # 进程被信号杀掉时缓冲区内容会随进程一起丢失，日志就成了空文件。
    python -u train.py --model "$m" --epochs "$EPOCHS" --tag "$TAG" \
      2>&1 | tee -a "logs/${TAG}_${m}.log"
    code=${PIPESTATUS[0]}
    # 判成功的依据是"结果文件是否落盘"，而不是退出码。
    if [ -f "$result" ]; then
      [ "$code" -ne 0 ] && echo ">>> ${m} 结果已落盘，退出码 $code 来自退出阶段，忽略"
      break
    fi
    [ "$code" -eq 0 ] && break
    echo ">>> $m 异常退出（exit=${code}）"
    if [ "$try" -eq "$MAX_TRY" ]; then
      echo ">>> $m 连续 $MAX_TRY 次失败，放弃。日志见 logs/${TAG}_${m}.log"
    fi
  done
done

echo ""
echo "################ 全部结束，汇总 ################"
python -u compare.py --tag "$TAG"
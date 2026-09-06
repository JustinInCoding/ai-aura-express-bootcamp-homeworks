#!/usr/bin/env bash
# 四网络对比：每个模型一个独立进程，崩溃后自动重试续训。
#
# 为什么分进程：单进程连续训 4 个模型时踩到 libiomp5 的 OpenMP 段错误，一崩就全丢。
# 分进程后 OMP 运行时每次从干净状态起步，单个模型失败也不影响其余三个的结果落盘。
#
# 为什么要重试 + 断点：GoogLeNet 的 backward 在本机会间发性触发段错误（README 5.4），
# 根因在 torch 2.2.2 的 CPU 后端。train.py 每个 epoch 存断点，重试时自动续训。
#
# 为什么 GoogLeNet 单独降线程：崩溃是 OMP 并行区的竞态，线程少了竞态窗口就没了。
# 只对 GoogLeNet 降，其余三个网络保持 4 线程全速。
#
# 用法：bash run_all.sh [epochs] [tag] [每个模型最大尝试次数]
# 环境变量：
#   THREADS_GOOGLENET=2  GoogLeNet 用的 intra-op 线程数（默认 2，仍崩就试 1）
#   THREADS=0            其余模型的线程数，0 = 用默认的 4
#   FORCE=1              重跑已有结果的模型（默认跳过）

set -u  # 故意不加 -e：某个模型最终失败也要继续跑后面的

EPOCHS=${1:-5}
TAG=${2:-baseline}
MAX_TRY=${3:-6}
THREADS=${THREADS:-0}
THREADS_GOOGLENET=${THREADS_GOOGLENET:-2}
FORCE=${FORCE:-0}
mkdir -p logs results

for m in alexnet vgg googlenet resnet; do
  result="results/${TAG}_${m}.json"
  if [ -f "$result" ] && [ "$FORCE" != "1" ]; then
    # 变量名一律加花括号：bash 3.2 会把紧跟其后的全角标点的首字节也算进变量名
    echo "######## 跳过 ${m}：$result 已存在（FORCE=1 可强制重跑）"
    continue
  fi

  if [ "$m" = "googlenet" ]; then t=$THREADS_GOOGLENET; else t=$THREADS; fi

  echo ""
  echo "################ $m (epochs=$EPOCHS, tag=$TAG, threads=$t) ################"
  for try in $(seq 1 "$MAX_TRY"); do
    [ "$try" -gt 1 ] && echo ">>> 第 $try 次尝试（从断点续训）"
    # -u 关掉 stdout 块缓冲：输出接到管道时 Python 默认攒满 4KB 才写，
    # 进程被信号杀掉时缓冲区内容会随进程一起丢失，日志就成了空文件。
    # tee -a 追加，避免重试覆盖掉前几次的记录。
    python -u train.py --model "$m" --epochs "$EPOCHS" --tag "$TAG" --threads "$t" \
      2>&1 | tee -a "logs/${TAG}_${m}.log"
    code=${PIPESTATUS[0]}
    # 判成功的依据是"结果文件是否落盘"，而不是退出码。
    # 原因：训练完成、结果写盘之后，解释器退出阶段的 GC 仍可能段错误（退出码 139）。
    # 只看退出码会把一个已经跑完的模型判成失败，从头重训 —— GoogLeNet 白烧过 30 分钟。
    if [ -f "$result" ]; then
      [ "$code" -ne 0 ] && echo ">>> ${m} 结果已落盘，退出码 $code 来自退出阶段崩溃，忽略"
      break
    fi
    [ "$code" -eq 0 ] && break
    echo ">>> $m 异常退出（exit=${code}）"
    if [ "$try" -eq "$MAX_TRY" ]; then
      echo ">>> $m 连续 $MAX_TRY 次失败，放弃。日志见 logs/${TAG}_${m}.log"
      echo ">>> 建议改用单线程重跑：THREADS_GOOGLENET=1 bash run_all.sh $EPOCHS $TAG"
    fi
  done
done

echo ""
echo "################ 全部结束，汇总 ################"
python -u compare.py --tag "$TAG"

#!/usr/bin/env bash
# 超参数单变量对照实验（GPU 版）。
#
# 实验内容与 CPU 版完全一致，见 experiments.md 实验二。
# 固定用 VGG，每组只改一个变量，基准复用 baseline 结果。
#
# 与 CPU 版的区别：
#   - 去掉 --threads 参数（GPU 上不存在 OpenMP 线程问题）
#   - 重试次数从 3 降到 2（GPU 上崩溃概率远低于 CPU）
#   - 注释更新为 GPU 语境
#
# 用法：bash sweep.sh [epochs] [每组最大尝试次数]
# 环境变量：FORCE=1 重跑已有结果的组（默认跳过）

set -u

EPOCHS=${1:-5}
MAX_TRY=${2:-2}
MODEL=${MODEL:-vgg}
FORCE=${FORCE:-0}
mkdir -p logs results

# 格式："标签|额外参数"。标签同时用于结果文件名，必须唯一。
CONFIGS=(
  "lr001|--lr 0.001"                    # 学习率降 10 倍：收敛更慢，看是否欠拟合
  "lr1|--lr 0.1"                        # 学习率升 10 倍：看是否震荡/发散
  "adam|--optimizer adam --lr 0.001"    # 换自适应优化器（Adam 的常用 lr 就是 1e-3）
  "bs256|--batch-size 256"              # 大 batch：梯度更稳但更新次数少，且更吃并行
  "cosine|--scheduler cosine"           # 余弦退火：后期小 lr 精修
  "augment|--augment"                   # 随机旋转+平移增强：看正则化效果
)

for cfg in "${CONFIGS[@]}"; do
  tag=${cfg%%|*}          # | 之前是标签
  extra=${cfg#*|}         # | 之后是额外参数
  result="results/${tag}_${MODEL}.json"

  if [ -f "$result" ] && [ "$FORCE" != "1" ]; then
    echo "######## 跳过 ${tag}：$result 已存在（FORCE=1 可强制重跑）"
    continue
  fi

  echo ""
  echo "################ ${tag}  [$extra] ################"
  for try in $(seq 1 "$MAX_TRY"); do
    [ "$try" -gt 1 ] && echo ">>> 第 $try 次尝试（从断点续训）"
    # $extra 故意不加引号：需要让 shell 按空格拆成多个参数
    python -u train.py --model "$MODEL" --epochs "$EPOCHS" --tag "$tag" $extra \
      2>&1 | tee -a "logs/${tag}_${MODEL}.log"
    code=${PIPESTATUS[0]}
    if [ -f "$result" ]; then
      [ "$code" -ne 0 ] && echo ">>> ${tag} 结果已落盘，退出码 $code 来自退出阶段，忽略"
      break
    fi
    [ "$code" -eq 0 ] && break
    echo ">>> ${tag} 异常退出（exit=${code}）"
    [ "$try" -eq "$MAX_TRY" ] && echo ">>> ${tag} 连续 $MAX_TRY 次失败，放弃"
  done
done

echo ""
echo "################ 全部结束，汇总 ################"
python -u sweep_compare.py --model "$MODEL"
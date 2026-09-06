#!/usr/bin/env bash
# 超参数单变量对照实验（作业要求 3：调整超参数，观察准确率变化）。
#
# 为什么固定用 VGG：它在四个网络里最快（117s/epoch），6 组配置约 1 小时；
# 换 AlexNet 同样的实验要 4 小时以上。结论关心的是"超参数怎么影响训练"，
# 这个规律在各架构间是通用的，不需要每个网络都重跑一遍。
#
# 为什么每组只改一个变量：同时改多个的话，准确率变化无法归因到具体哪一项。
# 对照基准就是已经跑好的 results/baseline_vgg.json（lr=0.01 / SGD+momentum0.9
# / batch=64 / 无调度 / 无增强），所以这里不必重跑基准。
#
# 用法：bash sweep.sh [epochs] [每组最大尝试次数]
# 环境变量：FORCE=1 重跑已有结果的组（默认跳过）

set -u  # 故意不加 -e：某一组失败也要继续跑后面的

EPOCHS=${1:-5}
MAX_TRY=${2:-3}
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
    # -u 关掉 stdout 块缓冲，否则进程被信号杀掉时缓冲区里的日志会一起丢
    python -u train.py --model "$MODEL" --epochs "$EPOCHS" --tag "$tag" $extra \
      2>&1 | tee -a "logs/${tag}_${MODEL}.log"
    code=${PIPESTATUS[0]}
    # 判成功看结果文件而非退出码：训练完成落盘后，解释器退出阶段的 GC
    # 仍可能段错误（README 5.5），只看退出码会把跑完的实验判成失败并重跑
    if [ -f "$result" ]; then
      [ "$code" -ne 0 ] && echo ">>> ${tag} 结果已落盘，退出码 $code 来自退出阶段崩溃，忽略"
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

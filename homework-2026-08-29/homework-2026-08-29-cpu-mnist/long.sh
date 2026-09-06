#!/usr/bin/env bash
# 20 epochs 下 cosine × augment 的 2×2 因子实验。
#
# 验证 experiments.md 末尾的两个待验证方向：
#   1. 5 epochs 不够 —— cosine 和 augment 的末轮都是最佳轮，还在上升期
#   2. cosine + augment 能否叠加 —— 两者机制不冲突、过拟合方向互补
#
# 为什么要跑 e20_base：没有 20 轮的自己的基准，就没法把"轮数从 5 涨到 20"
# 和"加了 cosine/augment"分开。拿 e20_cos 跟 5 轮的 baseline 比，涨的那部分
# 无法归因。四组齐了还能算交互项：cosaug - cos - aug + base，正数才叫可叠加。
#
# 用法：bash long.sh [epochs] [每组最大尝试次数]
# 环境变量：FORCE=1 重跑已有结果的组（默认跳过）；MODEL=vgg 换架构

set -u  # 故意不加 -e：某一组失败也要继续跑后面的

EPOCHS=${1:-20}
MAX_TRY=${2:-3}
MODEL=${MODEL:-vgg}
FORCE=${FORCE:-0}
mkdir -p logs results

# 格式："标签|额外参数"。标签同时用于结果文件名，必须唯一。
# 顺序上先跑两个基准侧的组，万一时间不够，前两组也已经能回答方向 1。
CONFIGS=(
  "e20_base|"                                  # 2×2 的对照原点：无调度、无增强
  "e20_cos|--scheduler cosine"                 # 只开调度（T_max 会跟着 epochs 走）
  "e20_aug|--augment"                           # 只开增强
  "e20_cosaug|--scheduler cosine --augment"     # 两个都开，看是否叠加
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
  echo "################ ${tag}  [${extra:-无额外参数}] ################"
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
python -u sweep_compare.py --model "$MODEL" --prefix long --base e20_base \
  --tags e20_base e20_cos e20_aug e20_cosaug

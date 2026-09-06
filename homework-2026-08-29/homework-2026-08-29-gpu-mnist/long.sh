#!/usr/bin/env bash
# 20 epochs 下 cosine × augment 的 2×2 因子实验（GPU 版）。
#
# 实验内容与 CPU 版完全一致，见 experiments.md 实验三。
# 背景与四组的意义见 experiments.md 第 3 节。
#
# 用法：bash long.sh [epochs] [每组最大尝试次数]
# 环境变量：FORCE=1 重跑已有结果的组（默认跳过）；MODEL=vgg 换架构

set -u

EPOCHS=${1:-20}
MAX_TRY=${2:-2}
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
python -u sweep_compare.py --model "$MODEL" --prefix long --base e20_base \
  --tags e20_base e20_cos e20_aug e20_cosaug
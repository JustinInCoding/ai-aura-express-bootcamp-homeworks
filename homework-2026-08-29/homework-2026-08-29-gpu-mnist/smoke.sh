#!/usr/bin/env bash
# 冒烟测试：几十秒内验证移植是否正确，再决定要不要开长跑。
#
# 这一步不是可选的。GPU 版相对 CPU 版改了 device 搬运、计时同步、DataLoader
# worker、CUDA RNG 四处，任何一处写错都不会报错，只会悄悄给出错误的数字。
#
# 用法：bash smoke.sh

set -euo pipefail
OUT=/tmp/smoke_gpu
rm -rf "$OUT"

echo "==== 1/4 四个网络的参数量必须与 CPU 版完全一致 ===="
# models.py 是逐字节复制过来的，参数量对不上就说明文件被改动过
python - <<'PY'
from models import MODELS, build, count_params
EXPECT = {"alexnet": 4.82, "vgg": 1.13, "googlenet": 0.27, "resnet": 2.76}
ok = True
for name in MODELS:
    got = round(count_params(build(name)) / 1e6, 2)
    mark = "OK " if got == EXPECT[name] else "!! "
    ok &= got == EXPECT[name]
    print(f"  {mark}{name:<11} {got:.2f}M  (期望 {EXPECT[name]:.2f}M)")
raise SystemExit(0 if ok else "参数量与 CPU 版不一致，models.py 被改过了")
PY

echo ""
echo "==== 2/4 单模型 3 个 batch 跑通 ===="
python -u train.py --model vgg --epochs 1 --limit-batches 3 --tag smoke --out "$OUT"

echo ""
echo "==== 3/4 确认真的用上了 GPU ===="
# 上一步的日志里若出现「device=cpu」就说明白跑了。这里再断言一次。
python - <<'PY'
import json, glob
r = json.load(open(glob.glob("/tmp/smoke_gpu/smoke_vgg.json")[0]))
env = r["env"]
print(f"  torch {env['torch']}  gpu={env.get('gpu', '无')}")
if not env["cuda_available"]:
    raise SystemExit("!! 本次跑在 CPU 上，检查实例是否为带卡规格、驱动是否正常")
print("  OK  确认使用 GPU")
PY

echo ""
echo "==== 4/4 随机性自检：base/cosine 的 e1 差异必须落在同配置噪声本底内 ===="
# 口径与 CPU 版不同（CPU 版要求逐位一致），原因：cudnn.benchmark 靠「实测耗时」给
# 每种卷积形状挑最快算法，掐表本身有噪声，两次进程可能选中不同实现 → 浮点累加顺序
# 不同 → 结果在 1e-4 量级漂移（4090 + torch 2.8.0 实测 Δloss ~2e-4、Δacc ~0.1pp）。
# 所以 GPU 上「跨进程逐位一致」不可达，检查改成两步：
#   1) 同配置复跑一遍 (dup)，量出本机运行间噪声本底，顺带确认它没大到离谱；
#   2) 要求 base vs cosine（第 1 轮 lr 同为初始值 0.01，理论上是同一计算过程）
#      的差异不超过「2 倍本次实测噪声」（下限 Δloss 0.005 / Δacc 0.5pp 兜底）——
#      seed 真没控住时（shuffle 顺序变了）差异会大一到两个数量级。
# 实测本底本身会随机波动（4090 上 0.1~0.4pp 都出现过），容差跟着本次测量走，
# 不写死一个数，否则本底偏大的一天会误报。
for t in smk_base smk_dup smk_cos; do
  extra=""
  [ "$t" = "smk_cos" ] && extra="--scheduler cosine"
  python -u train.py --model vgg --epochs 1 --limit-batches 20 \
    --tag "$t" --out "$OUT" $extra > /dev/null
done
python - <<'PY'
import json
def e1(tag):
    return json.load(open(f"/tmp/smoke_gpu/{tag}_vgg.json"))["history"][0]

NOISE_FLOOR_LOSS, NOISE_FLOOR_ACC = 0.005, 0.5   # 本底下限：噪声测得极小时的兜底
SANITY_LOSS, SANITY_ACC = 0.01, 1.0              # 本底天花板：超过说明本机噪声异常
base, dup, cos = e1("smk_base"), e1("smk_dup"), e1("smk_cos")
noise = (abs(base["train_loss"] - dup["train_loss"]),
         abs(base["test_acc"] - dup["test_acc"]))
diff = (abs(base["train_loss"] - cos["train_loss"]),
        abs(base["test_acc"] - cos["test_acc"]))
print(f"  baseline    e1: loss={base['train_loss']:.6f} acc={base['test_acc']:.4f}")
print(f"  同配置复跑  e1: loss={dup['train_loss']:.6f} acc={dup['test_acc']:.4f}")
print(f"  cosine      e1: loss={cos['train_loss']:.6f} acc={cos['test_acc']:.4f}")
print(f"  同配置噪声本底:  Δloss={noise[0]:.6f}  Δacc={noise[1]:.2f}pp")
print(f"  base vs cosine:  Δloss={diff[0]:.6f}  Δacc={diff[1]:.2f}pp")
if noise[0] > SANITY_LOSS or noise[1] > SANITY_ACC:
    raise SystemExit("!! 同配置两次跑差异就异常大，本机数值噪声失控，先排查再继续")
# base/cos 在零假设下与 dup 同分布，容差取 2 倍本次实测噪声（下限兜底），不写死
ok = (diff[0] <= max(2 * noise[0], NOISE_FLOOR_LOSS)
      and diff[1] <= max(2 * noise[1], NOISE_FLOOR_ACC))
print("  OK  随机性已控住（base/cos 差异落在同配置噪声本底内）" if ok
      else "  !! base/cos 差异远超噪声本底，seed 没控住")
raise SystemExit(0 if ok else "随机性自检失败")
PY

echo ""
echo "==== 冒烟测试全部通过，可以开始正式实验 ===="
echo "下一步：bash run_all.sh 5 baseline"
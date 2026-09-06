"""把 MNIST 的 IDX(ubyte) 原始文件转成可以直接打开看的图片。

为什么不用 torchvision 读：IDX 的头部只有 8/16 字节，自己解析一遍就能看清
"这堆二进制到底是怎么排的"，而 datasets.MNIST 会把这一层完全藏起来。

IDX 格式（来自 http://yann.lecun.com/exdb/mnist/）：

  图片文件 idx3-ubyte，16 字节头 + 像素数据
    offset 0  4字节  magic number = 0x00000803(2051)  最后一字节 3 = 三个维度
    offset 4  4字节  图片张数     60000 / 10000
    offset 8  4字节  行数         28
    offset 12 4字节  列数         28
    offset 16 起      每字节一个像素，0=黑 255=白，按行优先铺开，无分隔符

  标签文件 idx1-ubyte，8 字节头 + 标签数据
    offset 0  4字节  magic number = 0x00000801(2049)  最后一字节 1 = 一个维度
    offset 4  4字节  标签个数
    offset 8  起      每字节一个标签，取值 0-9

所有整数都是**大端序**（high endian），这是 IDX 格式的规定，
所以下面用 ">i4" 而不是本机默认的小端。

用法：
    python export_images.py                      # 测试集前 100 张，拼成一张总览图
    python export_images.py --split train -n 64  # 换训练集，取 64 张
    python export_images.py --mode files -n 50   # 另存为 50 个独立 png
    python export_images.py --mode both -n 100   # 两种都要
    python export_images.py --digit 7 -n 36      # 只看数字 7
"""

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path("data/MNIST/raw")
FILES = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}


def read_images(path):
    """读 idx3-ubyte，返回 (N, 28, 28) 的 uint8 数组。"""
    with open(path, "rb") as f:
        # ">IIII" = 大端序的 4 个 uint32，正好吃掉 16 字节的头
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise SystemExit(f"{path} 的 magic 是 {magic}，不是图片文件（应为 2051）")
        # 头之后全是像素，一次读完再 reshape 成 N 张 28x28
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n * rows * cols:
        raise SystemExit(f"{path} 像素数不符：期望 {n * rows * cols}，实际 {data.size}")
    print(f"  {path.name}: magic={magic} 张数={n} 尺寸={rows}x{cols}")
    return data.reshape(n, rows, cols)


def read_labels(path):
    """读 idx1-ubyte，返回 (N,) 的 uint8 数组。"""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))   # 标签文件头只有 8 字节
        if magic != 2049:
            raise SystemExit(f"{path} 的 magic 是 {magic}，不是标签文件（应为 2049）")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n:
        raise SystemExit(f"{path} 标签数不符：期望 {n}，实际 {data.size}")
    print(f"  {path.name}: magic={magic} 个数={n}")
    return data


def make_grid(images, labels, cols, scale, pad=2):
    """把 N 张图拼成一张总览图（contact sheet），标签写在文件名里说明不了，
    所以这里改用留白分隔 + 返回排列顺序，由调用方另存一份标签清单。"""
    n = len(images)
    rows = (n + cols - 1) // cols
    h = w = images.shape[1]
    # 底色用 30 而不是 0：纯黑背景下分不清"图片边缘"和"数字里的黑像素"
    canvas = np.full((rows * (h + pad) - pad, cols * (w + pad) - pad), 30, np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas[r * (h + pad):r * (h + pad) + h, c * (w + pad):c * (w + pad) + w] = img
    out = Image.fromarray(canvas, mode="L")
    if scale > 1:
        # NEAREST 而不是默认的双线性：28x28 放大时要看清像素本身，不要被插值抹平
        out = out.resize((out.width * scale, out.height * scale), Image.NEAREST)
    return out, rows


def main():
    p = argparse.ArgumentParser(
        description="把 MNIST 的 ubyte 原始文件导出成可查看的图片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("-n", "--count", type=int, default=100, help="导出多少张")
    p.add_argument("--offset", type=int, default=0, help="从第几张开始")
    p.add_argument("--digit", type=int, default=-1, choices=range(-1, 10),
                   help="只导出某个数字，-1 表示不筛选")
    p.add_argument("--mode", default="grid", choices=["grid", "files", "both"],
                   help="grid=拼一张总览图；files=每张单独存 png")
    p.add_argument("--cols", type=int, default=10, help="总览图每行几张")
    p.add_argument("--scale", type=int, default=3, help="放大倍数，28x28 太小看不清")
    p.add_argument("--raw", default=str(RAW), help="ubyte 文件所在目录")
    p.add_argument("--out", default="./results/mnist_png")
    args = p.parse_args()

    img_name, lbl_name = FILES[args.split]
    raw = Path(args.raw)
    print(f"解析 IDX 头部：")
    images = read_images(raw / img_name)
    labels = read_labels(raw / lbl_name)

    idx = np.arange(len(images))
    if args.digit >= 0:
        idx = idx[labels == args.digit]
        print(f"筛选数字 {args.digit}：共 {len(idx)} 张")
    idx = idx[args.offset:args.offset + args.count]
    if len(idx) == 0:
        raise SystemExit("筛选/偏移之后没有剩下任何图片")
    sel_img, sel_lbl = images[idx], labels[idx]
    print(f"导出 {len(idx)} 张（{args.split} 集，原始下标 {idx[0]}..{idx[-1]}）")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}" + (f"_digit{args.digit}" if args.digit >= 0 else "")

    if args.mode in ("grid", "both"):
        grid, rows = make_grid(sel_img, sel_lbl, args.cols, args.scale)
        gpath = out / f"grid_{suffix}_{len(idx)}.png"
        grid.save(gpath)
        print(f"\n总览图已保存 {gpath}  （{rows} 行 x {args.cols} 列，{grid.width}x{grid.height}px）")
        # 图上没法写字（避免引入字体依赖），所以标签按同样的排列另存一份文本
        tpath = out / f"grid_{suffix}_{len(idx)}_labels.txt"
        lines = [" ".join(f"{v}" for v in sel_lbl[r * args.cols:(r + 1) * args.cols])
                 for r in range(rows)]
        tpath.write_text("\n".join(lines) + "\n")
        print(f"对应标签已保存 {tpath}（排列与图一致）")
        print("\n标签速览：")
        for line in lines:
            print("  " + line)

    if args.mode in ("files", "both"):
        fdir = out / suffix
        fdir.mkdir(parents=True, exist_ok=True)
        for i, (im, lb) in zip(idx, zip(sel_img, sel_lbl)):
            # 文件名带上真实下标和标签，方便回头对照原始数据集
            Image.fromarray(im, mode="L").save(fdir / f"{i:05d}_label{lb}.png")
        print(f"\n{len(idx)} 个独立 png 已保存到 {fdir}/")

    print(f"\nmacOS 下直接看：open {out}")


if __name__ == "__main__":
    main()

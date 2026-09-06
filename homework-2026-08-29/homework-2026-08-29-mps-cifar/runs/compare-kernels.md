| run | 模型 | 参数量 | 配置 | 最优测试精度 | 末轮测试精度 | 末轮训练精度 | 秒/epoch | 总耗时(min) |
|---|---|---|---|---|---|---|---|---|
| alexnet3x3 | alexnet3x3 | 6.98M | sgd lr=0.1 cosine | 0.8626 (@10) | 0.8626 | 0.8681 | 56.8s | 9.5 |
| baseline-alexnet | alexnet5x5 | 7.18M | sgd lr=0.1 cosine | 0.8484 (@10) | 0.8484 | 0.8594 | 68.1s | 11.4 |
| baseline-vgg16 | vgg16 | 14.72M | sgd lr=0.1 cosine | 0.7656 (@10) | 0.7656 | 0.7674 | 92.6s | 15.5 |

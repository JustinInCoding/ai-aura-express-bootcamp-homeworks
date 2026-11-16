### 做 EDA 计划

@homework-2026-11-14/douban-comments/dataset/comments.csv @homework-2026-11-14/douban-comments/dataset/movies.csv  帮我写一个分析这个数据集的 EDA 方案，方案要从统计和可视化的方向来展示数据的分布，让我更容易理解数据的分布情况，每个数据的展示下面都要给具体的分析

我想做评论的打分模型，帮我基于此进行方案设计，只写方案，不落地，把方案写在  homework-2026-11-14/douban-comments/docs 目录下 （有具体目标的情况）

### 编写 EDA 流程 jupyter notebook

@/homework-2026-11-14/douban-comments/docs/01-eda-plan.md 针对于现有方案，帮我写一个 jupyter notebook 的文档，并且帮我运行一遍，放在 homework-2026-11-14/douban-comments 目录下

### 运行不起来

帮我在 @homework-2026-11-14/douban-comments 目录下新建一个 .venv 环境，然后安装 jieba, wordcloud 等依赖，然后运行

```bash

cd homework-2026-11-14/douban-comments
python3 -m venv .venv
.venv/bin/pip install jupyter pandas numpy matplotlib jieba wordcloud
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb

```

### 常用统计学可视化图表

常用的统计学中可视化展示的图形类型有哪些，每种类型都用来做什么，怎么看，帮我写一个 jupyter notebook 来进行详细说明

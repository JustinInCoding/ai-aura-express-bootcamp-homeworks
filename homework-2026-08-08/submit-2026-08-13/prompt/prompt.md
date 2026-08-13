## 提示词

### 任务描述

当前 `../data` 目录下有关于 pugb 的数据集，其中 `train_V2.csv` 是训练集数据, `test_V2.csv` 是测试集数据。每个样本包含以下特征：
- `Id`: Player’s Id 
- `groupId`: ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
- `matchId`: ID to identify matches. There are no matches that are in both the training and testing set.
- `assists`: Number of enemy players this player damaged that were killed by teammates.
- `boosts`:  Number of boost items used.
- `damageDealt`: Total damage dealt. Note: Self inflicted damage is subtracted.
- `DBNOs`: Number of enemy players knocked
- `headshotKills`: Number of enemy players killed with headshots.
- `heals`: Number of healing items used.
- `killPlace`: Ranking in match of number of enemy players killed.
- `killPoints`: Kills-based external ranking of players. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
- `kills`: Number of enemy players killed.
- `killStreaks`: Max number of enemy players killed in a short amount of time.
- `longestKill`: Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
- `matchDuration`: Duration of match in seconds.
- `matchType`: String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
- `maxPlace`: Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
- `numGroups`: Number of groups we have data for in the match.
- `rankPoints`: Elo-like ranking of players. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes the place of “None”.
- `revives`: Number of times this player revived teammates.
- `rideDistance`: Total distance traveled in vehicles measured in meters.
- `roadKills`: Number of kills while in a vehicle.
- `swimDistance`: Total distance traveled by swimming measured in meters.
- `teamKills`: Number of times this player killed a teammate.
- `vehicleDestroys`: Number of vehicles destroyed.
- `walkDistance`: Total distance traveled on foot measured in meters.
- `weaponsAcquired`: Number of weapons picked up.
- `winPoints`: Win-based external ranking of players. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.

预测玩家的排名（1-第一名， 0-最后一名）：
- winPlacePerc

### 任务要求

- 使用 jupyter notebook 来完成任务
- 确定一种机器学习方法（随机森林，决策树，线性回归）, 需要做数据处理，包括缺失值处理、异常值处理、特征工程等, 训练后，需要评估模型 score 分数 > 0.925, MAE < .059
- 再确定一种机器学习方法（随机森林，决策树，线性回归）来做
- 两种机器学习方法再经过超参数优化、特征工程、数据预处理等不同操作之后的准确性差距对比
- 最后总结在 PUBG 比赛中获胜，哪些特征因素比较重要

### 任务计划
- 先不着急做，首先基于你的判断，先确定一个算法
- 然后关于数据预处理、特征工程、也需要列出
- 然后将整体计划写出来保存到 `plan.md` 中

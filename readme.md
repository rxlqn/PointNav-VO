## 1_28
- 现在训练集用的val_mini所以最多并行环境只有3个，现在用的是3个
- 4卡同时训练
- 评估时候也要改数据集名称

## 1_31
- 修改launch让参数默认话从而可以调试。launch只是用来并行的初始化文件，所以应该直接跑run.py
- _collect_rollout_step 中与环境交互
- ddppo-trainer.py中train进行更新
## 1_28
- 现在训练集用的val_mini所以最多并行环境只有3个，现在用的是3个
- 4卡同时训练
- 评估时候也要改数据集名称

## 1_31
- 修改launch让参数默认话从而可以调试。launch只是用来并行的初始化文件，所以应该直接跑run.py
- _collect_rollout_step 中与环境交互
- ddppo-trainer.py中train进行更新

## 2_4
- 推理和训练流程基本上搞清楚了，现在测试下GRU的性能怎么样
- 把CHECKPOINT_INTERVAL调大
- 要改rollout?
- 改策略中的视觉编码器部分和状态编码器
- 他现在是step是连着的，我如何能做到切割原来的step为动态的呢，这样效率会不会降低呢，会的，不同环境的
  每个episode数量是不一样的所以不能这样直接切割。
  - 现在的训练流程是收集完成训练
  - 最重要的是要在state_encoder中加入mask_hidden, 还不够，因为如果中间被截断之前的状态就看不到了。
    但是如果长度不统一就不能并行同时采集多个环境了。。但是可以4张卡同时用也不是不行？ddppo本身支持长度不同时候的截断？
  - 目前的训练框架难以捕捉到完整的trajectory，所以不一定适合transformer
    - 要修改目前的训练框架从而获得完整的训练轨迹。两种修改方式，并行探索再遍历训练。还有可以对短的轨迹进行补零操作
  - 如果实在不行可以先只采集一个环境的信息

## 2_5
- 目前看来由于他没有一个实时的GPS，所以他预训练了一个视觉里程计，并且用这个视觉里程计去估计终点到相机的极坐标
- 他的VO模块只是相当于一个插件，由于竞赛不提供GPS数据所以需要一个VO来估计这个数据，可能可以替代他的VO工具
  - 目前可以修改的是可以进行实时的训练更新VO。
  - VO的视觉编码器和policy的没有关系
    - 所以有两个地方可以改，一个是VO部分，一个是policy部分的视觉编码器
    - inspire.ai 的方法把视觉里程计和policy合在了一起，并且都分别加上了lstm要回归历史信息
- sync_frac是用来同步多GPU之间的梯度
- 他原来的RL-policy的输入只有depth？？ 跑了一个rgb depth输入的
- 视觉里程计的方法应该在这个任务中是必不可缺的，目前他们的仅仅使用LSTM就达到了很好的效果所以LSTM应该是有用的。
  - 现在觉得应该先合并VO和policy的视觉模型达到更好的效果后面再进行修改才好

## 2_7
- 要搞清楚他的传感器到底是怎么回事
- 他的policy网络和critic网络都只有一层，感觉高度耦合CategoricalNet，CriticHead
- 可以先复现下inspire他们的比赛结果，目前有一个很好的视觉里程计？视觉网络没有很好的监督信号？
- 测试一个num_steps为32的，num_env, batch为3的情况，避免lstm太长难以收敛, 并且critic和policy网络各加一层
    - 虽然batchsize增大，但是因为把step降下来了所以显存占用也小了
    - 因为num_step下了，所以更新的更频繁了
- 现在看上去更不稳定了，怀疑是batch太小了不稳定？以前是每128更新一次，现在是每32更新一次,环境变化剧烈的情况下应该步数大一点比较合适
    - 如果想单独改lstm训练长度可以单独修改？
      - 这里更正确的修改方法应该是在RNNstateencoder里面直接修改成把时间序列截断，这样其他就没有干扰了
      - 如果要截断还要pred value也随着序列进行变化

## 2_9
- 重写了GRU 并且返回了index对准了相应的东西包括action在内的
- 后面还可以加图像增强
    
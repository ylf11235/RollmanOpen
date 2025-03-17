此仓库包含对Rollman智能体比赛中的录像对局进行Behavior Cloning所用到效果较好的网络。
训练数据来源主要是 omegafantasy かずさ v6 这个版本，约1500局（15万步）训练加上数据增强可消除过拟合。
正常使用常用的Resnet或者Alphago范式的网络都只能达到75%的准确率且收敛极慢。目前A2Cv2_0312经过10万个iterations动作准确率能收敛到93%-95%。
实测能力（pac_reward）只能达到原版的30%左右。

1. A2Cv2_0312.py包含完整网络结构。
2. GymEnvironment.py里提供了官方sdk下开发的member function，它能从env中获取网络输入需要的两个tensor。

为保证比赛公平性只开源这两个部分。
未开源部分包括：
· 爬取数据
· 数据预处理
· 训练代码（包含数据增强）
· inference代码
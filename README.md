# deep_learning(代码持续优化ing...)  
main.py->入口  
model.fnn.my_fnn.py->纯numpy从0实现前馈神经网络前向传播-反向更新代码，并实现了交叉熵+softmax，adamW正则化  
model.cnn.my_cnn.py->纯numpy从0实现卷积神经网络前向-反向更新代码。
类似pytorch或tensorflow，我将Conv2d, Dense, Flatten, AdamW分别封装为实现类，
损失函数同样使用交叉熵+softmax。
并且，网络可以对每层结构(Conv2d|Dense)定义不同的正则化项  
model.cnn.pytorch_cnn.py|tensorflow_cnn.py->分别使用现有框架实现的cnn网络，
作用是对比在相同参数下，我自己实现的卷积网络的正确性和训练效果

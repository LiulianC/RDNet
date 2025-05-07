## 关键文件的关键关系
train.py 训练大循环
engine.py 定义了训练 验证 测试 三种逻辑
base_model.py 是cls_model_eval_reg.py的基类 做一些参数和gpu环境的配置
cls_model_eval_nocls_reg.py 是一个抽象model，管理三个net：net_i net_c netD
net_c 就是 models/arch/classifier.py 是一个pretrained的分类器 6分类 相当于是通道注意力之类的
net_i 就是 先用models/arch/focalnet.py生成多尺度图像特征后 再用 models/arch/RDnet_.py 处理 RDnet.py的末尾使用decoder.py 还原图像 
netD  没有被使用 它是networks.py 里面的 可以被选择的各类模型

保存的只有net_i的参数 net_c的参数属于是请外援 事先加载了 是pretrained
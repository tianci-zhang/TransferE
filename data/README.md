数据说明：
四个文件夹分别存储FB15k、FB15k-237、WN18和WN18RR数据集的数据
每个文件夹均包含以下文件：
entity2id.txt：存储实体的ID字典
relation2id.txt：存储关系的ID字典
train.txt：训练数据集，存储三元组形式为(h,t,r)，r放在第三列
test.txt：测试数据集
valid.txt：验证数据集

每次训练时用训练集的数据进行训练，每训练一个epoch用测试集做测试。
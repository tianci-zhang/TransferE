# basic-TrmE

#### 代码说明


#### main.py：主函数，也是跑程序时直接调用的函数


```
main()     #设置模型参数大小并掉用train()进行训练和测试          
train()                   
test()
```

####  load_data.py：用于处理和加载数据

```
load_data()        #从数据集加载数据
Data_Loader()      #得到输入模型的三元组序列
X_MASK_Train()     #为三元组序列输入做MASK操作，通过训练得到对MASK的实体/关系的预测
X_MASK_Test()
```


 
#### Model.py:  定义TrmE模型

```
class TrmE             
class Block
class BlockLayer
```
                   
   

#### SubLayer.py:  定义每个block里的两层 
```
class MultiHeadedAttention        #多头注意力层
class PositionwiseFeedForward     #前向层
 class SublayerConnection         #残差和层归一化

```
                     
             
#### Mudules.py： 其他层
```
class Predict        #接在TrmE encoder之后做预测的线性层和softmax层 
class LabelSmoothing  #对标签做平滑
class NoamOpt         #优化函数
```                    

#### Evaluation.py：对测试集的预测结果做评估
```  
Evaluation()                        
Evaluation_Head()             #得到真正的头实体在所有实体中概率分布的排名
Evaluation_Tail()             #得到真正的尾实体在所有实体中概率分布的排名
Evaluation_Relation()         #得到真正的关系在所有关系中概率分布的排名
Print_mean_rank()             #根据排名计算评估指标值
```  

#### 如何跑代码：

1. 在main.py的main()函数中定义好训练任务、模型参数以及存储参数和实验结果的路径，例如：
```
task = "relation-pre"
d_model = 256
d_ff = 512
dropout = 0.1 
N = 6 
h = 4 
triple_length = 64 
d_per = d_model//h
num_epoch = 50
batch_size = 256
batch_split = 1 
data_dir = "./data/FB15k"
checkpoints_path = "/home1/zhangtc/trme/FB15k-try/relation-pre_2" 
 result_path = "/home/zhangtc/TrmE/trme/FB15k-try/relation-pre_2" 
label_smoothing = 0.0
seed = 0
warmup = 10000
factor = 1
mask_percent = 0.15
```

2. ```python main.py ``` （linux下可用： ```nohup python -u main.py > results.out 2>&1 &```）

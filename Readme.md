## Transformer pytorch实现  
> 笔记

`setting.py`:模型相关参数，文件目录的配置文件。  
`utils.py`:一些工具函数，比如不等长句子的padding等等。  
`data_pre.py`:数据的预处理，得到输出模型的batch数据和相关的mask矩阵  
`model.py`:模型文件。通过调用**make_model方法**传入相关模型初始化参数，来对模型进行初始化。  
`train.py`:进行模型的训练。和最好模型的保存。  
`test.py`:对测试集句子的测试输出。  
`bleu_score.py`:对机器翻译评分。  
`one_trans.py`:实现单个句子进行翻译。  
`app.py`:通过使用one_trans文件封装的单个句子翻译的方法，实现flask api  


### flask api请求参数
> 简单api，没有进行检查校验和异常处理
```json
// POST请求参数
{
  "sentence": "your  translation sentences"
}
// return
{
  "result": "翻译结果",
  "msg": 'success',
  "code": 200
}
```

### 模型训练数据
使用`14533`条翻译数据进行训练。  
数据文件格式：en`\t`cn
  

### 结果评估
使用BLEU算法进行翻译效果评估[BLEU](https://www.cnblogs.com/by-dream/p/7679284.html)
BLEU算法评价结果：  
    
    对399条翻译句子效果进行评估
    验证集:0.1075088492716548，n-gram权重：(1,0,0,0)
          0.03417978514554449,n-gram权重：(1,0.2,0,0)
          
*Attention：运行代码之前需要自己在项目目录下新建一个save文件夹*  
[PyTorch官方Transformer接口](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)   

[Transformer讲解](https://thenet.top/index.php/archives/45/)


## 运行项目
1. `python train.py`：训练模型，保存模型
2. `python app.py`启动服务。
> `python test.py`，测试模型的测试集上的效果(这里用的是验证集)
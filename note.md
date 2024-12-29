#### 架构设计
- Modal(模型类)：定义模型细节，以及如何构建batch
- Reader(帮助类)：从文件中读入数据，进行共性处理
- Runner(帮助类)：控制模型训练和评测
<img src='./架构.png'>



#### 修改地区
1. 改动了basemodel中的np.object为object

# IRAG

## Overview
- 相比较之前的代码，主要修改了src/dataset src/model, IRAG_inference.py, train_IRAG.py代码。

## Dataset部分
- 目前只上传了dataset处理的代码，包括expla_graphs, scene_graphs和webqsp。
- 其中没有后缀的是用于训练GRAG的，没有变动。后缀带v2的是用于训练IRAG的（暂时只需要看webqsp的），在v2文件中有两个dataset，分别是WebQSPDataset和EdgePairDataset。
前者是以graph为粒度，后者是以节点对为粒度。
- 在训练过程中导入的是EdgePairDataset，在EdgePairDataset生成的时候将WebQSPdataset作为其“base_dataset”，然后从base_dataset中筛选特定数量的节点对进行返回。
- WebQSPDatast限制正样本的大概逻辑是：1.设定max_positive值 2.从已有的正样本集合中随机采样max值数量的正样本对 3.设置负样本采样数量（按倍数），然后采样对应数量的负样本。

## Model部分
- 主要添加了completion_model.py 和 LP.py两个代码
- 对于completion_model.py，逻辑和GRAG基本一致。在gpu的分配上，发现completion_model和LP model不能同时放在一张卡上，要不然会OOM或者数据模型不在一张卡上，所以代码中completion model放在了三张卡上
- 对于LP model，就是正常的GNN模型，我认为似乎不是造成性能瓶颈的原因

## Train/Inference部分
- train_GRAG.py几乎没有修改
- 对于train_IRAG.py，1.completion model进行预测，得到节点对预测的text_attr 2. 将新预测的text_attr更新到原来的graph当中，得到新的batch 3. 将新的batch放入LP中得到loss
- 对于IRAG_inference.py代码，1. completion model进行预测，生成新的text_attr，并更新到原来的graph当中 2. LP对新的graph进行评分，如果分数高于threshold，那么保留这个text_attr，否则不改动原来的graph。3.
当所有节点对预测完毕并且LP核验完成后，重新生成新的embedding和desc，构建新的Dataset和Dataloader 4. 由LLM进行generation。
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    配置参数
    """

    def __init__(self, dataset):
        self.model_name = 'Bert'
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # 校验集
        self.dev_path = dataset + '/data/dev.txt'
        # 类别
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # 模型训练结果
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 类别数
        self.num_classes = len(self.class_list)
        # 若超过1000batch效果还没有提升（相当于从数据集中取了1000次数据），则提前结束训练
        self.require_improvement = 1000
        # 把数据集迭代几次
        self.num_epochs = 3
        # 每次取的数据个数
        self.batch_size = 128
        # 每句话的处理长度（短填，长截）
        self.pad_size = 32
        # 学习率（0.00001）
        self.learning_rate = 5e-5
        # bert 预训练的模型位置
        self.bert_path = 'bert_pretrain'
        # bert 的分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert 隐藏层个数
        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            # 参数微调
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x[ids, seq_len, mask]
        context = x[0]  # 对应输入的句子 shape[128,32]
        mask = x[2]  # 对句子的padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]， shape[128,32]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)  # shape[128,768]
        out = self.fc(pooled)  # [128,10]
        return out

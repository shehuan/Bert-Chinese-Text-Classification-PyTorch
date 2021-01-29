from tqdm import tqdm
import torch
import time
import datetime
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    """
    返回值 train, dev, test
    每个返回值里有4个list: ids、label、ids_len、mask
    :param config:
    :return:
    """

    def load_dataset(file_path):
        """
        返回结果：
        :param file_path:
        :return:
        """
        contents = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                # 去除头尾空格
                line = line.strip()
                if not line:
                    continue
                content, label = line.split("\t")
                # 分词
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                pad_size = config.pad_size
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
                        # 句子长度小于pad_size，末尾补0
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test


class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        # 取完数据需要的次数
        self.n_batches = len(dataset) // batch_size
        # 当前已取的次数
        self.index = 0
        # 记录batch数量是否为整数
        self.residue = False
        if len(dataset) % self.n_batches != 0:
            self.residue = True

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
        return batches

    def _to_tensor(self, batches):
        # 样本数据 ids
        x = torch.LongTensor([item[0] for item in batches]).to(self.device)
        # 标签数据 label
        y = torch.LongTensor([item[1] for item in batches]).to(self.device)
        # 每个序列的真实长度
        seq_len = torch.LongTensor([item[2] for item in batches]).to(self.device)
        mask = torch.LongTensor([item[3] for item in batches]).to(self.device)
        return (x, seq_len, mask), y

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

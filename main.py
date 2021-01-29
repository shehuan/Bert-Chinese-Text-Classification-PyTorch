import time
import torch
import numpy as np
from importlib import import_module
import argparse
import utils
import train

parser = argparse.ArgumentParser(description='Bruce-Bert-Text-Classification')
parser.add_argument('--model', type=str, default='BruceBert', help='choose a model')
args = parser.parse_args()

if __name__ == '__main__':
    # 数据集地址
    dataset = 'THUCNews'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    # 保证每次运行结果一致
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print('开始加载数据集')
    train_data, dev_data, test_data = utils.build_dataset(config)

    train_iter = utils.build_iterator(train_data, config)
    dev_iter = utils.build_iterator(dev_data, config)
    test_iter = utils.build_iterator(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print("准备数据的时间：", time_dif)

    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)

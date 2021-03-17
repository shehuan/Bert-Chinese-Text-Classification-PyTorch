import random

data_list = []
file_dir = "THUCNews/data2/"
with open(file_dir + "original.txt", encoding='utf-8') as file:
    for line in file:
        data_list.append(line)
# 打乱分类好的数据
random.shuffle(data_list)
random.shuffle(data_list)
random.shuffle(data_list)
# 将数据按照1:1:8分为测试集、验证集、训练集
n = len(data_list) // 10
test_list = data_list[:n]
dev_list = data_list[n:n * 2]
train_list = data_list[n * 2:]
# 将测试集数据写入对应的文件
test = open(file_dir + "test.txt", 'w', encoding='utf-8')
for line in test_list:
    test.write(line)
# 将验证集数据写入对应的文件
dev = open(file_dir + "dev.txt", 'w', encoding='utf-8')
for line in test_list:
    dev.write(line)
# 将训练集数据写入对应的文件
train = open(file_dir + "train.txt", 'w', encoding='utf-8')
for line in test_list:
    train.write(line)

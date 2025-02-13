import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import os
import argparse
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from scipy.stats import bootstrap
import openpyxl
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay, auc, roc_auc_score
from Model import *
from LoadData import *

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--mod', default='EEGnet', type=str, help='model')
parser.add_argument('--data', '--dataset', default='wkh', type=str, help='dataset')
args, unknown = parser.parse_known_args()

def seed_torch(seed=1000):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def f1_statistic(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def ba_statistic(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # 默认从环境变量获取 GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}, CUDA_VISIBLE_DEVICES: {gpu_id}")

def create_model():
    if args.mod == 'EEGNet':
        model = EEGNet()
    elif args.mod == 'PLNet':
        model = PLNet()
    elif args.mod == 'PPNN':
        model = PPNN()
    elif args.mod == 'EEGInception':
        model = EEGInception()
    elif args.mod == 'LMDA':
        model = LMDA()
    elif args.mod == 'Conformer':
        model = Conformer()
    return model


def Train():

    model = create_model().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    # file_path = "./data/mkl.h5"
    file_path = "./data/%s.h5"%args.data

    # print(file_path)
    data = h5py.File(file_path, 'r')
    data_train = data['data_train']
    label_train = data['label_train']
    data_test = data['data_test']
    label_test = data['label_test']
    data_train = np.array(data_train)
    label_train = np.array(label_train)
    data_test = np.array(data_test)
    label_test = np.array(label_test)

    cont = 0
    for i in range(len(label_train)):
        if label_train[i] == 3:
            cont += 1
            label_train[i] = 1
        label_train[i] = label_train[i] - 1
    for i in range(len(label_test)):
        label_test[i] = label_test[i] - 1

    # print(label_train)
    # print(label_test)

    data_train = scale_data(data_train)
    data_train = torch.from_numpy(data_train).to(torch.float32).unsqueeze(1)
    label_train = torch.from_numpy(label_train).to(torch.int64)
    train_loader = DataLoader(DiabetesDataset(data_train, label_train), batch_size=256, shuffle=True)

    data_test = scale_data(data_test)
    data_test = torch.from_numpy(data_test).to(torch.float32).unsqueeze(1)
    label_test = torch.from_numpy(label_test).to(torch.int64)
    test_loader = DataLoader(DiabetesDataset(data_test, label_test), batch_size=1024, shuffle=False)

    best = 0.33
    train_len = len(label_train)

    # for epoch in tqdm(range(250)):
    for epoch in range(250):
        model.train()
        train_acc = 0
        label_collect = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            # print("Model output shape:", out.shape)
            # print("Target shape:", y.shape)

            loss = criterion(out, y)

            prediction = torch.argmax(out, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += (prediction == y).sum().float()

            label_collect.append(y)


        temp_acc = train_acc / train_len


    model.eval()
    correct_num = 0
    model.eval()
    label = np.zeros((0))
    pre = np.zeros((0))
    prob = np.zeros((0))

    for index, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)

        y_pred = model(x)
        probabilities = torch.sigmoid(y_pred).detach().to("cpu").numpy()  # 使用 sigmoid 转为概率
        positive_probs = probabilities[:, 1]
        _, pred = torch.max(y_pred, 1)
        out = torch.argmax(y_pred, dim=1).detach().to("cpu").numpy()

        correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
        label = np.concatenate((label, y.detach().to("cpu").numpy()))
        pre = np.concatenate((pre, out))
        prob = np.concatenate((prob, positive_probs))

    # Bootstrap
    n_iterations = 1000
    n_size = len(test_loader.dataset)  # 确保 test_loader.dataset 是一个包含测试集数据的对象
    f1_arr = []
    ba_arr = []

    for _ in range(n_iterations):
        indices = np.random.choice(range(n_size), n_size, replace=True)
        boot_label = label[indices]
        boot_pre = pre[indices]
        f1 = f1_score(boot_label, boot_pre, average='macro')
        ba = balanced_accuracy_score(boot_label, boot_pre)
        f1_arr.append(f1)
        ba_arr.append(ba)

    excel_file = "./data/unique_names_with_positions.xlsx"
    extracted_predictions = pd.read_excel(excel_file)
    sides = extracted_predictions['side'].values
    new_label = []
    new_pre = []
    for side_value in sides:
        # 根据side值获取对应的标签与预测
        # print("%s_%s" % (side_value, label[side_value]))
        new_label.append(label[side_value])
        new_pre.append(pre[side_value])
    new_label = np.array(new_label)
    new_pre = np.array(new_pre)
    f1_new = f1_score(new_label, new_pre, average='macro')
    ba_new = balanced_accuracy_score(new_label, new_pre)

    # 计算均值和标准差
    f1_mean = np.mean(f1_arr)
    f1_std = np.std(f1_arr)
    ba_mean = np.mean(ba_arr)
    ba_std = np.std(ba_arr)

    # 输出结果
    # print(f"F1-score (macro): {f1_mean:.4f} ± {f1_std:.4f}")
    # print(f"Balanced Accuracy: {ba_mean:.4f} ± {ba_std:.4f}")

    acc = accuracy_score(label, pre)
    f1 = f1_score(label, pre, average='macro')
    ba = balanced_accuracy_score(label, pre)
    auc = roc_auc_score(label, prob)

    # print("acc", acc)
    # print("f1_score", f1_score(label, pre, average='micro'))
    # print("f1_score macro", f1)
    # print("BA_test", balanced_accuracy_score(label, pre))
    # print("auc", auc)
    # print(confusion_matrix(label, pre))
    # with open(file_path, 'a') as file:
    #     file.write(f"F1-score (macro): {f1_mean:.4f} ± {f1_std:.4f} Balanced Accuracy: {ba_mean:.4f} ± {ba_std:.4f}\n")
    return auc, f1, ba, pre, prob, f1_new, ba_new

Best_ba = 0  # 初始化最佳平衡准确率
pre_file_path = "./excel/Pred/%s_%s.xlsx"%(args.mod, args.data)
prob_file_path = "./excel/Prob/%s_%s.xlsx"%(args.mod, args.data)
data_file_path = "./excel/Data/%s_%s.xlsx"%(args.mod, args.data)

results = pd.DataFrame(columns=["Model", "Subject", "AUC", "F1", "Balanced Accuracy", "Sample Balanced Accuracy", "Sample F1"])

for i in range(15):
    auc, f1, balanced_accuracy, pre, prob, f1_new, ba_new = Train()  # 假设 Train 返回这些结果
    results.loc[i] = [args.mod, args.data, auc, f1, balanced_accuracy, ba_new, f1_new]
    if ba_new > Best_ba:
        Best_ba = ba_new

        n_rows = 605
        pre_array = np.array(pre)
        prob_array = np.array(prob)
        num_cols = (len(pre_array) + n_rows - 1) // n_rows  # 计算需要的列数
        pre_data = []
        prob_data = []

        for col in range(num_cols):
            start_idx = col * n_rows
            end_idx = min((col + 1) * n_rows, len(pre_array))
            pre_data.append(pre_array[start_idx:end_idx])
            prob_data.append(prob_array[start_idx:end_idx])

        # 将切片后的数据转为 DataFrame 并填充空白
        pre_df = pd.DataFrame(pre_data).T
        prob_df = pd.DataFrame(prob_data).T
        pre_df.to_excel(pre_file_path, index=False, header=[f"Column {i+1}" for i in range(num_cols)])
        prob_df.to_excel(prob_file_path, index=False, header=[f"Column {i+1}" for i in range(num_cols)])

summary = results[["AUC", "F1", "Balanced Accuracy", "Sample Balanced Accuracy", "Sample F1"]].agg(["mean", "std"])
# 格式化均值 ± 标准差
formatted_summary = summary.loc["mean"].apply(lambda x: f"{x:.4f}") + " ± " + summary.loc["std"].apply(lambda x: f"{x:.4f}")

results.loc["Summary"] = [args.mod, args.data] + formatted_summary.tolist()

results.to_excel(data_file_path, index=False)
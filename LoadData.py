import h5py
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, Dataset

def scale_data(data):
    scaler = preprocessing.StandardScaler()
    if data.shape[1] == 66:
        for i in range(data.shape[0]):
            data[i, :62, :] = scaler.fit_transform(data[i, :62, :])
            data[i, 62:, :] = scaler.fit_transform(data[i, 62:, :])
    else:
        for i in range(data.shape[0]):
            data[i, :, :] = scaler.fit_transform(data[i, :, :])
    return data

class DiabetesDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


def get_eeg(Files, channel):
    end_data = np.zeros((0, channel, 376))
    target_data = np.zeros((0, channel, 376))
    non_target_data = np.zeros((0, channel, 376))
    end_test = np.zeros((0, channel, 376))
    target_test = np.zeros((0, channel, 376))
    non_target_test = np.zeros((0, channel, 376))

    for file in Files:
        print(file)
        end_data0 = np.array(h5py.File(file, 'r')['end'])
        target_data0 = np.array(h5py.File(file, 'r')['target'])
        non_target_data0 = np.array(h5py.File(file, 'r')['non_target'])

        end_test0 = np.array(h5py.File(file, 'r')['end_test'])
        target_test0 = np.array(h5py.File(file, 'r')['tgrget_test'])
        # target_test0 = np.array(h5py.File(file, 'r')['target_test'])
        non_target_test0 = np.array(h5py.File(file, 'r')['non_target_test'])

        end_data = np.concatenate((end_data, end_data0), axis=0)
        target_data = np.concatenate((target_data, target_data0), axis=0)
        non_target_data = np.concatenate((non_target_data, non_target_data0), axis=0)
        end_test = np.concatenate((end_test, end_test0), axis=0)
        target_test = np.concatenate((target_test, target_test0), axis=0)
        non_target_test = np.concatenate((non_target_test, non_target_test0), axis=0)

    print("target size:{}".format(target_data.shape))
    print("end size:{}".format(end_data.shape))
    print("non target size:{}".format(non_target_data.shape))
    print("target_test size:{}".format(target_test.shape))
    print("end_test size:{}".format(end_test.shape))
    print("non_target_test size:{}".format(non_target_test.shape))
    len_train = target_data.shape[0] + non_target_data.shape[0] + end_data.shape[0]
    return target_data, end_data, non_target_data, target_test, end_test, non_target_test, len_train

def get_eeg_em(Files_eeg, Files_em):
    channel = 62
    end_data = np.zeros((0, channel, 376))
    target_data = np.zeros((0, channel, 376))
    non_target_data = np.zeros((0, channel, 376))
    end_test = np.zeros((0, channel, 376))
    target_test = np.zeros((0, channel, 376))
    non_target_test = np.zeros((0, channel, 376))
    for file in Files_eeg:
        print(file)
        end_data0 = np.array(h5py.File(file, 'r')['end'])
        target_data0 = np.array(h5py.File(file, 'r')['target'])
        non_target_data0 = np.array(h5py.File(file, 'r')['non_target'])

        end_test0 = np.array(h5py.File(file, 'r')['end_test'])
        target_test0 = np.array(h5py.File(file, 'r')['tgrget_test'])
        # target_test0 = np.array(h5py.File(file, 'r')['target_test'])
        non_target_test0 = np.array(h5py.File(file, 'r')['non_target_test'])

        end_data = np.concatenate((end_data, end_data0), axis=0)
        target_data = np.concatenate((target_data, target_data0), axis=0)
        non_target_data = np.concatenate((non_target_data, non_target_data0), axis=0)
        end_test = np.concatenate((end_test, end_test0), axis=0)
        target_test = np.concatenate((target_test, target_test0), axis=0)
        non_target_test = np.concatenate((non_target_test, non_target_test0), axis=0)

    channel_em = 4
    end_data_em = np.zeros((0, channel_em, 376))
    target_data_em = np.zeros((0, channel_em, 376))
    non_target_data_em = np.zeros((0, channel_em, 376))
    end_test_em = np.zeros((0, channel_em, 376))
    target_test_em = np.zeros((0, channel_em, 376))
    non_target_test_em = np.zeros((0, channel_em, 376))
    for file in Files_em:
        print(file)
        end_data0 = np.array(h5py.File(file, 'r')['end'])
        target_data0 = np.array(h5py.File(file, 'r')['target'])
        non_target_data0 = np.array(h5py.File(file, 'r')['non_target'])

        end_test0 = np.array(h5py.File(file, 'r')['end_test'])
        target_test0 = np.array(h5py.File(file, 'r')['tgrget_test'])
        non_target_test0 = np.array(h5py.File(file, 'r')['non_target_test'])

        end_data_em = np.concatenate((end_data_em, end_data0), axis=0)
        target_data_em = np.concatenate((target_data_em, target_data0), axis=0)
        non_target_data_em = np.concatenate((non_target_data_em, non_target_data0), axis=0)
        end_test_em = np.concatenate((end_test_em, end_test0), axis=0)
        target_test_em = np.concatenate((target_test_em, target_test0), axis=0)
        non_target_test_em = np.concatenate((non_target_test_em, non_target_test0), axis=0)

    end_data = np.concatenate((end_data, end_data_em), axis=1)
    target_data = np.concatenate((target_data, target_data_em), axis=1)
    non_target_data = np.concatenate((non_target_data, non_target_data_em), axis=1)
    end_test = np.concatenate((end_test, end_test_em), axis=1)
    target_test = np.concatenate((target_test, target_test_em), axis=1)
    non_target_test = np.concatenate((non_target_test, non_target_test_em), axis=1)

    print("target size:{}".format(target_data.shape))
    print("end size:{}".format(end_data.shape))
    print("non target size:{}".format(non_target_data.shape))
    print("target_test size:{}".format(target_test.shape))
    print("end_test size:{}".format(end_test.shape))
    print("non_target_test size:{}".format(non_target_test.shape))
    len_train = target_data.shape[0] + non_target_data.shape[0] + end_data.shape[0]
    return target_data, end_data, non_target_data, target_test, end_test, non_target_test, len_train

def get_em(Files):
    channels = 4
    end_data = np.zeros((0, channels, 376))
    target_data = np.zeros((0, channels, 376))
    non_target_data = np.zeros((0, channels, 376))
    end_test = np.zeros((0, channels, 376))
    target_test = np.zeros((0, channels, 376))
    non_target_test = np.zeros((0, channels, 376))

    for file in Files:
        print(file)
        end_data0 = np.array(h5py.File(file, 'r')['end'])
        target_data0 = np.array(h5py.File(file, 'r')['target'])
        non_target_data0 = np.array(h5py.File(file, 'r')['non_target'])

        end_test0 = np.array(h5py.File(file, 'r')['end_test'])
        target_test0 = np.array(h5py.File(file, 'r')['tgrget_test'])
        non_target_test0 = np.array(h5py.File(file, 'r')['non_target_test'])

        end_data = np.concatenate((end_data, end_data0), axis=0)
        target_data = np.concatenate((target_data, target_data0), axis=0)
        non_target_data = np.concatenate((non_target_data, non_target_data0), axis=0)
        end_test = np.concatenate((end_test, end_test0), axis=0)
        target_test = np.concatenate((target_test, target_test0), axis=0)
        non_target_test = np.concatenate((non_target_test, non_target_test0), axis=0)
    print("target size:{}".format(target_data.shape))
    print("end size:{}".format(end_data.shape))
    print("non target size:{}".format(non_target_data.shape))
    print("target_test size:{}".format(target_test.shape))
    print("end_test size:{}".format(end_test.shape))
    print("non_target_test size:{}".format(non_target_test.shape))
    len_train = target_data.shape[0] + end_data.shape[0] + non_target_data.shape[0]
    return target_data, end_data, non_target_data, target_test, end_test, non_target_test, len_train

def generate_eeg_loader(non_target, target, end, batch_size=256, shuffle=True):
    data = np.concatenate((non_target, target, end), axis=0)
    data = scale_data(data)
    label = [0 for i in range(non_target.shape[0])] + [1 for i in range(target.shape[0])] + [2 for i in range(end.shape[0])]
    data = torch.from_numpy(data).to(torch.float32).unsqueeze(1)
    loader = DataLoader(DiabetesDataset(data, label), batch_size=batch_size, shuffle=shuffle)
    return loader

def generate_eeg_loader_2(non_target, target, batch_size=256, shuffle=True):
    data = np.concatenate((non_target, target), axis=0)
    data = scale_data(data)
    label = [0 for i in range(non_target.shape[0])] + [1 for i in range(target.shape[0])]
    data = torch.from_numpy(data).to(torch.float32).unsqueeze(1)
    loader = DataLoader(DiabetesDataset(data, label), batch_size=batch_size, shuffle=shuffle)
    return loader


def generate_eeg_test_loader(non_target, target, end, batch_size=1024, shuffle=False):
    data = np.concatenate((non_target, target, end), axis=0)
    data = scale_data(data)
    label = [0 for i in range(non_target.shape[0])] + [1 for i in range(target.shape[0])] + [2 for i in range(end.shape[0])]
    data = np.array(data)
    label = np.array(label)
    data = torch.from_numpy(data).to(torch.float32).unsqueeze(1)
    loader = DataLoader(DiabetesDataset(data, label), batch_size=batch_size, shuffle=shuffle)
    return loader

def generate_eeg_test_loader_2(non_target, target, batch_size=1024, shuffle=False):
    data = np.concatenate((non_target, target), axis=0)
    data = scale_data(data)
    label = [0 for i in range(non_target.shape[0])] + [1 for i in range(target.shape[0])]
    data = np.array(data)
    label = np.array(label)
    data = torch.from_numpy(data).to(torch.float32).unsqueeze(1)
    loader = DataLoader(DiabetesDataset(data, label), batch_size=batch_size, shuffle=shuffle)
    return loader


def generate_em_loader(non_target, target, end, batch_size=256, shuffle=True):
    data = np.concatenate((non_target, target, end), axis=0)
    data = scale_data(data)
    label = [0 for i in range(non_target.shape[0])] + [1 for i in range(target.shape[0])] + [2 for i in range(end.shape[0])]
    data = torch.from_numpy(data).to(torch.float32)
    loader = DataLoader(DiabetesDataset(data, label), batch_size=batch_size, shuffle=shuffle)
    return loader


def generate_em_test_loader(non_target, target, end, batch_size=1024, shuffle=False):
    data = np.concatenate((non_target, target, end), axis=0)
    data = scale_data(data)
    label = [0 for i in range(non_target.shape[0])] + [1 for i in range(target.shape[0])] + [2 for i in range(end.shape[0])]
    data = np.array(data)
    label = np.array(label)
    data = torch.from_numpy(data).to(torch.float32)
    loader = DataLoader(DiabetesDataset(data, label), batch_size=batch_size, shuffle=shuffle)
    return loader



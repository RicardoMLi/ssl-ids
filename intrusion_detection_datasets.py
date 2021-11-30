import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils import data_preprocess, data_preprocess_nsl, data_preprocess_kdd, data_preprocess_cic, data_preprocess_iscx, data_preprocess_cidds
from utils import drop_extra_label


class UNSW_NB15Dataset(Dataset):
    def __init__(self, train_csv_path, test_csv_path, image_size, is_training=True):
        super(UNSW_NB15Dataset, self).__init__()

        self.image_size = image_size
        self.is_training = is_training
        self.train_data = pd.read_csv(train_csv_path)
        self.test_data = pd.read_csv(test_csv_path)
        self.data = drop_extra_label(self.train_data, self.test_data, ['id', 'attack_cat'])
        self.Y = self.data.pop('label').values
        self.X = data_preprocess(self.data).values.astype(np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        label = self.Y[index]
        img = np.reshape(self.X[index], (1, self.image_size, self.image_size))
        # img = torch.from_numpy(img)
        # img = self.X[index]

        if self.is_training:
            return img
        else:
            return img, label


class NSL_KDDDataset(Dataset):
    def __init__(self, train_csv_path, test_csv_path, image_size):
        super(NSL_KDDDataset, self).__init__()

        self.image_size = image_size
        self.train_data = data_preprocess_nsl(pd.read_csv(train_csv_path))
        self.test_data = data_preprocess_nsl(pd.read_csv(test_csv_path))
        self.data = pd.concat([self.train_data, self.test_data], axis=0)
        self.data.fillna(value=0, inplace=True)
        self.Y = self.data.pop('attack_cat').values
        self.X = self.data.values.astype(np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        label = self.Y[index]
        img = np.pad(self.X[index], (0, 75), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return img, label


class KDD_CUPDataset(Dataset):
    def __init__(self, csv_path, image_size):
        super(KDD_CUPDataset, self).__init__()

        self.image_size = image_size
        self.data = data_preprocess_kdd(pd.read_csv(csv_path))
        self.Y = self.data.pop('label').values
        self.X = self.data.values.astype(np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        label = self.Y[index]
        img = np.pad(self.X[index], (0, 78), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return img, label


class CIC_IDS2017Dataset(Dataset):
    def __init__(self, csv_path, image_size):
        super(CIC_IDS2017Dataset, self).__init__()

        self.image_size = image_size
        self.data = data_preprocess_cic(pd.read_csv(csv_path))
        self.Y = self.data.pop('Label').values
        self.X = self.data.values.astype(np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        label = self.Y[index]
        img = np.pad(self.X[index], (0, 118), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return img, label


class ISCX_IDS2012Dataset(Dataset):
    def __init__(self, normal_csv_path, attack_csv_path, image_size, total_num):
        super(ISCX_IDS2012Dataset, self).__init__()

        self.image_size = image_size
        self.normal_data = pd.read_csv(normal_csv_path).sample(n=total_num//2)
        self.attack_data = pd.read_csv(attack_csv_path).sample(n=total_num//2)
        self.data = data_preprocess_iscx(pd.concat([self.normal_data, self.attack_data], axis=0))
        self.Y = self.data.pop('label').values
        self.X = self.data.values.astype(np.float32)
        self.pad_num = image_size**2 - self.X.shape[1]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        label = self.Y[index]
        img = np.pad(self.X[index], (0, self.pad_num), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return img, label


class CIDDS_001Dataset(Dataset):
    def __init__(self, base_csv_path, image_size):
        super(CIDDS_001Dataset, self).__init__()

        self.image_size = image_size
        self.df = pd.DataFrame()
        for csv_path in os.listdir(base_csv_path):
            print("正在处理文件", csv_path)
            data_path = os.path.join(base_csv_path, csv_path)
            data_df = pd.read_csv(data_path, dtype={'Bytes': str})
            self.df = pd.concat([self.df, data_df], axis=0)

        self.data = data_preprocess_cidds(self.df)
        self.Y = self.data.pop('class').values
        self.X = self.data.values.astype(np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        label = self.Y[index]
        img = np.pad(self.X[index], (0, 123), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return img, label
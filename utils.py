import math
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import preprocessing
from functools import wraps
from torchvision import transforms


# 移除额外的标签
def drop_extra_label(df_train, df_test, labels):
    for label in labels:
        df_train.drop(label, axis=1, inplace=True)
        df_test.drop(label, axis=1, inplace=True)

    return pd.concat([df_train, df_test], axis=0)


# 对于离散型特征采用最大最小归一化
def min_max_norm(df, name):
    x = df[name].values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[name] = x_scaled


def numerical_split_ohe(df, name):
    pd_to_np = df[name].tolist()
    np_split = []

    categories = np.linspace(0, 1, num=256, endpoint=False)
    quantization = range(0, 256)

    for value in pd_to_np:
        for i in range(len(categories) - 1):
            if categories[i] <= float(value) <= categories[i + 1]:
                np_split.append(quantization[i])
                break
            if float(value) > categories[-1]:
                np_split.append(quantization[-1])
                break

    df[name] = np_split


# 对数据集进行预处理
def data_preprocess(df):
    # 将proto、state、service、label移到最后几列
    traincols = list(df.columns.values)
    traincols.pop(traincols.index('proto'))
    traincols.pop(traincols.index('state'))
    traincols.pop(traincols.index('service'))
    df = df[traincols + ['proto', 'state', 'service']]

    for i in range(0, len(df.columns.values) - 3):
        min_max_norm(df, df.columns.values[i])
        # numerical_split_ohe(df, df.columns.values[i])

    # 将所有字符型特征进行onehot encoding
    return pd.get_dummies(df, columns=['proto', 'state', 'service'])


def random_flip(images):
    transform = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomCrop(size=(12, 12)),
        transforms.Resize(size=(14, 14))
    )
    images = transform(images)

    return images


def random_shuffle(images):
    transformed_images = torch.zeros_like(images)
    for index in range(images.size(0)):
        image = np.reshape(images.cpu().numpy()[index], 196)
        np.random.shuffle(image)
        transformed_images[index] = torch.from_numpy(np.reshape(image, (1, 14, 14)))
    # images = np.reshape(images, 196)
    # np.random.shuffle(images)
    # transformed_images = np.reshape(images, (1, 14, 14))
    transform = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
    )
    images = transform(transformed_images)

    return images


def add_labels(df):
    df.columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                  "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                  "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
                  "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                  "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                  "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                  "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                  "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
                  "attack_cat", "level"]

    return df


def preprocess_labels(df):
    df.drop("level", axis=1, inplace=True)
    is_attack = df['attack_cat'].map(lambda a: 0 if a == 'normal' else 1)
    df['attack_cat'] = is_attack

    return df


# 对于连续型特征采用Z-Score方式归一化
def z_score_norm(df, name):
    df[name] = preprocessing.scale(df[name].to_list())


def data_preprocess_nsl(df):

    df.dropna(inplace=True)
    # 添加标签
    df = add_labels(df)

    # 处理标签
    df = preprocess_labels(df)

    # 将所有连续型特征归一化
    continuous_features_ids = [1, 5, 6, 10, 11, 13, 16, 17, 18, 19]
    for feature_id in continuous_features_ids:
        z_score_norm(df, df.columns[feature_id - 1])

    # 将所有离散型特征归一化
    discrete_features_ids = [8, 9, 15, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    for feature_id in discrete_features_ids:
        min_max_norm(df, df.columns[feature_id - 1])

    # 由于训练和测试数据集中num_outbound_cmds这一列所有值均为0，故删除此列
    df.drop('num_outbound_cmds', axis=1, inplace=True)

    return pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])


def data_preprocess_kdd(df):

    df.dropna(inplace=True)
    # 将label标签变为二分类问题
    is_attack = df['label'].map(lambda a: 0 if a == 'normal' else 1)
    df['label'] = is_attack

    min_max_norm(df, df.columns[0])
    for feature_id in range(4, len(df.columns.values)-1):
        min_max_norm(df, df.columns[feature_id])

    return pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])


def data_preprocess_cic(df):

    # 将infinity替换为Nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)
    # 将label标签变为二分类问题
    is_attack = df['Label'].map(lambda a: 0 if a == 'BENIGN' else 1)
    df['Label'] = is_attack

    for feature_id in range(0, len(df.columns.values)-1):
        min_max_norm(df, df.columns[feature_id])

    return df


def data_preprocess_iscx(df):
    # 将ip_src和ip_dst去掉
    df.drop('ip.src', axis=1, inplace=True)
    df.drop('ip.dst', axis=1, inplace=True)

    # 将label转换为0和1
    is_attack = df['label'].map(lambda a: 0 if a == 'normal' else 1)
    df['label'] = is_attack

    min_max_norm(df, df.columns[0])
    min_max_norm(df, df.columns[1])
    for feature_id in range(3, len(df.columns.values) - 1):
        min_max_norm(df, df.columns[feature_id])

    return pd.get_dummies(df, columns=['frame.protocols'])


def data_preprocess_cidds(df):
    # 将ip_src和ip_dst去掉
    df.drop('Date first seen', axis=1, inplace=True)
    df.drop('Src IP Addr', axis=1, inplace=True)
    df.drop('Dst IP Addr', axis=1, inplace=True)

    # 将label转换为0和1
    is_attack = df['class'].map(lambda a: 0 if a == 'normal' else 1)
    df['class'] = is_attack

    # 将Bytes这列中的M转换为byte
    M2b = df['Bytes'].map(lambda a: math.floor(float(a[:-1]) * 1048576) if a.endswith('M') else a)
    df['Bytes'] = M2b

    features_ids = [0, 2, 3, 4, 5, 6, 8]
    for feature_id in features_ids:
        min_max_norm(df, df.columns[feature_id])

    return pd.get_dummies(df, columns=['Proto', 'Flags', 'attackType', 'attackID', 'attackDescription'])


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)



def mse_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    # output shape: [512]
    distance = 0.5 * ((x - y)**2).sum(dim=-1)

    return distance


if __name__ == "__main__":

    train_csv_path = './datasets/UNSW_NB15_training-set.csv'
    test_csv_path = './datasets/UNSW_NB15_testing-set.csv'
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    data = drop_extra_label(train_data, test_data, ['id', 'attack_cat'])
    Y = data.pop('label').values
    X = data_preprocess(data).values.astype(np.float32)


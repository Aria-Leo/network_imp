import gzip
import zipfile
import pickle
import numpy as np
import pandas as pd


class PKLDataLoader:

    @staticmethod
    def load(pkl_file_path, visualization=False, aspect_ratio=1, channels=1):
        f = gzip.open(pkl_file_path, 'rb')
        data = pickle.load(f, encoding='bytes')
        f.close()
        processed_data = []
        # 数据统一归一化到[-1, 1]
        for obj in data:
            features, labels = obj
            if len(features.shape) == 2 and visualization:
                area = features.shape[1]
                row_length = int(np.sqrt(area / channels / aspect_ratio))
                col_length = int(aspect_ratio * row_length)
                features = features.reshape(features.shape[0], channels, row_length, col_length)
            feature_min = features.min()
            feature_max = features.max()
            if feature_max - feature_min <= 1:  # [0, 1]
                features = features * 2 - 1
            elif feature_max - feature_min > 2:  # [0, 255]
                features = features / 127.5 - 1
            processed_data.append((features, labels))
        return (d for d in processed_data)


class CSVDataLoader:

    @staticmethod
    def load(zip_file_path, visualization=False, aspect_ratio=1, channels=1, samples=100):
        # train validation test
        processed_data = [None, None, None]
        with zipfile.ZipFile(zip_file_path, mode='r') as zip_filer:
            for file_name in zip_filer.namelist():
                if '.csv' not in file_name:
                    continue
                with zip_filer.open(file_name, mode='r') as filer:
                    data = pd.read_csv(filer)
                    random_idx = np.random.permutation(data.shape[0])
                    if isinstance(samples, int) and samples < data.shape[0]:
                        data = data.iloc[random_idx[:samples]]
                    elif isinstance(samples, float) and samples < 1:
                        data = data.iloc[random_idx[:int(data.shape[0] * samples)]]
                    features, labels = data.drop('label', axis=1).values, data['label'].values
                    if len(features.shape) == 2 and visualization:
                        area = features.shape[1]
                        row_length = int(np.sqrt(area / channels / aspect_ratio))
                        col_length = int(aspect_ratio * row_length)
                        features = features.reshape(features.shape[0], channels, row_length, col_length)
                    # 归一化到[-1, 1]
                    feature_min = features.min()
                    feature_max = features.max()
                    if feature_max - feature_min <= 1:  # [0, 1]
                        features = features * 2 - 1
                    elif feature_max - feature_min > 2:  # [0, 255]
                        features = features / 127.5 - 1
                    if 'train' in file_name:
                        # 训练集|验证集划分
                        idx = np.random.permutation(len(labels))
                        split_pos = int(len(labels) * 0.8)
                        train_data = (features[idx[:split_pos]], labels[idx[:split_pos]])
                        validation_data = (features[idx[split_pos:]], labels[idx[split_pos:]])
                        processed_data[0], processed_data[1] = train_data, validation_data
                    else:
                        test_data = (features, labels)
                        processed_data[2] = test_data
        return (d for d in processed_data)


if __name__ == '__main__':
    file_path = r'data\fashion_mnist.zip'
    pkl_fp = r'data\mnist.pkl.gz'
    PKLDataLoader.load(pkl_fp)

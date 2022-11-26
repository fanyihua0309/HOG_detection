import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib
from utils import hog_feature


def read_data(pos_path, neg_path, img_size):
    features = []
    # 1: 有行人; 0: 无行人
    labels = []
    for cnt, img_name in enumerate(os.listdir(pos_path)):
        img_path = os.path.join(pos_path, img_name)
        img = Image.open(img_path).convert('L')

        img = np.array(img)
        h, w = img.shape
        img = img[h // 2 - img_size[0] // 2:h // 2 + img_size[0] // 2, w // 2 - img_size[1] // 2:w // 2 + img_size[1] // 2]
        feature = hog_feature(img)
        feature = np.array(feature)
        features.append(feature)
        labels.append(1)

    for img_name in os.listdir(neg_path):
        img_path = os.path.join(neg_path, img_name)
        img = Image.open(img_path).convert('L')
        # img = img.resize(img_size)
        img = np.array(img)
        h, w = img.shape
        img = img[h // 2 - img_size[0] // 2:h // 2 + img_size[0] // 2,
              w // 2 - img_size[1] // 2:w // 2 + img_size[1] // 2]
        img = np.array(img)
        feature = hog_feature(img)
        feature = np.array(feature)
        features.append(feature)
        labels.append(0)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def train():
    # 读取数据，提取特征
    pos_path = 'data/Train/pos'
    neg_path = 'data/Train/neg'
    img_size = (104, 48)
    features, labels = read_data(pos_path, neg_path, img_size)
    print(f'features: {features.shape}')
    print(f'labels: {labels.shape}')

    # 划分训练、测试数据集
    train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.20,
                                                                                random_state=42)
    model = LinearSVC()
    model.fit(train_features, train_labels)
    score = model.score(test_features, test_labels)
    print(f'score: {score}')
    # 保存模型
    model_save_path = 'model/model_my_hog.dat'
    joblib.dump(model, model_save_path)


if __name__ == '__main__':
    train()


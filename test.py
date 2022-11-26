import os.path
import numpy as np
from PIL import Image
from skimage.feature import hog
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
import joblib
from utils import hog_feature


# 可视化 HOG 特征描述符
def visualize_HOG(img_path, mode='show'):
    img = Image.open(img_path).convert('L')
    img = np.array(img)
    feature, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2',
        visualize=True)
    plt.imshow(hog_img, cmap='gray')
    if mode == 'save':
        img_name = os.path.basename(img_path)
        img_save_path = os.path.join('results/hog', img_name)
        plt.axis('off')
        plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


# 测试单张图像，画出预测框
def test_single_image(model_path, img_path, mode='show'):
    model = joblib.load(model_path)
    print(f'Load model: {model_path}')
    img = Image.open(img_path).convert('L')
    print(f'Read image: {img_path}')
    img = np.array(img)
    plt.imshow(img, cmap='gray')

    h, w = img.shape
    box_h, box_w = 104, 48
    stride = 10
    row = (w - box_w) // stride
    col = (h - box_h) // stride
    boxes = []
    scores = []
    candidate_num = 0
    for i in range(row):
        for j in range(col):
            box = img[i * stride:i * stride + box_h, j * stride:j * stride + box_w]
            if not box.shape == (box_h, box_w):
                continue
            feature = hog_feature(box)
            feature = feature.reshape(1, -1)
            pred = model.predict(feature)[0]
            # 若预测结果图中有行人
            if pred == 1:
                candidate_num += 1
                score = model.decision_function(feature)[0]
                # 根据预测的得分值进一步筛选，存储候选框的坐标与分值
                if score > 1:
                    boxes.append([i * stride, j * stride, i * stride + box_h, j * stride + box_w])
                    scores.append(score)
    boxes = np.array(boxes)
    # 对候选框进行极大值抑制得到最终结果
    pick = non_max_suppression(boxes, probs=scores, overlapThresh=0.3)
    print(f'Total candidate: {candidate_num}   Final: {len(pick)}')
    # 画预测框
    for (x1, y1, x2, y2) in pick:
        plt.gca().add_patch(plt.Rectangle((x1, y1), box_w, box_h, edgecolor='r', lw=1.5, facecolor='none'))
    # 保存包含预测框的图像
    if mode == 'save':
        img_name = os.path.basename(img_path)
        img_save_path = os.path.join('results', img_name)
        plt.axis('off')
        plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    model_path = 'model/model_my_hog.dat'
    img_path = 'data/Test/pos/crop_000026g.png'
    # visualize_HOG(img_path)
    test_single_image(model_path, img_path)
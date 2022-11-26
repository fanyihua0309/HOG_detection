import os
from PIL import Image
import cv2
import numpy as np


class HOG():
    def __init__(self, cell_size=8, block_size=16, stride=8):
        self.cell_size = cell_size
        self.block_size = block_size
        self.stride = stride

    def pre_process(self, img):
        h, w = img.shape
        # 将图像尺寸处理成可以被 cell 大小整除
        if h % self.cell_size:
            pad = self.cell_size - h % self.cell_size
            img = np.pad(img, ((pad // 2, pad - pad // 2), (0, 0)))
        if w % self.cell_size:
            pad = self.cell_size - w % self.cell_size
            img = np.pad(img, ((0, 0), (pad // 2, pad - pad // 2)))
        # 将 0-255 归一化为 0-1
        img = img / 255
        # Gamma 校正
        img = np.power(img, 1 / 2)
        return img

    def compute_gradient(self, img):
        # 在 x 轴方向左右两侧分别填充 1 列 边界值
        img = img.astype('float32')
        img_pad = np.pad(img, ((0, 0), (1, 1)), 'edge')
        right = img_pad[:, 2:]
        left = img_pad[:, :-2]
        grad_x = right - left
        # 将 x 方向梯度大小为 0 的替换为很小的常数 1e-12，防止 x, y 方向梯度均为 0 时算除法求反正切结果 nan
        grad_x[grad_x == 0.0] = 1e-12

        # 在 y 轴方向上下两侧分别填充 1 列 边界值
        img_pad = np.pad(img, ((1, 1), (0, 0)), 'edge')
        up = img_pad[2:, :]
        down = img_pad[:-2, :]
        grad_y = up - down

        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
        orient = np.arctan(grad_y / grad_x)
        angle = orient / np.pi * 180
        # 将 -90-90 的角度转换为 0-180
        h, w = img.shape
        for i in range(h):
            for j in range(w):
                if angle[i][j] < 0:
                    angle[i][j] += 180
        return grad, angle

    def single_cell_binning(self, grad_cell, angle_cell):
        bin = np.zeros((9))
        for i in range(self.cell_size):
            for j in range(self.cell_size):
                grad = grad_cell[i][j]
                angle = angle_cell[i][j]
                angle /= 20
                r1 = angle - int(angle)
                r2 = int(angle) + 1 - angle
                grad1 = grad * r1
                grad2 = grad * r2
                bin[int(angle)] += grad1
                bin[(int(angle) + 1) % 9] += grad2
        return bin

    def cell_binning(self, grad, angle):
        h, w = grad.shape
        row, col = h // self.cell_size, w // self.cell_size
        bins = np.zeros((row, col, 9))
        for i in range(row):
            for j in range(col):
                grad_cell = grad[i * self.cell_size:(i + 1) * self.cell_size,
                            j * self.cell_size:(j + 1) * self.cell_size]
                angle_cell = angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                bins[i][j] = self.single_cell_binning(grad_cell, angle_cell)
        return bins

    def block_binning(self, bins):
        block_size = self.block_size // self.cell_size
        stride = self.stride // self.cell_size
        h, w, b = bins.shape
        row = (h - block_size) // stride
        col = (w - block_size) // stride
        block_bins = np.zeros((row, col, b * block_size * block_size))
        for i in range(row):
            for j in range(col):
                block = bins[i * stride:i * stride + block_size, j * stride:j * stride + block_size, :]
                block = block.reshape(1, 1, -1)
                norm = np.sqrt((block * block).sum())
                block_bins[i, j, :] = block / norm
        return block_bins.reshape(-1)

    def pipeline(self, img):
        img = self.pre_process(img)
        grad, angle = self.compute_gradient(img)
        bins = self.cell_binning(grad, angle)
        feature = self.block_binning(bins)
        return feature


if __name__ == '__main__':
    hog = HOG()
    path = 'data/Train/pos/crop001001a.png'
    img = Image.open(path).convert('L')
    img = np.array(img)
    feature = hog.pipeline(img)
    print(feature.shape)

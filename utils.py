from skimage.feature import hog
from HOG import HOG


def hog_feature(img, mode='ours'):
    if mode == 'ours':
        my_hog = HOG()
        feature = my_hog.pipeline(img)
    else:
        feature = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2',
                      feature_vector=True)
    return feature
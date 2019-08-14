import numpy as np
import tifffile as tif
import random

class DataSet():
    '''
    通过读入文件生成数据集
    '''

    def __init__(self, image_path, label_path):

        self.image_path = np.array(image_path)
        self.label_path = np.array(label_path)
        self.batch_count = 0
        self.epoch_count = 0

    def num_examples(self):
        '''
        得到样本的数量
        :return:
        '''

        return self.image_path.shape[0]

    def next_batch(self, batch_size):
        '''
        next_batch函数
        :param batch_size:
        :return:
        '''

        start = self.batch_count * batch_size
        end = start + batch_size
        self.batch_count += 1

        if end > self.image_path.shape[0]:
            self.batch_count = 0
            random_index = np.random.permutation(self.image_path.shape[0])
            '''
            permutation不直接在原来的数组上进行操作，
            而是返回一个新的打乱顺序的数组，并不改变原来的数组。
            '''
            self.image_path = self.image_path[random_index]
            self.label_path = self.label_path[random_index]
            self.epoch_count += 1
            start = self.batch_count * batch_size
            end = start + batch_size
            self.batch_count += 1

        image_batch, label_batch = self.read_path(self.image_path[start:end],
                                                  self.label_path[start:end])
        return image_batch, label_batch

    def read_path(self, x_path, y_path):
        '''
        将路径读为图片
        :param x_path:
        :param y_path:
        :return:
        '''
        x = []
        y = []
        for i in range(x_path.shape[0]):
            x.append(self.pre_processing(tif.imread(x_path[i])))
            y.append(self.one_hot(tif.imread(y_path[i])))

        return np.array(x), np.array(y)

    # def transform(self, img):
    #
    #     return img

    def pre_processing(self,img):
        # Random exposure and saturation (0.9 ~ 1.1 scale)

        img[:,:,:] = np.where(img[:,:,:]>7700,7700,img)
        img[:, :, :] = np.where(img[:, :, :] < 50, 50, img)
        img = (img - 50)/(7700 - 50)
        # rand_s = random.uniform(0.9, 1.1)
        # rand_v = random.uniform(0.9, 1.1)
        #
        # # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #颜色空间转换
        #
        # tmp = np.ones_like(img[:, :, 1]) * 255
        # img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
        # img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)
        #
        # img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        # Centering helps normalization image (-1 ~ 1 value)
        return img

    def one_hot(self,img):
        img = img[:,:]
        img_onehot = np.zeros([256,256,2])
        img_onehot[:,:,0] = np.where(img>=1,0,1)
        img_onehot[:,:,1] = np.where(img>=1,1,0)
        return img_onehot
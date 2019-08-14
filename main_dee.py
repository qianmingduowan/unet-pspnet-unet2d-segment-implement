from training.training_dee import *
from structure.unet import Unet
from structure.unet2d import unet_model_2d_attention
from data_reader.arable_land_reader import DataSet
from metics.metrics import f1
import pandas as pd
from structure.pspnet import pspnet50

def main(argv=None):
    # batch_size = 8
    input_size_1 = 256
    # input_size_2 = 256
    # data_root_path = '/input0/VOCdevkit_train/VOC2012/'
    batch_size = 16
    csv_path = 'path_list.csv'
    steps_per_epoch = 7200 // batch_size
    validation_steps = 1050// batch_size
    data_path_df = pd.read_csv(csv_path)
    data_path_df = data_path_df.sample(frac=1)
    data_path_df_train = data_path_df[:7200]
    data_path_df_val = data_path_df[7200:]
    initial_learning_rate = 0.01
    # 8250为总训练数量
    # validation_steps = 8250 // batch_size
    load_train = DataSet(image_path=data_path_df_train['image'].values, label_path=data_path_df_train['label'].values)
    load_test  = DataSet(image_path=data_path_df_val['image'].values, label_path=data_path_df_val['label'].values)
    # csv_path = './input0/path_list.csv'
    model = pspnet50(input_shape=(256, 256, 3), num_classes=2, lr_init=0.01)
    # model = Unet(input_shape = input_size_1,n_labels = 2,initial_learning_rate=initial_learning_rate,metrics='accuracy')
    # 这个metrics定义有点问题
    # model = unet_model_2d_attention([256, 256, 3], 2, batch_normalization=True)
    train_model(model,
                'model_file.hdf5',
                train_generator_data(load_train, batch_size=batch_size),
                train_generator_data(load_test, batch_size=batch_size),
                #             val_generator_data(voc),
                steps_per_epoch=steps_per_epoch,
                learning_rate_drop=0.2,
                # learning_rate_epochs=20,
                validation_steps = validation_steps,
                initial_learning_rate=initial_learning_rate,
                learning_rate_patience=5)

if __name__=='__main__':
    main()
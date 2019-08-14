from keras.engine import Input,Model
from keras.optimizers import Adam
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose , Activation
from metics.metrics import f1 , mean_iou
import metics.metrics as m


def Conv2d_BN(x,nb_filter,kernel_size = (3,3),strides = (1,1), padding = 'same'):
    x = Conv2D(nb_filter,kernel_size,strides=strides,padding = padding)(x)
    x = BatchNormalization(axis = 3)(x)
    x = LeakyReLU(alpha= 0.1)(x)
    return x
def Conv2dT_BN(x,filters,kernel_size = (3,3),strides = (2,2), padding = 'same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def Unet(input_shape,n_labels,initial_learning_rate=0.001,metrics='accuracy'):
    input = Input(shape=(input_shape, input_shape, 3))
    conv1 = Conv2d_BN(input,8)
    conv1 = Conv2d_BN(conv1,8)
    pool1 = MaxPooling2D(pool_size = (2,2),strides = (2,2),padding = 'same')(conv1)

    conv2 = Conv2d_BN(pool1, 16)
    conv2 = Conv2d_BN(conv2, 16)
    pool2 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 32)
    conv3 = Conv2d_BN(conv3, 32)
    pool3 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 64)
    conv4 = Conv2d_BN(conv4, 64)
    pool4 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 128)
    # conv5 = Dropout(0.5)(conv5) #这里用dropout是否会影响结果？
    conv5 = Conv2d_BN(conv5, 128)
    # conv5 = Dropout(0.5)(conv5)

    convt1 = Conv2dT_BN(conv5,64)
    concat1 = concatenate([conv4,convt1],axis = 3)
    # concat1 = Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1,64)
    conv6 = Conv2d_BN(conv6,64)

    convt2 = Conv2dT_BN(conv6,32)
    concat2 = concatenate([conv3,convt2],axis = 3)
    # concat2 = Dropout(0.5)(convt2)
    conv7 = Conv2d_BN(concat2,32)
    conv7 = Conv2d_BN(conv7,32)

    convt3 = Conv2dT_BN(conv7, 16)
    concat3 = concatenate([conv2, convt3], axis=3)
    # concat3 = Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 16)
    conv8 = Conv2d_BN(conv8, 16)

    convt4 = Conv2dT_BN(conv8, 8)
    concat4 = concatenate([conv1, convt4], axis=3)
    # concat4 = Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 8)
    conv9 = Conv2d_BN(conv9, 8)
    # conv9 = Dropout(0.5)(conv9)

    output = Conv2D(filters = n_labels, kernel_size = (1,1) , strides = (1,1), padding = 'same' )(conv9)
    output = Activation('softmax')(output)
    model = Model(input, output)

    metrics = [metrics,f1,mean_iou]
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss= 'mean_squared_error', metrics=metrics)

    return model
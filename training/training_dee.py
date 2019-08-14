from functools import partial
import math
from keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler,ReduceLROnPlateau,EarlyStopping,TensorBoard
from keras.models import load_model

def train_generator_data(DataSet,batch_size):
    while True:
        x,y=DataSet.next_batch(batch_size)
        yield (x,y)

# def val_generator_data(voc_reader):
#     while True:
#         x,y=voc_reader.next_val_batch()
#         yield (x,y)


def step_decay(epoch,initial_lrate,drop,epochs_drop):
    return initial_lrate * math.floor((1+epoch)/float(epochs_drop))


def get_callbacks(model_file,initial_learning_rate=0.0001,learning_rate_drop=0.5,learning_rate_epochs=40,
                  learning_rate_patience=50,logging_file="training.log",verbosity=1,early_stopping_patience=None):
    callbacks=list()

#     weights.{epoch:02d}-{val_loss:.2f}.hdf5
    callbacks.append(ModelCheckpoint(model_file, monitor='val_mean_iou', save_best_only=True,mode='max'))
    callbacks.append(CSVLogger(logging_file,append=True))
    callbacks.append(TensorBoard())

    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(monitor='val_mean_iou',verbose=verbosity, patience=early_stopping_patience))
    return callbacks


def train_model(model, model_file, training_generator,
                validation_generator,
                steps_per_epoch,
                validation_steps,
                initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=300,
                learning_rate_patience=None, early_stopping_patience=None):

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=get_callbacks(model_file,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience))
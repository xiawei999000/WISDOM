'''
build the integrated diagnostic model M_IS:
using the short, long diameters, diamter ratio combined with the img predictions
'''

import os
import matplotlib.pyplot as plt
import pickle
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.keras.metrics import AUC

# control the radomness
import numpy as np
my_seed = 666
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)

# enable the specified GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# zero padding for the clinical features
def patches_feas_zero_padding(patches_feas, patches_num):
    p_num = len(patches_feas)
    patches_feas_padding = np.zeros([p_num, patches_num])
    for ind_p in range(0, p_num):
        P_patches_feas = patches_feas[ind_p]
        patch_num_p = len(P_patches_feas)
        P_patches_feas_temp = np.zeros(patches_num)
        for ind_patch in range(0, patch_num_p):
            LN_feas = P_patches_feas[ind_patch]
            P_patches_feas_temp[ind_patch] = LN_feas
        patches_feas_padding[ind_p, :] = P_patches_feas_temp
    return patches_feas_padding

# define focal loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def get_basic_model(feas):
    inputs = keras.Input(feas)
    x = layers.Dense(units=6, activation="relu")(inputs)
    x = layers.Dense(units=12, activation="relu")(x)
    x = layers.Dense(units=24, activation="relu")(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    # Define the model.
    SelfDefinedmodel = keras.Model(inputs, outputs, name="integrated diagnostic model")
    SelfDefinedmodel.summary()
    return SelfDefinedmodel


def get_WISDOM_model(shape_p):
    base_model = get_basic_model(shape_p[1])
    print('base_model for LN patches prediction')

    # patient level LN meta status
    model_input = keras.Input(shape_p)
    x1 = layers.TimeDistributed(base_model, name="TimeDistributed")(model_input)
    output_meta_status = layers.GlobalMaxPooling1D(name="meta_status")(x1)

    # patient level LN meta ratio
    output_meta_ratio = layers.GlobalAveragePooling1D(name="meta_ratio")(x1)

    WISDOM_model_MIS = keras.Model(inputs=model_input, outputs=[output_meta_status, output_meta_ratio],
                                   name="WISDOM_model_MIS")

    WISDOM_model_MIS.summary()

    return WISDOM_model_MIS


def model_training(x_train, y_train_meta_status, y_train_meta_ratio, x_val, y_val_meta_status, y_val_meta_ratio, shape_p,
                   base_model_name, epochs, batch_size, initial_learning_rate, meta_ratio_weight):

    # random shuffle the data and label
    # as the auc require the data set include two classes
    np.random.seed(my_seed)

    data_num_train = x_train.shape[0]
    index_train = np.arange(data_num_train)
    np.random.shuffle(index_train)
    x_train = x_train[index_train]
    y_train_meta_status = y_train_meta_status[index_train]
    y_train_meta_ratio = y_train_meta_ratio[index_train]

    data_num_val = x_val.shape[0]
    index_val = np.arange(data_num_val)
    np.random.shuffle(index_val)
    x_val = x_val[index_val]
    y_val_meta_status = y_val_meta_status[index_val]
    y_val_meta_ratio = y_val_meta_ratio[index_val]

    model = get_WISDOM_model(shape_p)


    optimer = keras.optimizers.Adam(learning_rate=initial_learning_rate)

    # # for class balance
    # CWs = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    # cw = dict(enumerate(CWs))

    auc = AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', name="auc",
              dtype=None, thresholds=None)

    # Compile model.
    model.compile(
        loss={"meta_status": focal_loss(alpha=.25, gamma=2),
              "meta_ratio": 'mean_squared_error'},
        loss_weights={'meta_status': 1, 'meta_ratio': meta_ratio_weight},
        optimizer=optimer,
        metrics={"meta_status": [auc],
              "meta_ratio": ['mean_squared_error']}
    )

    # Define callbacks.
    # validation metric
    output_name = 'meta_status'
    metric_val = output_name + '_' + 'auc'  # "acc"  # "auc"
    loss_val = output_name + '_' + 'loss'

    model_version_save_path = './models/' + base_model_name
    if not os.path.exists(model_version_save_path):
        os.makedirs(model_version_save_path)

    model_save_path = model_version_save_path + '/' + base_model_name + '_lr-' + str(
        initial_learning_rate) + '_batchSize-' + str(batch_size) + '_ratioW-' + str(meta_ratio_weight)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    output_model_file = model_save_path + '/epoch-{epoch:02d}_val_' + metric_val + '-{val_' + metric_val + ':.2f}.hdf5'

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        output_model_file, monitor='val_' + metric_val, save_best_only=True, save_weights_only=False, mode='max'
    )

    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                   cooldown=0,
                                   patience=10,
                                   min_lr=1e-8)

    EarlyStop = keras.callbacks.EarlyStopping(monitor='val_' + metric_val, min_delta=0.001,
                                                 patience=10, verbose=1, mode='max')

    # call back functions
    callbacks = [checkpoint_cb, lr_reducer, EarlyStop]  # , lr_scheduler, EarlyStop

    # model.summary()
    # Train the model, doing validation at the end of each epoch
    model.fit(x=x_train,
              y=[y_train_meta_status, y_train_meta_ratio],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, [y_val_meta_status, y_val_meta_ratio]),
              shuffle=True,
              verbose=1,
              # class_weight=cw,
              callbacks=callbacks
    )

    # # save the training and validation curve
    # fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    # ax = ax.ravel()
    # for i, metric in enumerate([metric_val, loss_val]):
    #     ax[i].plot(model.history.history[metric])
    #     ax[i].plot(model.history.history["val_" + metric])
    #     ax[i].set_title("Model {}".format(metric))
    #     ax[i].set_xlabel("epochs")
    #     ax[i].set_ylabel(metric)
    #     ax[i].legend(["train", "val"])
    # plt.savefig(model_version_save_path + '/train_val_curves_' +
    #             base_model_name + '_lr-' + str(initial_learning_rate) +
    #             '_batchSize-' + str(batch_size) + '.jpg')

if __name__ == '__main__':

    # basic settings
    feas_num = 4 #include the size features and img prediction

    center_name = 'CenterI'

    basic_model_name = 'MIS'

    # load data for model building
    patch_num = 80
    shape_p = (patch_num, feas_num)
    patches_preds = pickle.load(open(
        './data/' + center_name + '/output_preds/patches_preds_ImgLevel.bin',
        "rb"))

    patches_short_diameter = pickle.load(open(
        './data/' + center_name + '/patches_2d_bin_std_all/' + 'Patches_short_diameter.bin',
        "rb"))
    patches_short_diameter = patches_feas_zero_padding(patches_short_diameter, patch_num)

    patches_long_diameter = pickle.load(open(
        './data/' + center_name + '/patches_2d_bin_std_all/' + 'Patches_long_diameter.bin',
        "rb"))
    patches_long_diameter = patches_feas_zero_padding(patches_long_diameter, patch_num)

    patches_ratio_diameter = pickle.load(open(
        './data/' + center_name + '/patches_2d_bin_std_all/' + 'Patches_ratio_diameter.bin',
        "rb"))
    patches_ratio_diameter = patches_feas_zero_padding(patches_ratio_diameter, patch_num)

    # combine all data into one
    # feature normalization
    # diameter:10 mm cutoff for LN positive
    p_num = patches_preds.shape[0]
    all_p_data = np.zeros([p_num, patch_num, feas_num])
    all_p_data[:, :, 0] = patches_preds
    all_p_data[:, :, 1] = patches_short_diameter / 10
    all_p_data[:, :, 2] = patches_long_diameter / 10
    all_p_data[:, :, 3] = patches_ratio_diameter

    # load the labels and dataset partition
    all_LN_meta_labels = pickle.load(open('./data/' + center_name + '/patches_2d_bin_std_all/P_label.bin', "rb"))
    all_pN_labels = pickle.load(open('./data/' + center_name + '/patches_2d_bin_std_all/P_LN_meta_ratio.bin', "rb"))

    training_ind = pickle.load(
        open('./data/' + center_name + '/patches_2d_bin_std_all/training_ind_random_666.bin', "rb"))
    val_ind = pickle.load(open('./data/' + center_name + '/patches_2d_bin_std_all/val_ind_random_666.bin', "rb"))

    # arrange the training and validation set
    x_train = [all_p_data[i] for i in training_ind]
    y_train_meta_status = [all_LN_meta_labels[i] for i in training_ind]
    y_train_meta_ratio = [all_pN_labels[i] for i in training_ind]

    x_val = [all_p_data[i] for i in val_ind]
    y_val_meta_status = [all_LN_meta_labels[i] for i in val_ind]
    y_val_meta_ratio = [all_pN_labels[i] for i in val_ind]

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    y_train_meta_status = np.array(y_train_meta_status)
    y_val_meta_status = np.array(y_val_meta_status)
    y_train_meta_ratio = np.array(y_train_meta_ratio)
    y_val_meta_ratio = np.array(y_val_meta_ratio)

    # training parameters
    learning_rate_i = 2e-4
    batch_size_i = 8
    meta_ratio_weight_i = 1
    epochs = 100

    model_training(x_train, y_train_meta_status, y_train_meta_ratio, x_val, y_val_meta_status, y_val_meta_ratio,
                               shape_p, basic_model_name, epochs, batch_size_i, learning_rate_i, meta_ratio_weight_i)

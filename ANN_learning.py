# training the ANN model

import os
import glob
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# model parameters
num_hidden_neurons = 12
activ_f1 = 'sigmoid'  # 'sigmoid' 'relu' 'tanh' 'linear' 'Softmax' 'LeakyReLU' 'swish' 'softsign' 'softplus'
activ_f2 = 'sigmoid'  # 'elu' 'gelu' 'hard_sigmoid'
optim_alg = 'adam'  # 'adam' 'sgd' 'rmsprop'
loss_func = 'mean_squared_error'  # 'mean_squared_error' 'mean_absolute_error'
learn = False   # training the model (True) or reading the weights of a trained model from a file (False)
num_epochs = 20000
batch_size_param = 200
bias_1 = True  # bias - additional coefficient for signal shifting
bias_2 = True
# draw/do not draw graphs
fig_0 = True   # learning curve
fig_1 = True   # subset 1 (C(NH4HF2)=10 wt.%)
fig_2 = True   # subset 2 (T=90 °C)
fig_3 = True   # predicted surface outside the dataset
fig_4 = True   # error histograms
# other settings
seed = 156  # 'default' - determines the randomness of shuffling the points before splitting into training and test sets
switch = 0  # 0 - use both parts of the dataset for training,
            # 1 - use only the first subset (m=10),
            # 2 - use only the second subset (T=90)
sanity = False  # if the predicted value is less than 0, change it to 0; if greater than 1, change it to 1
data_normalization = 'without_norm'  # 'norm_min_max' 'z_score' 'without_norm'
save_freq = 1000  # model saving frequency

if learn is False:
    with open('training_history.json', 'r') as f:
        history_data = json.load(f)
    print('Model parameters:')
    print(history_data['custom_params'])
    switch = int(history_data['custom_params']['switch'])
    seed = int(history_data['custom_params']['seed'])
    data_normalization = history_data['custom_params']['data_normalization']


# splitting the inputs_list (features) and targets dataset into k sets for cross-validation, returns a cross_data list
# that contains k lists of four elements in the order [inp_train, targ_train, inp_test, targ_test]
def cross_valid_sets(inputs_list, targets, k):
    c_div = len(inputs_list) // k
    r_div = len(inputs_list) % k
    intervals = []  # contains the start and end indices in the split of the original set ([0, 6, 12, 17])
    for i in range(k+1):
        intervals.append(c_div*i)
    for i in range(r_div):
        for j in range(i+1, len(intervals)):
            intervals[j] += 1
    cross_data = []  # k datasets of the form [cross_inp_train, cross_targ_train, cross_inp_test, cross_targ_test]
    for i in range(k):
        cross_inp_train = []
        cross_inp_test = []
        cross_targ_train = []
        cross_targ_test = []
        for j in range(len(inputs_list)):
            if j < intervals[i] or j >= intervals[i+1]:
                cross_inp_train.append(inputs_list[j])
                cross_targ_train.append(targets[j])
            else:
                cross_inp_test.append(inputs_list[j])
                cross_targ_test.append(targets[j])
        cross_data.append([cross_inp_train, cross_targ_train, cross_inp_test, cross_targ_test])
    return cross_data


# takes a set of features and targets, returns the same shuffled data, you can set a seed for reproducibility
def shuffle(inputs_list, targets, seed='default'):
    if len(inputs_list) != len(targets):
        print('Input lists have different lengths!')
        return 0
    if seed == 'default':
        random.seed()
    else:
        random.seed(seed)
    shuffle_list = []
    for i in range(len(targets)):
        combine_list = []
        for j in range(len(inputs_list[0])):
            combine_list.append(inputs_list[i][j])
        combine_list.append(targets[i])
        shuffle_list.append(combine_list)
    random.shuffle(shuffle_list)
    inputs_list = []
    targets = []
    for i in range(len(shuffle_list)):
        combine_list = []
        for j in range(len(shuffle_list[0])-1):
            combine_list.append(shuffle_list[i][j])
        inputs_list.append(combine_list)
        targets.append(shuffle_list[i][-1])
    return inputs_list, targets


# function for clipping predicted target values to the range [0, 1]
def clip_output(x):
    clipped = tf.clip_by_value(x, 0.0, 1.0)
    return clipped.numpy()

# all dataset
if switch == 0:
    x0 =     [[40,   0, 10], [50,   0,  10], [70,   0, 10],    # [T, t, m]
              [40,  15, 10], [50,  15,  10], [70,  15, 10],
              [40,  20, 10], [50,  20,  10], [70,  20, 10],
              [40,  30, 10], [50,  30,  10], [70,  30, 10],
              [40,  60, 10], [50,  60,  10], [70,  60, 10],
              [40,  90, 10], [50,  90,  10], [70,  90, 10],
              [40, 120, 10], [50, 120,  10], [70, 120, 10],
              [40, 150, 10], [50, 150,  10], [70, 150, 10],
              [40, 180, 10], [50, 180,  10], [70, 180, 10],
              [40, 210, 10], [50, 210,  10], [70, 210, 10],
              [40, 240, 10], [50, 240,  10], [70, 240, 10],
              [40, 270, 10], [50, 270,  10], [70, 270, 10],
              [40, 300, 10], [50, 300,  10], [70, 300, 10],
              [90,   0,  0], [90,   0,  1], [90,   0, 2.5], [90,   0, 10], [90,   0, 20], [90,   0, 30], [90,   0, 40], [90,   0, 50],
              [90,  15,  0], [90,  15,  1], [90,  15, 2.5], [90,  15, 10], [90,  15, 20], [90,  15, 30], [90,  15, 40], [90,  15, 50],
              [90,  20,  0], [90,  20,  1], [90,  20, 2.5], [90,  20, 10], [90,  20, 20], [90,  20, 30], [90,  20, 40], [90,  20, 50],
              [90,  30,  0], [90,  30,  1], [90,  30, 2.5], [90,  30, 10], [90,  30, 20], [90,  30, 30], [90,  30, 40], [90,  30, 50],
              [90,  60,  0], [90,  60,  1], [90,  60, 2.5], [90,  60, 10], [90,  60, 20], [90,  60, 30], [90,  60, 40], [90,  60, 50],
              [90,  90,  0], [90,  90,  1], [90,  90, 2.5], [90,  90, 10], [90,  90, 20], [90,  90, 30], [90,  90, 40], [90,  90, 50],
              [90, 120,  0], [90, 120,  1], [90, 120, 2.5], [90, 120, 10], [90, 120, 20], [90, 120, 30], [90, 120, 40], [90, 120, 50],
              [90, 150,  0], [90, 150,  1], [90, 150, 2.5], [90, 150, 10], [90, 150, 20], [90, 150, 30], [90, 150, 40], [90, 150, 50],
              [90, 180,  0], [90, 180,  1], [90, 180, 2.5], [90, 180, 10], [90, 180, 20], [90, 180, 30], [90, 180, 40], [90, 180, 50],
              [90, 210,  0], [90, 210,  1], [90, 210, 2.5], [90, 210, 10], [90, 210, 20], [90, 210, 30], [90, 210, 40], [90, 210, 50],
              [90, 240,  0], [90, 240,  1], [90, 240, 2.5], [90, 240, 10], [90, 240, 20], [90, 240, 30], [90, 240, 40], [90, 240, 50],
              [90, 270,  0], [90, 270,  1], [90, 270, 2.5], [90, 270, 10], [90, 270, 20], [90, 270, 30], [90, 270, 40], [90, 270, 50],
              [90, 300,  0], [90, 300,  1], [90, 300, 2.5], [90, 300, 10], [90, 300, 20], [90, 300, 30], [90, 300, 40], [90, 300, 50],
              [90, 360,  0], [90, 360,  1], [90, 360, 2.5], [90, 360, 10], [90, 360, 20], [90, 360, 30], [90, 360, 40], [90, 360, 50]]
    # α
    y0 =     [0.0,   0.0,   0.0,
              0.07,  0.08,  0.08,
              0.1,   0.119, 0.1,
              0.1,   0.128, 0.1,
              0.12,  0.13,  0.18,
              0.124, 0.136, 0.2,
              0.13,  0.14,  0.23,
              0.136, 0.186, 0.26,
              0.14,  0.21,  0.29,
              0.16,  0.24,  0.34,
              0.19,  0.28,  0.39,
              0.21,  0.36,  0.48,
              0.24,  0.44,  0.56,
              0.0, 0.0,     0.0,     0.0,   0.0,     0.0,     0.0,     0.0,
              0.0, 0.14265, 0.12421, 0.1,   0.41102, 0.4680,  0.49415, 0.49415,
              0.0, 0.16726, 0.28102, 0.146, 0.4542,  0.5210,  0.62241, 0.62241,
              0.0, 0.17463, 0.30504, 0.25,  0.52048, 0.63143, 0.74238, 0.74238,
              0.0, 0.18098, 0.32717, 0.327, 0.78231, 0.80182, 0.82134, 0.82134,
              0.0, 0.18354, 0.34976, 0.334, 0.87734, 0.91624, 0.95514, 0.95514,
              0.0, 0.1861,  0.37228, 0.344, 0.97237, 0.96231, 0.96142, 0.96142,
              0.0, 0.21032, 0.39276, 0.359, 0.97685, 0.97136, 0.96144, 0.96144,
              0.0, 0.23454, 0.40022, 0.372, 0.98133, 0.97441, 0.96149, 0.96149,
              0.0, 0.25933, 0.40673, 0.425, 0.98152, 0.97628, 0.97105, 0.97105,
              0.0, 0.28413, 0.41324, 0.441, 0.98184, 0.97951, 0.97119, 0.97119,
              0.0, 0.34543, 0.41499, 0.478, 0.9821,  0.98012, 0.97214, 0.97214,
              0.0, 0.40674, 0.41674, 0.593, 0.98535, 0.98240, 0.97546, 0.97546,
              0.0, 0.44826, 0.45826, 0.614, 0.98844, 0.98650, 0.98436, 0.98436]
# subset 1 (C(NH4HF2)=10 wt.%)
elif switch == 1:
    x0 = [[40,   0, 10], [50,   0, 10], [70,   0, 10], [90,   0, 10],
          [40,  15, 10], [50,  15, 10], [70,  15, 10], [90,  15, 10],
          [40,  20, 10], [50,  20, 10], [70,  20, 10], [90,  20, 10],
          [40,  30, 10], [50,  30, 10], [70,  30, 10], [90,  30, 10],
          [40,  60, 10], [50,  60, 10], [70,  60, 10], [90,  60, 10],
          [40,  90, 10], [50,  90, 10], [70,  90, 10], [90,  90, 10],
          [40, 120, 10], [50, 120, 10], [70, 120, 10], [90, 120, 10],
          [40, 150, 10], [50, 150, 10], [70, 150, 10], [90, 150, 10],
          [40, 180, 10], [50, 180, 10], [70, 180, 10], [90, 180, 10],
          [40, 210, 10], [50, 210, 10], [70, 210, 10], [90, 210, 10],
          [40, 240, 10], [50, 240, 10], [70, 240, 10], [90, 240, 10],
          [40, 270, 10], [50, 270, 10], [70, 270, 10], [90, 270, 10],
          [40, 300, 10], [50, 300, 10], [70, 300, 10], [90, 300, 10]]
    # α
    y0 = [0.0,   0.0,   0.0,  0.0,
          0.07,  0.08,  0.08, 0.1,
          0.1,   0.119, 0.1,  0.146,
          0.1,   0.128, 0.1,  0.25,
          0.12,  0.13,  0.18, 0.327,
          0.124, 0.136, 0.2,  0.334,
          0.13,  0.14,  0.23, 0.344,
          0.136, 0.186, 0.26, 0.359,
          0.14,  0.21,  0.29, 0.372,
          0.16,  0.24,  0.34, 0.425,
          0.19,  0.28,  0.39, 0.441,
          0.21,  0.36,  0.48, 0.478,
          0.24,  0.44,  0.56, 0.593]
# subset 2 (T=90 °C)
elif switch == 2:
    x0 =     [[90,   0,  0], [90,   0,  1], [90,   0, 2.5], [90,   0, 10], [90,   0, 20], [90,   0, 30], [90,   0, 40], [90,   0, 50],
              [90,  15,  0], [90,  15,  1], [90,  15, 2.5], [90,  15, 10], [90,  15, 20], [90,  15, 30], [90,  15, 40], [90,  15, 50],
              [90,  20,  0], [90,  20,  1], [90,  20, 2.5], [90,  20, 10], [90,  20, 20], [90,  20, 30], [90,  20, 40], [90,  20, 50],
              [90,  30,  0], [90,  30,  1], [90,  30, 2.5], [90,  30, 10], [90,  30, 20], [90,  30, 30], [90,  30, 40], [90,  30, 50],
              [90,  60,  0], [90,  60,  1], [90,  60, 2.5], [90,  60, 10], [90,  60, 20], [90,  60, 30], [90,  60, 40], [90,  60, 50],
              [90,  90,  0], [90,  90,  1], [90,  90, 2.5], [90,  90, 10], [90,  90, 20], [90,  90, 30], [90,  90, 40], [90,  90, 50],
              [90, 120,  0], [90, 120,  1], [90, 120, 2.5], [90, 120, 10], [90, 120, 20], [90, 120, 30], [90, 120, 40], [90, 120, 50],
              [90, 150,  0], [90, 150,  1], [90, 150, 2.5], [90, 150, 10], [90, 150, 20], [90, 150, 30], [90, 150, 40], [90, 150, 50],
              [90, 180,  0], [90, 180,  1], [90, 180, 2.5], [90, 180, 10], [90, 180, 20], [90, 180, 30], [90, 180, 40], [90, 180, 50],
              [90, 210,  0], [90, 210,  1], [90, 210, 2.5], [90, 210, 10], [90, 210, 20], [90, 210, 30], [90, 210, 40], [90, 210, 50],
              [90, 240,  0], [90, 240,  1], [90, 240, 2.5], [90, 240, 10], [90, 240, 20], [90, 240, 30], [90, 240, 40], [90, 240, 50],
              [90, 270,  0], [90, 270,  1], [90, 270, 2.5], [90, 270, 10], [90, 270, 20], [90, 270, 30], [90, 270, 40], [90, 270, 50],
              [90, 300,  0], [90, 300,  1], [90, 300, 2.5], [90, 300, 10], [90, 300, 20], [90, 300, 30], [90, 300, 40], [90, 300, 50],
              [90, 360,  0], [90, 360,  1], [90, 360, 2.5], [90, 360, 10], [90, 360, 20], [90, 360, 30], [90, 360, 40], [90, 360, 50]]
    # α
    y0 =     [0.0, 0.0,     0.0,     0.0,   0.0,     0.0,     0.0,     0.0,
              0.0, 0.14265, 0.12421, 0.1,   0.41102, 0.4680,  0.49415, 0.49415,
              0.0, 0.16726, 0.28102, 0.146, 0.4542,  0.5210,  0.62241, 0.62241,
              0.0, 0.17463, 0.30504, 0.25,  0.52048, 0.63143, 0.74238, 0.74238,
              0.0, 0.18098, 0.32717, 0.327, 0.78231, 0.80182, 0.82134, 0.82134,
              0.0, 0.18354, 0.34976, 0.334, 0.87734, 0.91624, 0.95514, 0.95514,
              0.0, 0.1861,  0.37228, 0.344, 0.97237, 0.96231, 0.96142, 0.96142,
              0.0, 0.21032, 0.39276, 0.359, 0.97685, 0.97136, 0.96144, 0.96144,
              0.0, 0.23454, 0.40022, 0.372, 0.98133, 0.97441, 0.96149, 0.96149,
              0.0, 0.25933, 0.40673, 0.425, 0.98152, 0.97628, 0.97105, 0.97105,
              0.0, 0.28413, 0.41324, 0.441, 0.98184, 0.97951, 0.97119, 0.97119,
              0.0, 0.34543, 0.41499, 0.478, 0.9821,  0.98012, 0.97214, 0.97214,
              0.0, 0.40674, 0.41674, 0.593, 0.98535, 0.98240, 0.97546, 0.97546,
              0.0, 0.44826, 0.45826, 0.614, 0.98844, 0.98650, 0.98436, 0.98436]

# shuffle the input data with a fixed random_state
inputs_list, targets_list = shuffle(x0, y0, seed=seed)          # seed='default' - if you set seed=1 or any other number, the result will be reproducible
cross_data = cross_valid_sets(inputs_list, targets_list, k=10)  # [cross_inp_train, cross_targ_train, cross_inp_test, cross_targ_test]

# training dataset
x_train = np.array(cross_data[0][0])
y_train = np.array(cross_data[0][1])

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)  # axis=0 - calculate sd along columns

# test dataset
x_val = np.array(cross_data[0][2])
y_val = np.array(cross_data[0][3])


# normalization of values
def norm_min_max(x):
    norm_temp = 100  # coefficients for scaling input data to the range 0 - 1
    norm_time = 400
    norm_m = 100
    x = np.array(x)
    x = x.astype(np.float32)
    for i in range(len(x)):
        x[i][0] = x[i][0]/norm_temp
        x[i][1] = x[i][1]/norm_time
        x[i][2] = x[i][2]/norm_m
    return x


def z_score(x):
    x = np.array(x)
    x = x.astype(np.float32)
    x_train_std[x_train_std == 0] = 1  # if std[i] == 0, then std[i] = 1
    x_standardized = (x - x_train_mean) / x_train_std
    return x_standardized


def without_norm(x):
    return x


if data_normalization == 'norm_min_max':
    norm_func = norm_min_max
elif data_normalization == 'z_score':
    norm_func = z_score
else:
    norm_func = without_norm

# custom R^2 metric
@tf.keras.saving.register_keras_serializable()
def tf_r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

# Custom RMSE metric
@tf.keras.saving.register_keras_serializable()
def tf_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


# saves the neural network state every n steps
class CustomEpochModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_freq):
        super().__init__()
        self.filepath = filepath  # path for saving (in the file system)
        self.save_freq = save_freq  # Saving frequency (every save_freq epochs)

    def on_epoch_end(self, epoch, logs=None):
        # saving the model every save_freq epochs
        if epoch % self.save_freq == 0:
            # formatting the filename with the epoch number
            filepath = self.filepath.format(epoch=epoch)
            self.model.save(filepath)  # saving the entire model (architecture and weights)
            print(f"Model saved at {filepath}")

checkpoint_callback = CustomEpochModelCheckpoint(
    'tf_model_{epoch}.keras',            # path for saving with the epoch number
    save_freq=save_freq                  # model saving frequency
)


if learn:
    # deleting old saves
    files = glob.glob(os.path.join('./', 'tf_model*.keras'))
    for file in files:
        os.remove(file)
    if os.path.isfile('./training_history.json'):
        os.remove('./training_history.json')
    # creating the model
    model = models.Sequential()
    model.add(layers.Input(shape=(3,)))  # input layer
    model.add(layers.Dense(units=num_hidden_neurons, activation=activ_f1, use_bias=bias_1))  # hidden layer   sigmoid relu tanh linear Softmax LeakyReLU swish softsign softplus elu gelu hard_sigmoid
    #model.add(layers.Dense(units=num_hidden_neurons, activation=activ_f1, use_bias=bias_1))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=1, activation=activ_f2, use_bias=bias_2))  # output layer
    model.compile(optimizer=optim_alg, loss=loss_func, metrics=[tf_r2, tf_rmse])  # compiling the model
    model.summary()  # output of model information
    # training the model
    history = model.fit(norm_func(x_train), y_train, validation_data=(norm_func(x_val), y_val), epochs=num_epochs,
                        batch_size=batch_size_param, callbacks=[checkpoint_callback])
    #history = model.fit(np.array(inputs_list), np.array(targets_list), epochs=5000, batch_size=200, validation_split=0.1)
    model.save('tf_model.keras')
    history_data = {
        'history': history.history,
        'epoch': history.epoch,
        'params': history.params,
        'custom_params': {'activ_f1': activ_f1, 'activ_f2': activ_f2, 'optim_alg': optim_alg, 'loss_func': loss_func,
                           'num_epochs': num_epochs, 'batch_size_param': batch_size_param, 'seed': seed,
                           'switch': switch, 'bias_1': bias_1, 'bias_2': bias_2, 'data_normalization': data_normalization}
    }
    with open('training_history.json', 'w') as f:
        json.dump(history_data, f)
else:
    # loading the model
    model = load_model('tf_model.keras')
    model.summary()

# prediction of targets for the training and test dataset
y_train_pred = model.predict(norm_func(x_train))
y_val_pred = model.predict(norm_func(x_val))
if sanity:
    y_train_pred = clip_output(y_train_pred)
    y_val_pred = clip_output(y_val_pred)
y_val_pred_list = [round(x[0], 3) for x in y_val_pred]
print("Prediction on the test dataset:")
print(y_val_pred_list)
print('Experimental values:')
print(y_val)

# function for calculating metrics R^2, MAE, RMSE, min_err (max negative err), max_err (max positive err), err_list
def error_calc(y, y_pred):
    err_train = []
    mae_train = 0
    rmse_train = 0
    err3 = 0
    y_av = sum(y)/len(y)
    for i in range(len(y)):
        err = y_pred[i] - y[i]
        err_train.append(err)
        mae_train += abs(err)
        rmse_train += err**2
        err3 += (y[i] - y_av)**2
    r2_train = 1 - (rmse_train/err3)
    mae_train = mae_train/len(y)
    rmse_train = math.sqrt(rmse_train/len(y))
    min_err_train = min(err_train)
    if min_err_train > 0:
        min_err_train = [0]
    max_err_train = max(err_train)
    if max_err_train < 0:
        max_err_train = [0]
    print('R^2: {}, MAE: {}, RMSE: {}, max_-_err: {}, max_+_err: {}'.format(r2_train[0], mae_train[0], rmse_train, min_err_train[0], max_err_train[0]))
    return r2_train[0], mae_train[0], rmse_train, min_err_train[0], max_err_train[0], err_train

# calculation of metrics on the training and test sets
print('Train. set: ', end='')
r2_train, mae_train, rmse_train, min_err_train, max_err_train, err_train = error_calc(y_train, y_train_pred)
print('Test set: ', end='')
r2_test, mae_test, rmse_test, min_err_test, max_err_test, err_test = error_calc(y_val, y_val_pred)

# visualization of the training process
if fig_0:
    plt.plot(history_data['history']['loss'], label='Error on the training set')
    plt.plot(history_data['history']['val_loss'], label='Error on the test set')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

# surface 1 (C(NH4HF2=10 wt.%)) based on the ANN model prediction (subset 1)
if switch == 0 or switch == 1:
    xp = np.arange(30, 100, 2)  # 30, 100, 2 - T, °C
    yp = np.arange(0, 365, 5)   # 0, 350, 5 - time, min
    zn = []  # ANN predictions
    xgrid, ygrid = np.meshgrid(xp, yp)
    xp = list((np.array(xgrid)).reshape(len(xgrid)*len(xgrid[0])))
    yp = list((np.array(ygrid)).reshape(len(ygrid)*len(ygrid[0])))
    grid_1_value = []
    for i in range(len(xp)):
        grid_1_value.append([xp[i], yp[i], 10])  # T, t, m
    zn = model.predict(np.array(norm_func(grid_1_value)))
    if sanity:
        zn = clip_output(zn)
    zgrid = np.reshape(np.array(zn), (len(xgrid), len(xgrid[0])))

    # writing surface points to a *.csv file
    file = open('res_1_ANN.csv', 'w')
    file.write('x values (T, °C)\n')
    for i in range(len(xgrid)):
        string = ''
        for j in range(len(xgrid[0])):
            string += str(xgrid[i][j]) + '; '
        string += '\n'
        file.write(string)
    file.write('\n\n')
    file.write('y values (t, min)\n')
    for i in range(len(ygrid)):
        string = ''
        for j in range(len(ygrid[0])):
            string += str(ygrid[i][j]) + '; '
        string += '\n'
        file.write(string)
    file.write('\n\n')
    file.write('z values (alpha)\n')
    for i in range(len(zgrid)):
        string = ''
        for j in range(len(zgrid[0])):
            string += str(zgrid[i][j]) + '; '
        string += '\n'
        file.write(string)
    file.close()

    if fig_1:
        # selection of points of the NH4HF2 = 10 wt.% plane
        x1_pic, y1_pic, x2_pic, y2_pic, y_pic, x_pic = [], [], [], [], [], []
        for i in range(len(cross_data[0][0])):
            if cross_data[0][0][i][2] == 10 and cross_data[0][0][i][1] <= 300:
                x1_pic.append(cross_data[0][0][i])
                y1_pic.append(cross_data[0][1][i])
        for i in range(len(cross_data[0][2])):
            if cross_data[0][2][i][2] == 10 and cross_data[0][2][i][1] <= 300:
                x2_pic.append(cross_data[0][2][i])
                y2_pic.append(cross_data[0][3][i])
        x1_pic = np.array(x1_pic)
        x2_pic = np.array(x2_pic)
        x_pic = np.array(x_pic)
        y1_pic = np.array(y1_pic)
        y2_pic = np.array(y2_pic)
        y_pic = np.array(y_pic)

        # prediction of values for the training and test sets of the plane
        y_train_pred_v1 = model.predict(norm_func(x1_pic))
        y_val_pred_v1 = model.predict(norm_func(x2_pic))
        if sanity:
            y_train_pred_v1 = clip_output(y_train_pred_v1)
            y_val_pred_v1 = clip_output(y_val_pred_v1)
        y_val_pred_list_v1 = [round(x[0], 3) for x in y_val_pred_v1]
        y_train_pred_list_v1 = [round(x[0], 3) for x in y_train_pred_v1]
        print('____________________Plane NH4HF2 = 10 wt.%______________________')
        print("Prediction on the test set:")
        print(y_val_pred_list_v1)
        print('True values (experiment):')
        print(y2_pic)
        print("Prediction on the training set:")
        print(y_train_pred_list_v1)
        print('True values (experiment):')
        print(list(y1_pic))
        # calculation of metrics on the training and test sets
        print('Train. set: ', end='')
        r2_train, mae_train, rmse_train, min_err_train, max_err_train, err_train = error_calc(y1_pic, y_train_pred_v1)
        print('Test set: ', end='')
        r2_test, mae_test, rmse_test, min_err_test, max_err_test, err_test = error_calc(y2_pic, y_val_pred_v1)
        print('_______________________________________________________________')

        # plotting a graph
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(xp, yp, zn, color='green', s=1)  # ANN predicted values
        ax.scatter3D(x1_pic[:, 0], x1_pic[:, 1], y1_pic, color='blue', s=5)  # exp for train. set
        ax.scatter3D(x2_pic[:, 0], x2_pic[:, 1], y2_pic, color='red', s=5)  # exp for test set
        ax.set_xlabel('T, °C')
        ax.set_ylabel('t, min')
        ax.set_zlabel('α, fraction')
        plt.show()

    if fig_1:
        plt.figure(figsize=(10, 8))
        plt.title('α, fraction')
        plt.xlabel('T, °C')
        plt.ylabel('t, min')
        cs = plt.contour(xgrid, ygrid, zgrid, levels=[0.10, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1.00], colors='black', linewidths=1.0)  # contour line values
        cs.clabel(fontsize=16)  # adds labels for contour line values
        #plt.contourf(xgrid, ygrid, zgrid, 255, cmap=plt.colormaps['RdYlBu_r']) # hsv, rainbow, jet, turbo, brg, gist_rainbow, gnuplot, gnuplot2, RdYlGn, RdYlBu, Spectral
        #plt.colorbar()
        levels = np.linspace(0, 1, 256)  # 256 levels from 0 to 1
        cf = plt.contourf(xgrid, ygrid, zgrid, levels,
                          cmap=plt.colormaps['RdYlBu_r'],
                          extend='neither')  # do not expand beyond levels
        cbar = plt.colorbar(cf, ticks=np.arange(0, 1.1, 0.1))  # tick marks on the axis at intervals of 0.1
        cbar.set_label('α, fraction')
        plt.show()

if switch == 0 or switch == 2:
    # surface 2 (T = 90 °C) based on the ANN model prediction (subset 2)
    xp = np.arange(0, 50, 2)
    yp = np.arange(0, 350, 12)
    zn = []  # ANN prediction
    xgrid, ygrid = np.meshgrid(xp, yp)
    xp = list((np.array(xgrid)).reshape(len(xgrid)*len(xgrid[0])))
    yp = list((np.array(ygrid)).reshape(len(ygrid)*len(ygrid[0])))
    grid_2_value = []
    for i in range(len(xp)):
        grid_2_value.append([90, yp[i], xp[i]])  # T, t, m
    zn = model.predict(np.array(norm_func(grid_2_value)))
    if sanity:
        zn = clip_output(zn)
    zgrid = np.reshape(np.array(zn), (len(xgrid), len(xgrid[0])))

    # writing surface points to a *.csv file
    if True:
        file = open('res_2_ANN.csv', 'w')
        file.write('x values (NH4HF2, wt.%)\n')
        for i in range(len(xgrid)):
            string = ''
            for j in range(len(xgrid[0])):
                string += str(xgrid[i][j]) + '; '
            string += '\n'
            file.write(string)
        file.write('\n\n')
        file.write('y values (t, min)\n')
        for i in range(len(ygrid)):
            string = ''
            for j in range(len(ygrid[0])):
                string += str(ygrid[i][j]) + '; '
            string += '\n'
            file.write(string)
        file.write('\n\n')
        file.write('z values (alpha)\n')
        for i in range(len(zgrid)):
            string = ''
            for j in range(len(zgrid[0])):
                string += str(zgrid[i][j]) + '; '
            string += '\n'
            file.write(string)
        file.close()

    if fig_2:
        # selection of points of the T = 90 °C plane
        x1_pic, y1_pic, x2_pic, y2_pic, y_pic, x_pic = [], [], [], [], [], []
        for i in range(len(cross_data[0][0])):
            if cross_data[0][0][i][0] == 90:
                x1_pic.append(cross_data[0][0][i])
                y1_pic.append(cross_data[0][1][i])
        for i in range(len(cross_data[0][2])):
            if cross_data[0][2][i][0] == 90:
                x2_pic.append(cross_data[0][2][i])
                y2_pic.append(cross_data[0][3][i])
        x1_pic = np.array(x1_pic)
        x2_pic = np.array(x2_pic)
        x_pic = np.array(x_pic)
        y1_pic = np.array(y1_pic)
        y2_pic = np.array(y2_pic)
        y_pic = np.array(y_pic)

        # prediction of values for the training and test sets of the plane
        y_train_pred_v1 = model.predict(norm_func(x1_pic))
        y_val_pred_v1 = model.predict(norm_func(x2_pic))
        if sanity:
            y_train_pred_v1 = clip_output(y_train_pred_v1)
            y_val_pred_v1 = clip_output(y_val_pred_v1)
        y_val_pred_list_v1 = [round(x[0], 3) for x in y_val_pred_v1]
        y_train_pred_list_v1 = [round(x[0], 3) for x in y_train_pred_v1]
        print('____________________Plane T = 90 °C______________________')
        print("Prediction on the test set:")
        print(y_val_pred_list_v1)
        print('True values (experiment):')
        print(y2_pic)
        print("Prediction on the train set:")
        print(y_train_pred_list_v1)
        print('True values (experiment):')
        print(list(y1_pic))
        # calculation of metrics on the training and test sets
        print('Train. set: ', end='')
        r2_train, mae_train, rmse_train, min_err_train, max_err_train, err_train = error_calc(y1_pic, y_train_pred_v1)
        print('Test set: ', end='')
        r2_test, mae_test, rmse_test, min_err_test, max_err_test, err_test = error_calc(y2_pic, y_val_pred_v1)
        print('_________________________________________________________')

        # plotting a graph
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(xp, yp, zn, color='green', s=1)  # ANN predicted values
        ax.scatter3D(x1_pic[:, 2], x1_pic[:, 1], y1_pic, color='blue', s=5)  # exp for LS train
        ax.scatter3D(x2_pic[:, 2], x2_pic[:, 1], y2_pic, color='red', s=5)  # exp for LS test
        ax.set_xlabel('NH4HF2, wt.%')
        ax.set_ylabel('t, min')
        ax.set_zlabel('α, fraction')
        plt.show()

    if fig_2:
        plt.figure(figsize=(10, 8))
        plt.title('α, fraction')
        plt.xlabel('NH4HF2, wt.%')
        plt.ylabel('t, min')
        cs = plt.contour(xgrid, ygrid, zgrid, levels=[0.10, 0.25, 0.4, 0.50, 0.75, 0.95, 0.97, 0.98, 0.99, 1.00], colors='black', linewidths=1.0)  # contour line values
        cs.clabel(fontsize=16)  # adds labels for contour line values
        #plt.contourf(xgrid, ygrid, zgrid, 255, cmap=plt.colormaps['RdYlBu_r'])  # hsv, rainbow, jet, turbo, brg, gist_rainbow, gnuplot, gnuplot2, RdYlGn, RdYlBu, Spectral
        #plt.colorbar()
        levels = np.linspace(0, 1, 256)  # 256 levels from 0 to 1
        cf = plt.contourf(xgrid, ygrid, zgrid, levels,
                          cmap=plt.colormaps['RdYlBu_r'],
                          extend='neither')  # do not expand beyond levels
        cbar = plt.colorbar(cf, ticks=np.arange(0, 1.1, 0.1))  # tick marks on the axis at intervals of 0.1
        cbar.set_label('α, fraction')
        plt.show()

# predicted surface outside the dataset
if switch == 0:
    temp = 60  # prediction of the T = 60 °C surface
    x2 = np.array([[temp,   0,  1], [temp,   0, 2.5], [temp,   0, 10], [temp,   0, 20], [temp,   0, 30], [temp,   0, 40],
                   [temp,  15,  1], [temp,  15, 2.5], [temp,  15, 10], [temp,  15, 20], [temp,  15, 30], [temp,  15, 40],
                   [temp,  20,  1], [temp,  20, 2.5], [temp,  20, 10], [temp,  20, 20], [temp,  20, 30], [temp,  20, 40],
                   [temp,  30,  1], [temp,  30, 2.5], [temp,  30, 10], [temp,  30, 20], [temp,  30, 30], [temp,  30, 40],
                   [temp,  60,  1], [temp,  60, 2.5], [temp,  60, 10], [temp,  60, 20], [temp,  60, 30], [temp,  60, 40],
                   [temp,  90,  1], [temp,  90, 2.5], [temp,  90, 10], [temp,  90, 20], [temp,  90, 30], [temp,  90, 40],
                   [temp, 120,  1], [temp, 120, 2.5], [temp, 120, 10], [temp, 120, 20], [temp, 120, 30], [temp, 120, 40],
                   [temp, 150,  1], [temp, 150, 2.5], [temp, 150, 10], [temp, 150, 20], [temp, 150, 30], [temp, 150, 40],
                   [temp, 180,  1], [temp, 180, 2.5], [temp, 180, 10], [temp, 180, 20], [temp, 180, 30], [temp, 180, 40],
                   [temp, 210,  1], [temp, 210, 2.5], [temp, 210, 10], [temp, 210, 20], [temp, 210, 30], [temp, 210, 40],
                   [temp, 240,  1], [temp, 240, 2.5], [temp, 240, 10], [temp, 240, 20], [temp, 240, 30], [temp, 240, 40],
                   [temp, 270,  1], [temp, 270, 2.5], [temp, 270, 10], [temp, 270, 20], [temp, 270, 30], [temp, 270, 40],
                   [temp, 300,  1], [temp, 300, 2.5], [temp, 300, 10], [temp, 300, 20], [temp, 300, 30], [temp, 300, 40],
                   [temp, 360,  1], [temp, 360, 2.5], [temp, 360, 10], [temp, 360, 20], [temp, 360, 30], [temp, 360, 40]])
    y_pred_3 = model.predict(norm_func(x2))
    if sanity:
        y_pred_3 = clip_output(y_pred_3)
    # plotting a graph
    if fig_3:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(x2[:, 2], x2[:, 1], y_pred_3, color='green', s=5)
        ax.set_xlabel('NH4HF2, wt.%')
        ax.set_ylabel('t, min')
        ax.set_zlabel('α, %')
        plt.show()

# plot for displaying errors
y_pred_1 = [i[0] for i in y_train_pred]
y_pred_2 = [i[0] for i in y_val_pred]
err_train = [i[0] for i in err_train]
err_test = [i[0] for i in err_test]
print("________________________________")
print("Train. set: α(exp), α(ANN)")
print(cross_data[0][1])
print(y_pred_1)
print("________________________________")
print("Test set: α(exp), α(ANN)")
print(cross_data[0][3])
print(y_pred_2)
print("________________________________")
if fig_4:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # first subplot
    axes[0].plot([-0.1, 1.1], [-0.1, 1.1], color='black')
    axes[0].scatter(cross_data[0][1], y_pred_1, marker='o', s=9, color='blue', label='train set')
    axes[0].scatter(cross_data[0][3], y_pred_2, marker='o', s=9, color='red', label='test set')
    axes[0].set_xlabel("α, exp")
    axes[0].set_ylabel("α, ANN")
    axes[0].legend()
    # second subplot
    bin_width = 0.01  # width of one bin
    bin_edges = np.arange(-0.3, 0.3 + bin_width, bin_width)  # from -0.3 to 0.3 with step bin_width
    axes[1].hist(err_train, bins=bin_edges, color='blue', label='train set')
    axes[1].hist(err_test, bins=bin_edges, color='red', label='test set')
    axes[1].set_xlabel("α(ANN)-α(exp), fraction")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim([-0.3, 0.3])
    axes[1].set_ylim([0, 35])
    axes[1].legend()
    plt.tight_layout()
    plt.show()

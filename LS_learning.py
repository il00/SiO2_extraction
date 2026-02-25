# Application of ordinary least squares (LS) for fitting the coefficients
# of the approximating function to experimental values
# α = k0 + k1T + k2t + k3m + k4Tt + k5Tm + k6tm + k7TT + k8tt + k9mm + k10TTt + k11Ttt + k12TTm + k13Tmm +
# + k14ttm + k15tmm + k16Ttm + k17TTT + k18ttt + k19mmm + k22TTTT + k23TTTt + k24TTTm + k25TTtt + k26TTmm + k27Tttt +
# + k28Tmmm + k29TTmt + k30Tmtt + k31Tmmt + k32tttt + k33tttm + k34ttmm + k35tmmm + k36mmmm

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D

# draw/do not draw graphs
fig_1 = True  # subset 1 (C(NH4HF2)=10 wt.%)
fig_2 = True  # subset 2 (T=90 °C)
fig_3 = True  # predicted surface outside the dataset
fig_4 = True  # error histograms
sanity = True  # if the predicted value is less than 0, change it to 0; if greater than 1, change it to 1
seed = 1  # 'default' - determines the randomness of shuffling the points before splitting into training and test sets
switch = 0  # 0 - use both parts of the dataset for training,
            # 1 - use only the first subset (m=10),
            # 2 - use only the second subset (T=90)


# substitution of LS coefficients - replace depending on the equation
def calculate_lms_coof(T, t, m):
    T, t, m = float(T), float(t), float(m)
    return [T, t, m, T*t, T*m, t*m, T*T, t*t, m*m, T*T*t, T*t*t, T*T*m, T*m*m, t*t*m, t*m*m, T*t*m, T*T*T, t*t*t,
               m*m*m, T*T*T*T, T*T*T*t, T*T*T*m, T*T*t*t, T*T*m*m, T*t*t*t, T*m*m*m, T*T*m*t, T*m*t*t, T*m*m*t,
            t*t*t*t, t*t*t*m, t*t*m*m, t*m*m*m, m*m*m*m]


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
              0.0,   0.0,     0.0,     0.0,   0.0,     0.0,     0.0,     0.0,
              0.0,   0.14265, 0.12421, 0.1,   0.41102, 0.4680,  0.49415, 0.49415,
              0.0,   0.16726, 0.28102, 0.146, 0.4542,  0.5210,  0.62241, 0.62241,
              0.0,   0.17463, 0.30504, 0.25,  0.52048, 0.63143, 0.74238, 0.74238,
              0.0,   0.18098, 0.32717, 0.327, 0.78231, 0.80182, 0.82134, 0.82134,
              0.0,   0.18354, 0.34976, 0.334, 0.87734, 0.91624, 0.95514, 0.95514,
              0.0,   0.1861,  0.37228, 0.344, 0.97237, 0.96231, 0.96142, 0.96142,
              0.0,   0.21032, 0.39276, 0.359, 0.97685, 0.97136, 0.96144, 0.96144,
              0.0,   0.23454, 0.40022, 0.372, 0.98133, 0.97441, 0.96149, 0.96149,
              0.0,   0.25933, 0.40673, 0.425, 0.98152, 0.97628, 0.97105, 0.97105,
              0.0,   0.28413, 0.41324, 0.441, 0.98184, 0.97951, 0.97119, 0.97119,
              0.0,   0.34543, 0.41499, 0.478, 0.9821,  0.98012, 0.97214, 0.97214,
              0.0,   0.40674, 0.41674, 0.593, 0.98535, 0.98240, 0.97546, 0.97546,
              0.0,   0.44826, 0.45826, 0.614, 0.98844, 0.98650, 0.98436, 0.98436]
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
x1 = []
for i in range(len(cross_data[0][0])):
    T, t, m = cross_data[0][0][i][0], cross_data[0][0][i][1], cross_data[0][0][i][2],
    x1.append(calculate_lms_coof(T, t, m))
y1 = cross_data[0][1]
x1 = np.array(x1)
y1 = np.array(y1)

# test dataset
x2 = []
for i in range(len(cross_data[0][2])):
    T, t, m = cross_data[0][2][i][0], cross_data[0][2][i][1], cross_data[0][2][i][2],
    x2.append(calculate_lms_coof(T, t, m))
y2 = cross_data[0][3]
x2 = np.array(x2)
y2 = np.array(y2)

# least squares method (LS)
reg = LinearRegression()
reg.fit(x1, y1)
print('coefficients before T, t, m')
print(reg.coef_)
print('Intercept term of the equation')
print(reg.intercept_)
print('R2')
print(reg.score(x1, y1))

# prediction of α (targets) on the training and test sets
y_pred_1 = []
for i in range(len(x1)):
    predict_value = reg.predict([x1[i]])
    if sanity and predict_value[0] < 0:
        predict_value[0] = 0
    if sanity and predict_value[0] > 1:
        predict_value[0] = 1
    y_pred_1.append(predict_value)
y_pred_2 = []
for i in range(len(x2)):
    predict_value = reg.predict([x2[i]])
    if sanity and predict_value[0] < 0:
        predict_value[0] = 0
    if sanity and predict_value[0] > 1:
        predict_value[0] = 1
    y_pred_2.append(predict_value)


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
        min_err_train = 0
    max_err_train = max(err_train)
    if max_err_train < 0:
        max_err_train = 0
    print('R^2: {}, MAE: {}, RMSE: {}, max_-_err: {}, max_+_err: {}'.format(r2_train[0], mae_train[0], rmse_train, min_err_train[0], max_err_train[0]))
    return r2_train[0], mae_train[0], rmse_train, min_err_train[0], max_err_train[0], err_train


# calculation of metrics on the training and test sets
print('Train. set: ', end='')
r2_train, mae_train, rmse_train, min_err_train, max_err_train, err_train = error_calc(y1, y_pred_1)
print('Test set: ', end='')
r2_test, mae_test, rmse_test, min_err_test, max_err_test, err_test = error_calc(y2, y_pred_2)


if switch == 0 or switch == 1:
    # surface 1 (C(NH4HF2=10 wt.%)) based on the LS model prediction (subset 1)
    xp = np.arange(30, 100, 2)  # 30, 100, 2 - temperature, degrees Celsius
    yp = np.arange(0, 365, 5)   # 0, 365, 5 - time, min
    zn = []  # lsm
    xgrid, ygrid = np.meshgrid(xp, yp)
    xp = list((np.array(xgrid)).reshape(len(xgrid)*len(xgrid[0])))
    yp = list((np.array(ygrid)).reshape(len(ygrid)*len(ygrid[0])))
    for i in range(len(xp)):
        T, t, m = xp[i], yp[i], 10
        predict_value = reg.predict([calculate_lms_coof(T, t, m)])[0]
        if sanity and predict_value < 0:
            predict_value = 0
        if sanity and predict_value > 1:
            predict_value = 1
        zn.append(predict_value)
    zgrid = np.reshape(np.array(zn), (len(xgrid), len(xgrid[0])))

    # writing surface points to a *.csv file
    file = open('res_1_LS.csv', 'w')
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
        # plotting a graph
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(xp, yp, zn, color='green', s=1)  # LS predicted values
        ax.scatter3D(x1_pic[:, 0], x1_pic[:, 1], y1_pic, color='blue', s=5)  # exp for LS train
        ax.scatter3D(x2_pic[:, 0], x2_pic[:, 1], y2_pic, color='red', s=5)  # exp for LS test
        ax.set_xlabel('T, °C')
        ax.set_ylabel('t, min')
        ax.set_zlabel('α, faction')
        plt.show()

    if fig_1:
        plt.figure(figsize=(10, 8))
        plt.title('α, fraction')
        plt.xlabel('T, °C')
        plt.ylabel('t, min')
        cs = plt.contour(xgrid, ygrid, zgrid, levels=[0.10, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1.00], colors='black', linewidths=1.0)  # contour line values
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


if switch == 0 or switch == 2:
    # surface 2 (T = 90 °C) based on the LS model prediction (subset 2)
    xp = np.arange(0, 50, 2)    # C(NH4HF2), wt.%
    yp = np.arange(0, 350, 12)  # time, min
    zn = []  # LS predictions
    xgrid, ygrid = np.meshgrid(xp, yp)
    xp = list((np.array(xgrid)).reshape(len(xgrid)*len(xgrid[0])))
    yp = list((np.array(ygrid)).reshape(len(ygrid)*len(ygrid[0])))
    for i in range(len(xp)):
        T, t, m = 90, yp[i], xp[i]
        predict_value = reg.predict([calculate_lms_coof(T, t, m)])[0]
        if sanity and predict_value < 0:
            predict_value = 0
        if sanity and predict_value > 1:
            predict_value = 1
        zn.append(predict_value)
    zgrid = np.reshape(np.array(zn), (len(xgrid), len(xgrid[0])))

    # writing surface points to a *.csv file
    if True:
        file = open('res_2_LS.csv', 'w')
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
        # plotting a graph
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(xp, yp, zn, color='green', s=1)  # LS predicted values
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
        #plt.contourf(xgrid, ygrid, zgrid, 255, cmap=plt.colormaps['RdYlBu_r'])  # RdYlBu_r hsv, rainbow, jet, turbo, brg, gist_rainbow, gnuplot, gnuplot2, RdYlGn, RdYlBu, Spectral
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
    # prediction of the T = 60 °C surface
    temp = 60
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
    x3 = []
    for i in range(len(x2)):
        T, t, m = x2[i][0], x2[i][1], x2[i][2],
        x3.append(calculate_lms_coof(T, t, m))
    # prediction
    y_pred_3 = []
    for i in range(len(x3)):
        predict_value = reg.predict([x3[i]])
        if sanity and predict_value[0] < 0:
            predict_value[0] = 0
        if sanity and predict_value[0] > 1:
            predict_value[0] = 1
        y_pred_3.append(predict_value)
    # plotting a graph
    if fig_3:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(x2[:, 2], x2[:, 1], y_pred_3, color='green', s=5)  # LS
        ax.set_xlabel('NH4HF2, wt.%')
        ax.set_ylabel('t, min')
        ax.set_zlabel('α, fraction')
        plt.show()

# plot for displaying errors
y_pred_1 = [i[0] for i in y_pred_1]
y_pred_2 = [i[0] for i in y_pred_2]
err_train = [i[0] for i in err_train]
err_test = [i[0] for i in err_test]
print("________________________________")
print("Train. set: α(exp), α(LS)")
print(cross_data[0][1])
print(y_pred_1)
print("________________________________")
plt.scatter(cross_data[0][3], y_pred_2, marker='o', s=5, color='red', label='test set')
print("Test set: α(exp), α(LS)")
print(cross_data[0][3])
print(y_pred_2)
print("________________________________")

if fig_4:
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    # first subplot
    ax1.plot([-0.1, 1.1], [-0.1, 1.1], color='black')
    ax1.scatter(cross_data[0][1], y_pred_1, marker='o', s=9, color='blue', label='train set')
    ax1.scatter(cross_data[0][3], y_pred_2, marker='o', s=9, color='red', label='test set')
    ax1.set_xlabel("α, exp")
    ax1.set_ylabel("α, LS")
    ax1.legend()
    # second subplot
    bin_width = 0.01  # width of one bin
    bin_edges = np.arange(-0.3, 0.3 + bin_width, bin_width)
    ax2.hist(err_train, bins=bin_edges, color='blue', label='train set')
    ax2.hist(err_test, bins=bin_edges, color='red', label='test set')
    ax2.set_xlabel("α(LS)-α(exp), fraction")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim([-0.3, 0.3])
    ax2.set_ylim([0, 35])
    ax2.legend()
    plt.tight_layout()
    plt.show()
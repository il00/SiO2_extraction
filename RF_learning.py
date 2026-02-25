# Training the random forest model based on ExtraTreeRegressor()

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import export_text
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# draw/do not draw graphs
fig_1 = True  # subset 1 (C(NH4HF2)=10 wt.%)
fig_2 = True  # subset 2 (T=90 °C)
fig_3 = True  # predicted surface outside the dataset
fig_4 = True  # error histograms

# all dataset
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
m_1 = 10
temp_2 = 90
x = np.array(x0)
y = np.array(y0)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.90, random_state=666)  # 777  0.9859 0.91967
train_temp_1, train_time_1, train_m_1, test_temp_1, test_time_1, test_m_1, train_alpha_1, test_alpha_1 = [], [], [], [], [], [], [], []
train_temp_2, train_time_2, train_m_2, test_temp_2, test_time_2, test_m_2, train_alpha_2, test_alpha_2 = [], [], [], [], [], [], [], []
for i in range(len(x_train)):
    if x_train[i][2] == m_1 and x_train[i][1] <= 300:
        train_temp_1.append(x_train[i][0])
        train_time_1.append(x_train[i][1])
        train_m_1.append(x_train[i][2])
        train_alpha_1.append(y_train[i])
    if x_train[i][0] == temp_2:
        train_temp_2.append(x_train[i][0])
        train_time_2.append(x_train[i][1])
        train_m_2.append(x_train[i][2])
        train_alpha_2.append(y_train[i])
for i in range(len(x_test)):
    if x_test[i][2] == m_1:
        test_temp_1.append(x_test[i][0])
        test_time_1.append(x_test[i][1])
        test_m_1.append(x_test[i][2])
        test_alpha_1.append(y_test[i])
    if x_test[i][0] == temp_2:
        test_temp_2.append(x_test[i][0])
        test_time_2.append(x_test[i][1])
        test_m_2.append(x_test[i][2])
        test_alpha_2.append(y_test[i])


# training a random forest
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor
regr = ExtraTreesRegressor(n_estimators=60, criterion="absolute_error", max_depth=9, bootstrap=True, max_samples=0.8, random_state=777)
regr.fit(x_train, y_train)                 # training

y_pred = regr.predict(x_train)
y_pred_test = regr.predict(x_test)
print('R^2 (training set): ', regr.score(x_train, y_train))
print('R^2 (test set): ', regr.score(x_test, y_test))

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
    print('R^2: {}, MAE: {}, RMSE: {}, max_-_err: {}, max_+_err: {}'.format(r2_train, mae_train, rmse_train, min_err_train, max_err_train))
    return r2_train, mae_train, rmse_train, min_err_train, max_err_train, err_train

# calculation of metrics on the training and test sets
print('Train. set: ', end='')
r2_train, mae_train, rmse_train, min_err_train, max_err_train, err_train = error_calc(y_train, y_pred)
print('Test set: ', end='')
r2_test, mae_test, rmse_test, min_err_test, max_err_test, err_test = error_calc(y_test, y_pred_test)

# feature importances
importances = regr.feature_importances_
string = 'Impurity feature importance: T = {}%, time = {}%, m = {}%'.format(round(importances[0]*100, 2), round(importances[1]*100, 2), round(importances[2]*100, 2))
print(string)
per_imp = permutation_importance(regr, x_train, y_train, n_repeats=10, random_state=42, n_jobs=1)
string = 'Permutation feature importance (train. set): T = {}, time = {}, m = {}'.format(round(per_imp.importances_mean[0], 3), round(per_imp.importances_mean[1], 3), round(per_imp.importances_mean[2], 3))
print(string)
per_imp_sum = per_imp.importances_mean[0]+per_imp.importances_mean[1]+per_imp.importances_mean[2]
pers_0, pers_1, pers_2 = round(per_imp.importances_mean[0]*100/per_imp_sum, 2), round(per_imp.importances_mean[1]*100/per_imp_sum, 2), round(per_imp.importances_mean[2]*100/per_imp_sum, 2)
string = 'Permutation feature importance, (train. set, normalized values): T = {}%, time = {}%, m = {}%'.format(pers_0, pers_1, pers_2)
print(string)
per_imp = permutation_importance(regr, x_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
string = 'Permutation feature importance (test set): T = {}, time = {}, m = {}'.format(round(per_imp.importances_mean[0], 3), round(per_imp.importances_mean[1], 3), round(per_imp.importances_mean[2], 3))
print(string)
per_imp_sum = per_imp.importances_mean[0]+per_imp.importances_mean[1]+per_imp.importances_mean[2]
pers_0, pers_1, pers_2 = round(per_imp.importances_mean[0]*100/per_imp_sum, 2), round(per_imp.importances_mean[1]*100/per_imp_sum, 2), round(per_imp.importances_mean[2]*100/per_imp_sum, 2)
string = 'Permutation feature importance, (test set, normalized values): T = {}%, time = {}%, m = {}%'.format(pers_0, pers_1, pers_2)
print(string)

# surface 1 (C(NH4HF2=10 wt.%)) based on the RF model prediction (subset 1)
xp = np.arange(30, 100, 2)  # 30, 100, 2 - T, °C
yp = np.arange(0, 365, 5)  # 0, 350, 5 - time, min
y_res = []  # RF prediction
xgrid, ygrid = np.meshgrid(xp, yp)
xp = list((np.array(xgrid)).reshape(len(xgrid)*len(xgrid[0])))
yp = list((np.array(ygrid)).reshape(len(ygrid)*len(ygrid[0])))
X_pred = np.column_stack([xp, yp, np.full_like(xp, m_1)])
y_res = regr.predict(X_pred)
zgrid = np.reshape(np.array(y_res), (len(xgrid), len(xgrid[0])))

# writing surface points to a *.csv file
file = open('res_1_RF.csv', 'w')
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

# plotting a graph for surface 1 (C(NH4HF2=10 wt.%))
if fig_1:
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(xp, yp, y_res, color='green', s=1)  # RF predicted values
    #ax.scatter3D(temp_1, time_1, alpha_1, color='blue', s=60)    # exp
    ax.scatter3D(train_temp_1, train_time_1, train_alpha_1, color='blue', s=5)    # exp for RF train
    ax.scatter3D(test_temp_1, test_time_1, test_alpha_1, color='red', s=5)    # exp for RF test
    ax.set_xlabel('T, °C')
    ax.set_ylabel('t, min')
    ax.set_zlabel('α, fraction')
    plt.show()

    # drawing a heatmap
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

# surface 2 (T = 90 °C) based on the RF model prediction (subset 2)
xp = np.arange(0, 50, 2)
yp = np.arange(0, 350, 12)
y_res = []  # RF prediction
xgrid, ygrid = np.meshgrid(xp, yp)
xp = list((np.array(xgrid)).reshape(len(xgrid)*len(xgrid[0])))
yp = list((np.array(ygrid)).reshape(len(ygrid)*len(ygrid[0])))
X_pred = np.column_stack([np.full_like(xp, temp_2), yp, xp])
y_res = regr.predict(X_pred)
zgrid = np.reshape(np.array(y_res), (len(xgrid), len(xgrid[0])))

# writing surface points to a *.csv file
file = open('res_2_RF.csv', 'w')
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

# plotting a graph for surface 2 (T=90 °C)
if fig_2:
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(xp, yp, y_res, color='green', s=1)  # RF predicted values
    #ax.scatter3D(m_2, time_2, alpha_2, color='blue', s=60)    # exp.
    ax.scatter3D(train_m_2, train_time_2, train_alpha_2, color='blue', s=5)    # exp for RF train
    ax.scatter3D(test_m_2, test_time_2, test_alpha_2, color='red', s=5)    # exp for RF test
    ax.set_xlabel('NH4HF2, wt.%')
    ax.set_ylabel('t, min')
    ax.set_zlabel('α, fraction')
    plt.show()

    # drawing a heatmap
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
if fig_3:
    temp = 60  # prediction of the T = 60 °C surface
    x2 = np.array([[temp, 0, 1], [temp, 0, 2.5], [temp, 0, 10], [temp, 0, 20], [temp, 0, 30], [temp, 0, 40],
                   [temp, 15, 1], [temp, 15, 2.5], [temp, 15, 10], [temp, 15, 20], [temp, 15, 30], [temp, 15, 40],
                   [temp, 20, 1], [temp, 20, 2.5], [temp, 20, 10], [temp, 20, 20], [temp, 20, 30], [temp, 20, 40],
                   [temp, 30, 1], [temp, 30, 2.5], [temp, 30, 10], [temp, 30, 20], [temp, 30, 30], [temp, 30, 40],
                   [temp, 60, 1], [temp, 60, 2.5], [temp, 60, 10], [temp, 60, 20], [temp, 60, 30], [temp, 60, 40],
                   [temp, 90, 1], [temp, 90, 2.5], [temp, 90, 10], [temp, 90, 20], [temp, 90, 30], [temp, 90, 40],
                   [temp, 120, 1], [temp, 120, 2.5], [temp, 120, 10], [temp, 120, 20], [temp, 120, 30], [temp, 120, 40],
                   [temp, 150, 1], [temp, 150, 2.5], [temp, 150, 10], [temp, 150, 20], [temp, 150, 30], [temp, 150, 40],
                   [temp, 180, 1], [temp, 180, 2.5], [temp, 180, 10], [temp, 180, 20], [temp, 180, 30], [temp, 180, 40],
                   [temp, 210, 1], [temp, 210, 2.5], [temp, 210, 10], [temp, 210, 20], [temp, 210, 30], [temp, 210, 40],
                   [temp, 240, 1], [temp, 240, 2.5], [temp, 240, 10], [temp, 240, 20], [temp, 240, 30], [temp, 240, 40],
                   [temp, 270, 1], [temp, 270, 2.5], [temp, 270, 10], [temp, 270, 20], [temp, 270, 30], [temp, 270, 40],
                   [temp, 300, 1], [temp, 300, 2.5], [temp, 300, 10], [temp, 300, 20], [temp, 300, 30], [temp, 300, 40],
                   [temp, 360, 1], [temp, 360, 2.5], [temp, 360, 10], [temp, 360, 20], [temp, 360, 30],
                   [temp, 360, 40]])
    y_pred_3 = regr.predict(x2)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(x2[:, 2], x2[:, 1], y_pred_3, color='green', s=5)  # RF
    ax.set_xlabel('NH4HF2, wt.%')
    ax.set_ylabel('t, min')
    ax.set_zlabel('α, fraction')
    plt.show()

# plot for displaying errors
print("________________________________")
print("Train. set: α(exp), α(RF)")
print(y_train)
print(y_pred)
print("________________________________")
#plt.scatter(y_test, y_pred_2, marker='o', s=5, color='red', label='test set')
print("Test set: α(exp), α(RF)")
print(y_test)
print(y_pred_test)
print("________________________________")
if fig_4:
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    plt.plot([-0.1, 1.1], [-0.1, 1.1], color='black')
    plt.scatter(y_train, y_pred, marker='o', s=9, color='blue', label='train set')
    plt.scatter(y_test, y_pred_test, marker='o', s=9, color='red', label='test set')
    plt.xlabel("α, exp")
    plt.ylabel("α, RF")
    plt.legend()
    plt.subplot(1, 2, 2)
    # defining bins with fixed width
    bin_width = 0.01  # width of one bin
    bin_edges = np.arange(-0.3, 0.3 + bin_width, bin_width)  # from -0.3 to 0.3 with step bin_width
    plt.hist(err_train, bins=bin_edges, color='blue', label='train set')
    plt.hist(err_test, bins=bin_edges, color='red', label='test set')
    plt.xlabel("α(RF)-α(exp), fraction")
    plt.ylabel("Frequency")
    plt.xlim([-0.3, 0.3])
    plt.ylim([0, 35])
    plt.legend()
    plt.tight_layout()
    plt.show()

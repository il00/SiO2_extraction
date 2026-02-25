# visualization of the LS model prediction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LinearRegression
import random
import math

sanity = True   # if the predicted value is less than 0, change it to 0; if greater than 1, change it to 1
seed = 1  # 'default' - determines the randomness of shuffling the points before splitting into training and test sets

def calculate_lms_coof(T, t, m):
    T, t, m = float(T), float(t), float(m)
    return [T, t, m, T*t, T*m, t*m, T*T, t*t, m*m, T*T*t, T*t*t, T*T*m, T*m*m, t*t*m, t*m*m, T*t*m, T*T*T, t*t*t,
               m*m*m, T*T*T*T, T*T*T*t, T*T*T*m, T*T*t*t, T*T*m*m, T*t*t*t, T*m*m*m, T*T*m*t, T*m*t*t, T*m*m*t,
            t*t*t*t, t*t*t*m, t*t*m*m, t*m*m*m, m*m*m*m]

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

# dataset
if True:
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

# shuffle the input data with a fixed random_state
inputs_list, targets_list = shuffle(x0, y0, seed=seed)  # seed='default' - if you set seed=1 or any other number, the result will be reproducible
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
model = LinearRegression()
model.fit(x1, y1)


def function_batch(T, t, m):
    T_new = T.flatten()
    t_new = t.flatten()
    m_new = m.flatten()
    input_data = []
    for i in range(len(T_new)):
        input_data.append(calculate_lms_coof(T_new[i], t_new[i], m_new[i]))
    #input_data = np.column_stack((T.flatten(), t.flatten(), m.flatten()))
    predictions = model.predict(input_data)  #.reshape(T.shape)
    for i in range(len(predictions)):
        if  predictions[i] < 0:
            predictions[i] = 0
        elif predictions[i] > 1:
            predictions[i] = 1
    predictions = predictions.reshape(T.shape)
    return predictions


def update_plot(i, slider, label, ax, canvas, contour_levels_entry):
    fixed_values[i] = round(slider.get())
    label.config(text=f"{fixed_labels[i]} = {fixed_values[i]:.0f}")
    plot_contour(ax, i, fixed_values[i], contour_levels_entry)
    canvas.draw()


def plot_contour(ax, fixed_index, fixed_value, contour_levels_entry):
    ax.clear()
    levels_text = contour_levels_entry.get()
    levels = list(map(float, levels_text.split())) if levels_text else [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                        0.98, 0.99]

    if fixed_index == 0:
        X, Y = np.meshgrid(t_vals, m_vals)
        Z = function_batch(np.full_like(X, fixed_value), X, Y)
        ax.set_xlabel('t')
        ax.set_ylabel('m')
    elif fixed_index == 1:
        X, Y = np.meshgrid(T_vals, m_vals)
        Z = function_batch(X, np.full_like(Y, fixed_value), Y)
        ax.set_xlabel('T')
        ax.set_ylabel('m')
    else:
        X, Y = np.meshgrid(T_vals, t_vals)
        Z = function_batch(X, Y, np.full_like(Y, fixed_value))
        ax.set_xlabel('T')
        ax.set_ylabel('t')

    contour_filled = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')  # RdYlBu_r viridis
    contour_lines = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8)
    ax.set_title(f"{fixed_labels[fixed_index]} = {fixed_value:.0f}")
    #ax.figure.canvas.draw_idle()


# creating a window
root = tk.Tk()
root.title("Degree of SiO2 extraction as a function of T (°C), t (min), NH4HF2 (wt.%)")

# generating data
T_vals = np.linspace(30, 100, 50)
t_vals = np.linspace(0, 360, 50)
m_vals = np.linspace(0, 50, 50)
fixed_values = [65, 150, 23]
fixed_labels = ['T', 't', 'm']

# contour levels input field
frame = tk.Frame(root)
frame.pack()
tk.Label(frame, text="Contour levels (space-separated):").pack(side=tk.LEFT)
contour_levels_entry = tk.Entry(frame, width=50)
contour_levels_entry.pack(side=tk.LEFT)
contour_levels_entry.insert(0, "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.97 0.98")

# plots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

for i in range(3):
    plot_contour(axes[i], i, fixed_values[i], contour_levels_entry)
canvas.draw()

# sliders
slider_frame = tk.Frame(root)
slider_frame.pack(pady=10)

sliders = []
labels = []
for i in range(3):
    frame = tk.Frame(slider_frame)
    frame.pack(side=tk.LEFT, padx=30)

    label = tk.Label(frame, text=f"{fixed_labels[i]} = {fixed_values[i]:.0f}")
    label.pack()

    slider = ttk.Scale(frame, from_=30 if i == 0 else 0, to=100 if i == 0 else (360 if i == 1 else 50),
                       orient='horizontal', length=400)
    slider.set(fixed_values[i])
    slider.pack()


    def increment(idx):
        sliders[idx].set(round(sliders[idx].get()) + 1)
        #update_plot(idx, sliders[idx], labels[idx], axes[idx], canvas, contour_levels_entry)


    def decrement(idx):
        sliders[idx].set(round(sliders[idx].get()) - 1)
        #update_plot(idx, sliders[idx], labels[idx], axes[idx], canvas, contour_levels_entry)


    btn_frame = tk.Frame(frame)
    btn_frame.pack()

    btn_minus = tk.Button(btn_frame, text="-", command=lambda idx=i: decrement(idx))
    btn_minus.pack(side=tk.LEFT)

    btn_plus = tk.Button(btn_frame, text="+", command=lambda idx=i: increment(idx))
    btn_plus.pack(side=tk.LEFT)

    sliders.append(slider)
    labels.append(label)

    slider.config(
        command=lambda val, idx=i: update_plot(idx, sliders[idx], labels[idx], axes[idx], canvas, contour_levels_entry))

root.mainloop()

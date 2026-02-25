# visualization of the ANN model prediction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import tensorflow as tf
from tensorflow.keras import backend as K

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

# loading the ANN model
model = tf.keras.models.load_model("tf_model.keras")


def function_batch(T, t, m):
    input_data = np.column_stack((T.flatten(), t.flatten(), m.flatten()))
    predictions = model.predict(input_data).reshape(T.shape)
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
        ax.set_xlabel('t, min')
        ax.set_ylabel('NH4HF2, wt.%')
    elif fixed_index == 1:
        X, Y = np.meshgrid(T_vals, m_vals)
        Z = function_batch(X, np.full_like(Y, fixed_value), Y)
        ax.set_xlabel('T, °C')
        ax.set_ylabel('NH4HF2, wt.%')
    else:
        X, Y = np.meshgrid(T_vals, t_vals)
        Z = function_batch(X, Y, np.full_like(Y, fixed_value))
        ax.set_xlabel('T, °C')
        ax.set_ylabel('t, min')

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
fixed_labels = ['T, °C', 't, min', 'NH4HF2, wt.%']

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

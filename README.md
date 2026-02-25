# SiO2_extraction

This repository contains machine learning models for predicting the degree of SiO2 extraction from iron ore tailings using aqueous solution of ammonium bifluoride (NH4HF2). The prediction models take into account three key parameters: temperature, reaction time, and NH4HF2 concentration.

## Models Included
Three different approaches have been implemented:
- **Linear Least Squares regression (LS)** — worst results
- **Random Forest Regressor (RF)**
- **Fully Connected Artificial Neural Networks (ANN)** — best predictions

## Repository Structure
*_learning.py — files for model training and testing  
*_3D_plot.py — files for result visualization  
tf_model.keras — contains the trained ANN model  
training_history.json — contains the ANN model's training history  

## Technical Stack
The programs were developed using:
- Python 3.11.1
- TensorFlow 2.13.0 — for neural network implementation
- Scikit-learn 1.2.2 — for LS and Random Forest models
- Matplotlib 3.6.2 — for data visualization

## Installation
1. Install Python from [python.org](https://www.python.org/)
2. Install the required libraries using pip:
```bash
pip install tensorflow==2.13.0
pip install scikit-learn==1.2.2
pip install matplotlib==3.6.2
```

## Detailed Windows Installation Guide for Beginners

### Step 1: Installing Python
1. Go to the official Python website: [https://www.python.org/](https://www.python.org/)
2. Click the "Downloads" button and select Python 3.11.1 for Windows
3. Run the downloaded installer file
4. **IMPORTANT**: In the first installer window, make sure to check **"Add Python to PATH"**
5. Click "Install Now"
6. Wait for the installation to complete and click "Close"

### Step 2: Opening Command Prompt
1. Press the `Win + R` keys on your keyboard simultaneously
2. In the window that opens, type `cmd` and press Enter

### Step 3: Installing Required Libraries
In the command prompt, enter the following commands one by one (press Enter after each command and wait for the installation to complete):
```bash
pip install tensorflow==2.13.0
pip install scikit-learn==1.2.2
pip install matplotlib==3.6.2
```

### Step 4: Download the Repository Files
Download all files from this repository to your computer  
Place them in a single folder (for example: C:\Users\YourName\SiO2_extraction)  

### Step 5: Run One of the Models
Now you can run any of the three models. In the command prompt, navigate to your folder and execute the corresponding command. For example:  
```bash
python ANN_learning.py
python ANN_3D_plot.py
```

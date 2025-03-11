import catboost

# Install CatBoost if not already installed (uncomment if needed)
# !pip install catboost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# -------------------------------
# Data Loading
# -------------------------------
# Replace 'EMGData2.csv' with your actual file name.
df = pd.read_csv("EMGData2.csv", sep=",", engine='python')
df.columns = df.columns.str.strip()
print("Columns in CSV:", df.columns)

# Convert to numeric and drop rows with missing values.
df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
df['Ankle Angle (degrees)'] = pd.to_numeric(df['Ankle Angle (degrees)'], errors='coerce')
df['EMG'] = pd.to_numeric(df['EMG'], errors='coerce')
df = df.dropna(subset=['Time (s)', 'Ankle Angle (degrees)', 'EMG'])

# Extract data arrays.
time = df["Time (s)"].to_numpy()
ankle_angle = df["Ankle Angle (degrees)"].to_numpy()
emg_raw = df["EMG"].to_numpy()

# -------------------------------
# Improved Preprocessing Filters
# -------------------------------
fs = 7500  # Sampling frequency in Hz

# -------------------------------
# Plotting the Data
# -------------------------------
plt.figure(figsize=(14, 8))

# Plot raw vs. filtered EMG.
plt.subplot(2, 1, 1)
plt.plot(time, emg_raw, label="Raw EMG", alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("EMG Amplitude")
plt.title("Raw and Filtered EMG Signal")
plt.legend()
plt.grid(True)

# Plot ankle angle over time.
plt.subplot(2, 1, 2)
plt.plot(time, ankle_angle, color="tab:orange", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Ankle Angle (degrees)")
plt.title("Ankle Angle vs. Time")
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------------
# Autoregressive Coefficients Function
# -------------------------------
def compute_ar_coeffs(x, order=4):
    r_full = np.correlate(x, x, mode='full')
    mid = len(r_full) // 2
    r = r_full[mid:mid+order+1]  # lags 0 to order
    R = np.empty((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = r[abs(i - j)]
    r_right = r[1:order+1]
    try:
        ar_coeffs = np.linalg.solve(R, r_right)
    except np.linalg.LinAlgError:
        ar_coeffs = np.zeros(order)
    return ar_coeffs

# -------------------------------
# Feature Extraction Function
# -------------------------------
def compute_features(window, fs, ar_order=4):
    # --- Time Domain Features ---
    mav  = np.mean(np.abs(window))                          # Mean Absolute Value
    mavs = np.mean(np.abs(np.diff(window)))                 # Mean Absolute Value Slope
    ssi  = np.sum(window**2)                                # Simple Square Integral
    var  = np.var(window)                                   # Variance
    rms  = np.sqrt(np.mean(window**2))                      # Root Mean Square
    wl   = np.sum(np.abs(np.diff(window)))                  # Waveform Length

    # --- Frequency Domain Features ---
    fft_result = np.fft.rfft(window)
    power_spectrum = np.abs(fft_result)**2
    total_power = np.sum(power_spectrum)                    # Total Power (overall energy)
    if total_power == 0:
        total_power = 1e-10
    freqs = np.fft.rfftfreq(len(window), d=1/fs)

    # Frequency Median (MDF): frequency where cumulative power reaches 50% of total power.
    cumulative_power = np.cumsum(power_spectrum)
    idx_mdf = np.searchsorted(cumulative_power, total_power / 2)
    mdf = freqs[idx_mdf] if idx_mdf < len(freqs) else freqs[-1]

    # Modified Median Frequency (MMDF): frequency where cumulative power reaches 60% of total power.
    idx_mmdf = np.searchsorted(cumulative_power, 0.6 * total_power)
    mmdf = freqs[idx_mmdf] if idx_mmdf < len(freqs) else freqs[-1]

    # --- Autoregressive Coefficients ---
    ar_coeffs = compute_ar_coeffs(window, order=ar_order)  # Array of length 'ar_order'

    # Combine features into one vector:
    # [MAV, MAVS, SSI, Variance, RMS, WL, Total Power, MDF, MMDF, AR1, AR2, AR3, AR4]
    features = np.concatenate([
        np.array([mav, mavs, ssi, var, rms, wl, total_power, mdf, mmdf]),
        ar_coeffs
    ])
    return features

def compute_feature_matrix(emg_data, window_size, fs, ar_order=4):
    num_windows = len(emg_data) // window_size
    trimmed_data = emg_data[:num_windows * window_size]  # Use only complete windows
    windows = trimmed_data.reshape((num_windows, window_size))
    feature_matrix = np.array([compute_features(window, fs, ar_order) for window in windows])
    return feature_matrix

def compute_target_for_dataset(angle_data, window_size):
    num_windows = len(angle_data) // window_size
    trimmed_data = angle_data[:num_windows * window_size]
    windows = trimmed_data.reshape((num_windows, window_size))
    target_values = np.mean(windows, axis=1)
    return target_values

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
df = pd.read_csv("EMGData2.csv", sep=",", engine='python')
df.columns = df.columns.str.strip()
print("Columns in CSV:", df.columns)

df['EMG'] = pd.to_numeric(df['EMG'], errors='coerce')
df['Ankle Angle (degrees)'] = pd.to_numeric(df['Ankle Angle (degrees)'], errors='coerce')
df = df.dropna(subset=['EMG', 'Ankle Angle (degrees)'])

emg_data = df["EMG"].to_numpy()
angle_data = df["Ankle Angle (degrees)"].to_numpy()

# Optionally, apply a bandpass filter if desired.
# For this version, you may remove or comment out the filtering step.
fs = 7500  # Sampling rate in Hz

# -------------------------------
# Feature Extraction and Target Aggregation
# -------------------------------
window_size = 100   # Using 100 data points per window
ar_order = 4        # AR coefficient order

X = compute_feature_matrix(emg_data, window_size, fs, ar_order)
y = compute_target_for_dataset(angle_data, window_size)

# Replace any NaN values with 0.
X = np.nan_to_num(X, nan=0.0)

print("Feature matrix shape:", X.shape)
print("First row of features:\n", X[0])
print("First 10 target angles:", y[:10])

# -------------------------------
# Train/Test Split (70% training, 30% testing)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# -------------------------------
# Model Training and Evaluation
# -------------------------------

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("\nLinear Regression:")
print("  MSE =", mse_lr)
print("  R² =", r2_lr, "-> Accuracy: {:.2f}%".format(r2_lr*100))

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest Regressor:")
print("  MSE =", mse_rf)
print("  R² =", r2_rf, "-> Accuracy: {:.2f}%".format(r2_rf*100))

# Model 3: Support Vector Regressor (SVR)
svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print("\nSupport Vector Regressor (SVR):")
print("  MSE =", mse_svr)
print("  R² =", r2_svr, "-> Accuracy: {:.2f}%".format(r2_svr*100))

# Model 4: K-Neighbors Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print("\nK-Neighbors Regressor:")
print("  MSE =", mse_knn)
print("  R² =", r2_knn, "-> Accuracy: {:.2f}%".format(r2_knn*100))

# Model 5: Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)
print("\nGradient Boosting Regressor:")
print("  MSE =", mse_gbr)
print("  R² =", r2_gbr, "-> Accuracy: {:.2f}%".format(r2_gbr*100))

# Model 6: Extra Trees Regressor
etr_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
etr_model.fit(X_train, y_train)
y_pred_etr = etr_model.predict(X_test)
mse_etr = mean_squared_error(y_test, y_pred_etr)
r2_etr = r2_score(y_test, y_pred_etr)
print("\nExtra Trees Regressor:")
print("  MSE =", mse_etr)
print("  R² =", r2_etr, "-> Accuracy: {:.2f}%".format(r2_etr*100))

# Model 7: XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("\nXGBoost Regressor:")
print("  MSE =", mse_xgb)
print("  R² =", r2_xgb, "-> Accuracy: {:.2f}%".format(r2_xgb*100))

# Model 8: LightGBM Regressor
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)
print("\nLightGBM Regressor:")
print("  MSE =", mse_lgb)
print("  R² =", r2_lgb, "-> Accuracy: {:.2f}%".format(r2_lgb*100))

# Model 9: CatBoost Regressor
cat_model = cb.CatBoostRegressor(iterations=100, verbose=0, random_state=42)
cat_model.fit(X_train, y_train)
y_pred_cat = cat_model.predict(X_test)
mse_cat = mean_squared_error(y_test, y_pred_cat)
r2_cat = r2_score(y_test, y_pred_cat)
print("\nCatBoost Regressor:")
print("  MSE =", mse_cat)
print("  R² =", r2_cat, "-> Accuracy: {:.2f}%".format(r2_cat*100))

# Model 10: Bayesian Ridge Regression
from sklearn.linear_model import BayesianRidge
bayes_model = BayesianRidge()
bayes_model.fit(X_train, y_train)
y_pred_bayes = bayes_model.predict(X_test)
mse_bayes = mean_squared_error(y_test, y_pred_bayes)
r2_bayes = r2_score(y_test, y_pred_bayes)
print("\nBayesian Ridge Regression:")
print("  MSE =", mse_bayes)
print("  R² =", r2_bayes, "-> Accuracy: {:.2f}%".format(r2_bayes*100))

# -------------------------------
# Deep Learning Model: Fully Connected Network
# -------------------------------
deep_model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)  # Output layer for regression
])
deep_model.compile(optimizer=Adam(learning_rate=0.00001), loss='mse')
deep_model.fit(X_train, y_train, epochs=900, batch_size=66, verbose=0)
y_pred_deep = deep_model.predict(X_test).flatten()
mse_deep = mean_squared_error(y_test, y_pred_deep)
r2_deep = r2_score(y_test, y_pred_deep)
print("\nDeep Learning Model:")
print("  MSE =", mse_deep)
print("  R² =", r2_deep, "-> Accuracy: {:.2f}%".format(r2_deep*100))
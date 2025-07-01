import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.set_page_config(page_title="Smart Grid Load Monitoring", layout="wide")
st.title("⚡ Smart Grid Load Monitoring Dashboard")

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload your 'smart_grid_dataset.csv' file", type=["csv"])

# --- Helper: Evaluation Function ---
def evaluate_model(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    st.subheader(f"📊 {model_name} Evaluation")
    st.write(f"**MAE**: {mae:.4f}")
    st.write(f"**MSE**: {mse:.4f}")
    st.write(f"**RMSE**: {rmse:.4f}")
    st.write(f"**R2 Score**: {r2:.4f}")

    st.line_chart(pd.DataFrame({'Actual': y_true.values, 'Predicted': y_pred}, index=y_true.index))

    return mae, mse, rmse, r2

# --- Main App ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded and loaded successfully!")

    # Preprocessing
    df = df.ffill().bfill()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df = df.sort_values(by='Timestamp')
    df = df.drop_duplicates(subset='Timestamp')
    df.set_index('Timestamp', inplace=True)

    # Target
    target = 'Power Consumption (kW)'
    if target not in df.columns:
        st.error("❌ 'Power Consumption (kW)' not found in the dataset.")
        st.stop()

    # Feature Engineering
    for lag in range(1, 25):
        df[f'{target}_lag_{lag}'] = df[target].shift(lag)
    df[f'{target}_rolling_mean_24h'] = df[target].rolling(24).mean()
    df[f'{target}_rolling_std_24h'] = df[target].rolling(24).std()

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # Model Selection
    st.sidebar.header("🔍 Model Options")
    model_option = st.sidebar.selectbox("Choose a model", ("Linear Regression", "Random Forest", "LSTM"))

    if model_option == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        evaluate_model(y_test, preds, "Linear Regression")

    elif model_option == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        evaluate_model(y_test, preds, "Random Forest")

        # SHAP Interpretation
        st.subheader("🔎 SHAP Interpretation (Top Features)")
        explainer = shap.Explainer(model, X_train_scaled)
        shap_values = explainer(X_test_scaled, check_additivity=False)

        fig1 = shap.plots.beeswarm(shap_values, max_display=10, show=False)
        st.pyplot(fig=plt.gcf())

        fig2 = shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig=plt.gcf())

    elif model_option == "LSTM":
        X_train_lstm = np.reshape(X_train_scaled.values, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = np.reshape(X_test_scaled.values, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_train_lstm, y_train, epochs=10, verbose=0)

        lstm_preds = lstm_model.predict(X_test_lstm).flatten()
        evaluate_model(y_test, lstm_preds, "LSTM")

else:
    st.info("📂 Please upload the 'smart_grid_dataset.csv' file to start.")

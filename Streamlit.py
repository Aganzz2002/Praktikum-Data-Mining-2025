import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns  # Added missing import
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def clean_volume(volume):
    if isinstance(volume, str):
        return float(volume.replace('.', ''))
    return float(volume)

def main():
    st.title('📈 Analisis Saham dengan Regresi')
    
    uploaded_file = st.file_uploader("Upload data saham (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['Date'], dayfirst=True)
            
            # Clean Volume column more robustly
            df['Volume'] = df['Volume'].apply(clean_volume)
            df = df.sort_values('Date')
            
            # Tab setup
            tab1, tab2 = st.tabs(["Regresi Linear", "Regresi Logistik"])
            
            with tab1:
                st.header("Regresi Linear untuk Prediksi Harga")
                
                # Feature engineering
                df['Days'] = (df['Date'] - df['Date'].min()).dt.days
                df['MA_7'] = df['Close'].rolling(window=7).mean()
                df['MA_21'] = df['Close'].rolling(window=21).mean()
                df['Price_Change'] = df['Close'].pct_change()
                df = df.dropna()
                
                # Feature selection
                features = st.multiselect(
                    "Pilih fitur prediktor:",
                    options=['Days', 'MA_7', 'MA_21', 'Volume', 'Price_Change'],
                    default=['Days', 'MA_7']
                )
                
                # Ensure numeric data types
                X = df[features].astype(np.float64)
                y = df['Close'].astype(np.float64)
                
                # Data splitting
                test_size = st.slider("Persentase data testing:", 10, 40, 20)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, shuffle=False
                )
                
                # Model training with error handling
                try:
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Prediction and evaluation
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    col1.metric("MSE (Mean Squared Error)", f"{mse:.2f}")
                    col2.metric("R² Score", f"{r2:.2f}")
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.plot(df['Date'], df['Close'], label='Aktual', alpha=0.5)
                    ax.plot(X_test.index, y_pred, label='Prediksi', color='red')
                    ax.set_title('Perbandingan Harga Aktual vs Prediksi')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Show coefficients
                    st.subheader("Koefisien Model:")
                    for feat, coef in zip(features, model.coef_):
                        st.write(f"{feat}: {coef:.4f}")
                    st.write(f"Intercept: {model.intercept_:.2f}")
                
                except Exception as e:
                    st.error(f"Error dalam pelatihan model: {str(e)}")
                    st.error("Pastikan data input valid dan tidak mengandung nilai NaN atau infinity")

            with tab2:
                st.header("Regresi Logistik untuk Prediksi Arah Harga")
                
                # Prepare target
                df['Target'] = (df['Close'].pct_change() > 0).astype(int)
                df = df.dropna()
                
                # Feature selection
                features_log = st.multiselect(
                    "Pilih fitur prediktor:",
                    options=['MA_7', 'MA_21', 'Volume', 'Price_Change'],
                    default=['MA_7', 'Volume'],
                    key='log_features'
                )
                
                # Ensure numeric data types
                X_log = df[features_log].astype(np.float64)
                y_log = df['Target'].astype(np.int64)
                
                # Data splitting
                X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
                    X_log, y_log, test_size=0.2, shuffle=False
                )
                
                # Model training with error handling
                try:
                    log_model = LogisticRegression()
                    log_model.fit(X_train_log, y_train_log)
                    
                    # Prediction and evaluation
                    y_pred_log = log_model.predict(X_test_log)
                    accuracy = accuracy_score(y_test_log, y_pred_log)
                    
                    st.metric("Akurasi Model", f"{accuracy:.2%}")
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test_log, y_pred_log)
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm)
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('Actual')
                    st.pyplot(fig_cm)
                    
                    # Feature importance
                    st.subheader("Pengaruh Fitur:")
                    for feat, coef in zip(features_log, log_model.coef_[0]):
                        st.write(f"{feat}: {coef:.4f}")
                
                except Exception as e:
                    st.error(f"Error dalam pelatihan model logistik: {str(e)}")
        
        except Exception as e:
            st.error(f"Error dalam memproses file: {str(e)}")
            st.error("Pastikan format file CSV sesuai dan memiliki kolom yang diperlukan")

if __name__ == '__main__':
    main()
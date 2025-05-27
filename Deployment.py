import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from datetime import timedelta

st.title("Monthly Trouble Ticket Dashboard with Forecasting")

uploaded_files = st.file_uploader("Upload Excel Files", type=['xls', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    try:
        # Combine multiple Excel files
        df_list = [pd.read_excel(file) for file in uploaded_files]
        df = pd.concat(df_list, ignore_index=True)

        # Clean column headers
        df.columns = df.columns.str.replace(' ', '', regex=False)

        # Select date column
        date_column = st.selectbox("Select the date column:", df.columns)
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])

        # Optional CaseTypeName filter
        case_type_col = st.selectbox("Select the CaseTypeName column (or skip):", options=['None'] + list(df.columns))
        if case_type_col != 'None':
            selected_types = st.multiselect(
                f"Filter by values in {case_type_col}:",
                options=df[case_type_col].dropna().unique(),
                default=df[case_type_col].dropna().unique()
            )
            df = df[df[case_type_col].isin(selected_types)]

        # Add year and month for visualization
        df['Year'] = df[date_column].dt.year
        df['Month'] = df[date_column].dt.month_name().str[:3]

        # Group and plot monthly cases
        monthly = df.groupby(['Year', 'Month']).size().unstack(fill_value=0)
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        present_months = [m for m in month_order if m in monthly.columns]
        monthly = monthly[present_months]

        st.subheader("Monthly Trouble Tickets per Year")
        fig, ax = plt.subplots(figsize=(10, 5))
        monthly.T.plot(kind='line', marker='o', ax=ax)
        ax.set_title('Monthly Case Distribution per Year')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Cases')
        ax.set_xticks(range(len(present_months)))
        ax.set_xticklabels(present_months)
        ax.grid(True)
        st.pyplot(fig)

        # ---- Forecasting Section ----
        st.subheader("Forecasting Next 3 Months")

        # Create daily case counts
        df_daily = df.groupby(df[date_column].dt.date).size().reset_index(name='Count')
        df_daily['date'] = pd.to_datetime(df_daily[date_column])
        df_daily = df_daily[['date', 'Count']].sort_values('date')

        # Create lag features
        for lag in range(1, 8):
            df_daily[f'lag_{lag}'] = df_daily['Count'].shift(lag)
        df_daily.dropna(inplace=True)

        # Train on all available data
        X = df_daily.drop(['date', 'Count'], axis=1)
        y = df_daily['Count']

        models = {
            'XGBoost': XGBRegressor(random_state=42),
            'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42),
        }

        def calculate_mape(y_true, y_pred):
            epsilon = 1e-10
            return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

        best_model = None
        best_model_name = ''
        best_mape = float('inf')

        for name, model in models.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            mape = calculate_mape(y, y_pred)
            if mape < best_mape:
                best_mape = mape
                best_model = model
                best_model_name = name

        st.success(f"Best model: {best_model_name} with MAPE: {best_mape:.2f}%")

        # Forecast next 3 months (approx. 90 days)
        last_row = df_daily.iloc[-1]
        future_dates = [last_row['date'] + timedelta(days=i) for i in range(1, 91)]
        forecast_data = []
        recent_lags = list(last_row[[f'lag_{i}' for i in range(1, 8)]].values)

        for _ in range(90):
            X_input = np.array(recent_lags).reshape(1, -1)
            pred = best_model.predict(X_input)[0]
            forecast_data.append(pred)
            recent_lags = [pred] + recent_lags[:-1]

        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_data
        })

        # Combine with historical data
        full_plot = pd.concat([
            df_daily[['date', 'Count']].rename(columns={'Count': 'value'}).assign(type='actual'),
            forecast_df.rename(columns={'forecast': 'value'}).assign(type='forecast')
        ])

        # Plot forecast
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        for label, grp in full_plot.groupby('type'):
            ax2.plot(grp['date'], grp['value'], label=label)
        ax2.set_title("Daily Case Forecast (Next 3 Months)")
        ax2.set_ylabel("Cases")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")

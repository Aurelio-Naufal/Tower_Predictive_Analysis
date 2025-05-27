import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from datetime import timedelta
from io import BytesIO, StringIO

# Set page config at the top (optional)
st.set_page_config(page_title="Monthly Trouble Ticket Dashboard with Forecasting")

st.title("Monthly Trouble Ticket Dashboard with Forecasting")

@st.cache_data(show_spinner=True)
def load_data(files):
    df_list = []
    for file in files:
        file_ext = file.name.split('.')[-1].lower()
        if file_ext in ['xls', 'xlsx']:
            bytes_data = BytesIO(file.read())
            df_temp = pd.read_excel(bytes_data)
        elif file_ext == 'csv':
            string_data = StringIO(file.read().decode('utf-8'))
            df_temp = pd.read_csv(string_data)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        df_list.append(df_temp)
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.replace(' ', '', regex=False)
    return df

# Upload files
uploaded_files = st.file_uploader(
    "Upload Excel or CSV Files", 
    type=['xls', 'xlsx', 'csv'], 
    accept_multiple_files=True
)

# Initialize session_state 'df' if not existing
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Load & process button
if uploaded_files:
    if st.button("Load & Process Data"):
        try:
            df_loaded = load_data(uploaded_files)
            st.session_state['df'] = df_loaded  # Save to session_state
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load data: {e}")

# Use dataframe from session_state
df = st.session_state['df']

if df is not None:
    # Select date column
    date_column = st.selectbox("Select the date column:", df.columns)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])

    # Optional CaseTypeName filter
    case_type_col = st.selectbox("Select the Case Type column (or skip):", options=['None'] + list(df.columns))
    if case_type_col != 'None':
        selected_types = st.multiselect(
            f"Filter by values in {case_type_col}:",
            options=df[case_type_col].dropna().unique(),
            default=df[case_type_col].dropna().unique()
        )
    else:
        selected_types = None

    # Optional RegionName filter
    region_col = st.selectbox("Select the Region/Province column (or skip):", options=['None'] + list(df.columns))
    if region_col != 'None':
        selected_regions = st.multiselect(
            f"Filter by values in {region_col}:",
            options=df[region_col].dropna().unique(),
            default=df[region_col].dropna().unique()
        )
    else:
        selected_regions = None
    #Operator Column
    operator_col = st.selectbox("Select the Operator/Operator Group column (or skip):", options=['None'] + list(df.columns))
    if operator_col != 'None':
        selected_operators = st.multiselect(
            f"Filter by values in {operator_col}:",
            options=df[operator_col].dropna().unique(),
            default=df[operator_col].dropna().unique()
        )
    else:
        selected_operators = None
    
    # Model selection
    model_choice = st.selectbox("Choose forecasting model:", ['XGBoost', 'CatBoost', 'LightGBM'])

    if st.button("Run Forecast"):
        try:
            # Use a filtered copy
            df_filtered = df.copy()
            if selected_types:
                df_filtered = df_filtered[df_filtered[case_type_col].isin(selected_types)]
            if selected_regions:
                df_filtered = df_filtered[df_filtered[region_col].isin(selected_regions)]
            if selected_operators:
                df_filtered = df_filtered[df_filtered[operator_col].isin(selected_operators)]

            # Add year and month for visualization
            df_filtered['Year'] = df_filtered[date_column].dt.year
            df_filtered['Month'] = df_filtered[date_column].dt.month_name().str[:3]

            # Group and plot monthly cases
            monthly = df_filtered.groupby(['Year', 'Month']).size().unstack(fill_value=0)
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            present_months = [m for m in month_order if m in monthly.columns]
            monthly = monthly[present_months]

            monthly_long = monthly.T.reset_index().melt(id_vars='Month', var_name='Year', value_name='Count')
            # Note: after melt, columns are Month, Year, Count

            st.subheader("Monthly Trouble Tickets per Year")
            fig = px.line(
                monthly_long,
                x='Month',
                y='Count',
                color='Year',
                markers=True,
                title="Monthly Case Distribution per Year"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- Forecasting Section ----
            st.subheader("Forecasting Next 3 Months")

            # Daily case counts
            df_daily = df_filtered.groupby(df_filtered[date_column].dt.date).size().reset_index(name='Count')
            df_daily.rename(columns={date_column: 'date'}, inplace=True)  # Just in case
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily[['date', 'Count']].sort_values('date')

            # Create lag features
            for lag in range(1, 8):
                df_daily[f'lag_{lag}'] = df_daily['Count'].shift(lag)
            df_daily.dropna(inplace=True)

            # Prepare data
            X = df_daily.drop(['date', 'Count'], axis=1)
            y = df_daily['Count']

            model_map = {
                'XGBoost': XGBRegressor(random_state=42),
                'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
                'LightGBM': lgb.LGBMRegressor(random_state=42),
            }

            model = model_map[model_choice]
            model.fit(X, y)

            # Forecast next 90 days
            last_row = df_daily.iloc[-1]
            future_dates = [last_row['date'] + timedelta(days=i) for i in range(1, 91)]
            forecast_data = []
            recent_lags = list(last_row[[f'lag_{i}' for i in range(1, 8)]].values)

            for _ in range(90):
                X_input = np.array(recent_lags).reshape(1, -1)
                pred = model.predict(X_input)[0]
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

            if len(full_plot) > 1000:
                full_plot = full_plot.iloc[::2]

            fig2 = px.line(
                full_plot,
                x='date',
                y='value',
                color='type',
                title="Daily Case Forecast (Next 3 Months)",
                labels={"value": "Cases", "date": "Date"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("Please upload data and click 'Load & Process Data' to begin.")


# Import Dependancies
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


# Data Collection
START = "2005-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Market Deep Learning App')

stocks = ('^BSESN','GOOG', 'AAPL', 'MSFT', 'GME')

selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

train_dates = pd.to_datetime(data['Date'])
cols = list(data)[1:6]
df_for_training = data[cols].astype(float)


st.subheader('Raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()
st.subheader('Data Cleaning')
# Check for missing values
with st.echo():
        #Check for Missing Values
          df = data.loc[:,['Date','Close']]
          (df.Close.isna().sum())
          df_missing_date = df.loc[df.Close.isna() == True]
          df_missing_date.loc[:, ['Date']]

        # Replcase missing value with interpolation
          df.Close.interpolate(inplace=True)
          df = df.drop('Date', axis=1)

st.subheader('Data Transformation')
with st.echo():
        # Split train data and test data
        whole_data = int(len(df) * 1)
        train_size = int(len(df) * 0.8)
        # Use iloc to select a number of rows
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

with st.echo():
       # Scale the data
       # The input to scaler.fit -> array-like, sparse matrix, dataframe of shape (n_samples, n_features)
       from sklearn.preprocessing import MinMaxScaler
       scaler = MinMaxScaler().fit(train_data)
       train_scaled = scaler.transform(train_data)
       test_scaled = scaler.transform(test_data)

with st.echo():
        # Create input dataset
        # Th input shape should be [samples, time steps, features]
        def create_dataset(X, look_back=1):
            Xs, ys = [], []

            for i in range(len(X) - look_back):
                v = X[i:i + look_back]
                Xs.append(v)
                ys.append(X[i + look_back])

            return np.array(Xs), np.array(ys)

        X_train, y_train = create_dataset(train_scaled, 30)
        X_test, y_test = create_dataset(test_scaled, 30)

        X_train.shape
        y_train.shape
        X_test.shape
        y_test.shape

        X_test[:50].shape

st.sidebar.title('Hyperparameters')
n_neurons = st.sidebar.slider('Neurons', 1, 100, 50)
l_rate = st.sidebar.selectbox('Learning Rate', (0.0001, 0.001, 0.01), 1)
n_epochs = st.sidebar.number_input('Number of Epochs', 1, 50, 20)

st.subheader('Build the Bidirectional Deep Learning Model & Fit the Model')
with st.echo():
        # Import Dependancies
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import Sequential, layers, callbacks
        from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

        # Build The Model
        model = Sequential()
        # Input layer
        model.add(Bidirectional(LSTM(n_neurons, activation='relu', return_sequences=False),
                                input_shape=(X_train.shape[1], X_train.shape[2])))

        # Hidden layer
        # model.add(Bidirectional(LSTM(n_neurons)))

        # Output Layer
        model.add(Dense(1, activation='linear', name='Close'))

with st.echo():
        # Compile The Model
        opt = keras.optimizers.Adam(l_rate)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
train = st.button('Train Model')
if train:
    with st.spinner('Training Model…'):
        with st.echo():
            model.summary(print_fn=lambda x: st.write('{}'.format(x)))
history = model.fit(
                X_train,
                y_train,
                epochs=n_epochs,
                validation_split=0.2,batch_size=16, shuffle=False,callbacks=[early_stop]
            )
st.success('Model Training Complete!')

y_test = scaler.inverse_transform(y_test)
y_train = scaler.inverse_transform(y_train)

st.subheader('Plot the Model Loss')
with st.echo():
    st.line_chart(pd.DataFrame(history.history))


st.subheader('Making Model Predictions on Test Data & New Data Set')
with st.echo():
        X_new = X_test
        predictions = model.predict(X_new)
        predictions = scaler.inverse_transform(predictions)
predictions

y_test[:10]

with st.echo():
    # Plot test data vs prediction
    plt.figure(figsize=(10, 6))

    range_future = len(predictions)

    plt.plot(np.arange(range_future), np.array(y_test), label='Test data')
    plt.plot(np.arange(range_future), np.array(predictions), label='Prediction')

    plt.title('Test data vs prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)')
    plt.ylabel('Daily Closing Price of Stock')
    st.pyplot()



with st.echo():
       # Make New Test Data
       # Select 60 days of data from test data
       new_data = test_data.iloc[100:160]
       # Scale the input
       scaled_data = scaler.transform(new_data)


       # Reshape the input
       def create_dataset(X, look_back=1):
           Xs = []
           for i in range(len(X) - look_back):
               v = X[i:i + look_back]
               Xs.append(v)

           return np.array(Xs)


       X_30 = create_dataset(scaled_data, 30)

with st.echo():
        # Make prediction for new data
        predictions1 = model.predict(X_30)
        predictions1 = scaler.inverse_transform(predictions1)


st.subheader('Evaluate The Model Performance')
with st.echo():
    # Calculate MAE and RMSE
     errors = predictions - y_test
     mse = np.square(errors).mean()
     rmse = np.sqrt(mse)
     mae = np.abs(errors).mean()


     mae
     rmse

scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

trainX = []
trainY = []

n_future = 1  # Number of days we want to predict into the future
n_past = 14  # Number of past days we want to use to predict the future

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=False),
                        input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# fit model
history = model.fit(trainX, trainY, epochs=25, batch_size=16, validation_split=0.1, verbose=1, shuffle=False,
                    callbacks=[early_stop])

n_future = 90  # Redefining n_future to extend prediction dates beyond original n_future dates...
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

forecast = model.predict(trainX[-n_future:])  # forecast

forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

# Convert timestamp to date
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Close': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

original = data[['Date', 'Close']]
original['Date'] = pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2020-4-1']


st.subheader('Most Important: PREDICTING THE FUTURE')
with st.echo():
    def plot_future_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=original['Date'], y=original['Close'], name="Historical Trend"))
        fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Close'], name="Forecast"))
        fig.layout.update(title_text='Future Price Direction', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
plot_future_data()
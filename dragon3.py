import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Embedding, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt


# Function to train linear regression model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to train random forest model
def train_random_forest(X_train, y_train, num_estimators, max_depth, min_samples_split):
    model = RandomForestRegressor(n_estimators=num_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    return model


# Function to train RNN model
def train_rnn(X_train, y_train, rnn_units, epochs, recurrent_dropout):
    model = Sequential()
    model.add(SimpleRNN(units=rnn_units, activation='relu', recurrent_dropout=recurrent_dropout, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    return model

# Function to train CNN model
def train_cnn(X_train, y_train, num_filters, kernel_size, dropout_rate, epochs):
    model = Sequential()
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    return model


# Function to encode categorical features
def encode_categorical_features(df, categorical_columns):
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

sample_data = {
    'Hydrogen_in_kg': [5.5333, 7.1804, 6.0674],
    'Temp_Celsius': [70.9763, 74.3038, 72.0553],
    'Pressure_Bar': [38.7186, 46.2264, 56.7003],
    # Add more sample columns as needed
}

# Initialize sample DataFrame as the default
df = pd.DataFrame(sample_data)



# Set page configuration
st.set_page_config(page_title="🚀🐉 Train Your Dragon Model 🐉🚀", page_icon="🐉", layout="wide")

# Sidebar with GIFs
st.sidebar.markdown('<div style="display: flex; justify-content: space-between;">'
                    '<img src="https://media.giphy.com/media/0YwHADEH90Mjii6qHV/giphy.gif" width="150" height="150">'
                    '<img src="https://media.giphy.com/media/fe6NAMLeTWZq3v9Nmg/giphy.gif" width="150" height="150">'
                    '</div>', unsafe_allow_html=True)
st.sidebar.title("Model Selection and Hyperparameter Tuning")

# Model selection
model_selection = st.sidebar.selectbox("Select a Model", ["📈 Linear Regression", "📚 Random Forest", "📷 CNN", "📜 RNN"])

if model_selection in ["📷 CNN", "📜 RNN"]:
    epochs = st.sidebar.slider("Epochs 🔄", 1, 100, 10)
else:
    epochs = 10  # Default value

# Additional hyperparameters for each model
# Additional hyperparameters for each model
if model_selection == "📈 Linear Regression":
    st.sidebar.subheader("Linear Regression Hyperparameters")
    # Add any additional hyperparameters if needed
elif model_selection == "📚 Random Forest":
    st.sidebar.subheader("Random Forest Hyperparameters")
    num_estimators = st.sidebar.slider("Number of Estimators 🌲", 10, 100, 50, help="The number of trees in the forest.")
    max_depth = st.sidebar.slider("Max Depth 🏞️", 1, 20, 5, help="The maximum depth of the trees in the forest.")
    min_samples_split = st.sidebar.slider("Min Samples Split 🌱", 2, 20, 2, help="The minimum number of samples required to split an internal node.")
elif model_selection == "📷 CNN":
    st.sidebar.subheader("CNN Hyperparameters")
    num_filters = st.sidebar.slider("Number of Filters 🖼️", 32, 256, 128, help="The number of filters in the convolutional layer.")
    kernel_size = st.sidebar.slider("Kernel Size 🧩", 3, 9, 5, help="The size of the convolutional kernel.")
    dropout_rate = st.sidebar.slider("Dropout Rate 🚶‍♂️", 0.0, 0.5, 0.2, help="The fraction of input units to drop during training.")
elif model_selection == "📜 RNN":
    st.sidebar.subheader("RNN Hyperparameters")
    rnn_units = st.sidebar.slider("RNN Units 📜", 10, 100, 50, help="The number of RNN units in the layer.")
    recurrent_dropout = st.sidebar.slider("Recurrent Dropout 🚶‍♂️", 0.0, 0.5, 0.2, help="Fraction of the units to drop for the recurrent dropout.")

# Dataset upload
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file 📂", type=["csv"])

# Main content
# Use only the markdown for the heading to avoid duplicates
st.markdown("""
    <style>
    .heading {
        font-family: 'Georgia', serif; /* Custom font */
        font-size: 32px; /* Increased font size */
        font-weight: bold; /* Optional: Makes the font bold */
    }
    </style>
    <div style="display: flex; align-items: center;">
        <h1 class="heading" style="margin-right: 10px; display: inline-block;">🚀🐉 Train Your Dragon Model 🐉🚀</h1>
        <img src="https://media.tenor.com/xHErfSBH1hEAAAAj/spyro-the-dragon-purple-dragon.gif" style="width: 50px; height: 50px;">
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.write(df.head())

    st.subheader("Column Names in the Dataset:")
    st.write(df.columns.tolist())

    target_variable = st.selectbox("Select the Target Variable", df.columns)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    encode_categorical = st.checkbox("Encode Categorical Features")

    if encode_categorical:
        df = encode_categorical_features(df, categorical_columns)

    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Button to train model
# Placeholder for displaying modal
modal_placeholder = st.empty()

# Button to train model
# Button to train model
if st.sidebar.button("Train Model 🚀"):
    if model_selection == "📈 Linear Regression":
        trained_model = train_linear_regression(X_train, y_train)
        display_graphs = True
    elif model_selection == "📚 Random Forest":
        trained_model = train_random_forest(X_train, y_train, num_estimators, max_depth, min_samples_split)
        display_graphs = True
    elif model_selection == "📷 CNN":
        trained_model = train_cnn(X_train, y_train, num_filters, kernel_size, dropout_rate, epochs)
        display_graphs = False
    elif model_selection == "📜 RNN":
        trained_model = train_rnn(X_train, y_train, rnn_units, epochs, recurrent_dropout)
        display_graphs = False

    y_pred = trained_model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    st.subheader("Model Evaluation:")
    st.write(f"Mean Squared Error: {mse}")

    if display_graphs:
        # Calculate residuals
        residuals = y_test.values.flatten() - y_pred

        # Residual plot
        st.subheader("Residual Plot:")
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)
    
    # Display balloons after training is complete
    st.balloons()


    # Scatter Plot
    st.subheader("Scatter Plot:")
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    st.pyplot(plt)



# Embed Vimeo video
video_url = "https://player.vimeo.com/external/562638295.sd.mp4?s=ed5f30580cfde44ecee638581446f5e61c2690eb&profile_id=164&oauth2_token_id=57447761"
st.markdown(f'<video src="{video_url}" autoplay loop controls style="width:100%"></video>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.text("© S INDUSTRIES 🚀📊🧮🔍🚧🧱📦🖼️🧩📜")

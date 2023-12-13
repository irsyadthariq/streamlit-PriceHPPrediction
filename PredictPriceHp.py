import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Pembuatan dataframe
st.title("Prediksi Harga HP")
df_hp = pd.read_csv("phone_price.csv")

#pre-processing perubahan tipe data string ke tipe data float untuk memastikan konsistensi tipe data
df_hp['storage'] = df_hp['storage'].astype(float)
df_hp['ram'] = df_hp['ram'].astype(float)
df_hp['screenSize'] = df_hp['screenSize'].astype(float)
df_hp['frontCamera'] = df_hp['frontCamera'].astype(float)
df_hp['rearCamera'] = df_hp['rearCamera'].astype(float)
df_hp['battery'] = df_hp['battery'].astype(float)
df_hp['price'] = df_hp['price'].astype(float)

#x digunakan untuk memisahkan atribut yang akan digunakan untuk prediksi
#y digunakan sebagai target yang akan diprediksi
x = df_hp[['storage','ram','screenSize','frontCamera','rearCamera','battery']]
y = df_hp['price']

sidebar_ = st.sidebar.selectbox("Page:", ["About", "Dataset", "Data Analys", "Prediksi"])

if sidebar_ == "About":
    st.write(' Machine Learning ini merupakan machine learning yang digunakan untuk prediksi harga HP, menggunakan model regresi linear dengan atribut Storage, RAM, SCreen SIze, Front Camera, Rear Camera, dan Battery, dan targetnya yaitu Price.')
    st.image('vivo2.jpeg')

elif sidebar_ == "Dataset":
    st.subheader("Dataset")
    df_hp

elif sidebar_ == "Data Analys":
    st.subheader("Pengecekkan Data Kosong")
    missing_values = df_hp.isnull().sum()
    st.write(missing_values)

    st.subheader("Grafik Storage")
    st.line_chart(df_hp['storage'])

    st.subheader("Grafik RAM")
    st.line_chart(df_hp['ram'])

    st.subheader("Grafik Screen Size")
    st.line_chart(df_hp['screenSize'])

    st.subheader("Grafik Front Camera")
    st.line_chart(df_hp['frontCamera'])

    st.subheader("Grafik Rear Camera")
    st.line_chart(df_hp['rearCamera'])

    st.subheader("Grafik Battery")
    st.line_chart(df_hp['battery'])

else:
    #membagi dataset menjadi dua bagian: set palatihan dan set pengujian
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #membuat objek model regresi lienar
    model_regresi = LinearRegression()

    #melatih model dengan menggunakan set pelatihan
    model_regresi.fit(X_train, y_train)

    #menyimpan prediksi harga
    model_regresi_pred = model_regresi.predict(X_test)

    Storage = st.number_input("Storage")
    RAM = st.number_input("RAM")
    ScreenSize = st.number_input("Screen Size")
    FrontCamera = st.number_input("Front Camera")
    RearCamera = st.number_input("Rear Camera")
    Battery = st.number_input("Battery")

    button = st.button('Predict')

    if button:
        #Menyimpan input dari pengguna
        X = np.array([[Storage, RAM, ScreenSize, FrontCamera, RearCamera, Battery]])

        #memprediksi harga berdasarkan input pengguna
        harga_X = model_regresi.predict(X)
        st.write("Harga HP dalam Rupiah", harga_X)

        # Menampilkan diagram perbandingan harga sebenarnya dan prediksi
        fig, ax = plt.subplots()
        ax.scatter(X_test.iloc[:, 0], y_test, label='Actual Prices', color='blue')
        ax.scatter(X_test.iloc[:, 0], model_regresi_pred, label='Predicted Prices', color='red')
        ax.set_xlabel('Storage')
        ax.set_ylabel('Price')
        ax.legend()

        # Menampilkan diagram di aplikasi Streamlit
        st.pyplot(fig)

        #evaluasi model MAE, MSE, dan RMSE
        #mengukur rata-rata dari selisih absolut antara nilai sebenarnya dan prediksi
        mae = mean_absolute_error(y_test, model_regresi_pred)
        # st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

        #mengukur rata-rata dari kuadrat selisih antara nilai sebenarnya dan prediksi
        mse = mean_squared_error(y_test, model_regresi_pred)
        # st.write(f'Mean Squared Error (MSE): {mse:.2f}')

        #menujukkan bahwa keseluruhan model memiliki kesalahan yang lebih kecil
        rmse = np.sqrt(mse)
        # st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')











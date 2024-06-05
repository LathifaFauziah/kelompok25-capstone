import streamlit as st
import pickle
from sklearn.neighbors import NearestNeighbors


# Set the background colors
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0; /* Light gray background */
        margin: 0; /* Remove default margin for body */
        padding: 0; /* Remove default padding for body */
    }
    .st-bw {
        background-color: #eeeeee; /* White background for widgets */
    }
    .st-cq {
        background-color: #cccccc; /* Gray background for chat input */
        border-radius: 10px; /* Add rounded corners */
        padding: 8px 12px; /* Add padding for input text */
        color: black; /* Set text color */
    }

    .st-cx {
        background-color: white; /* White background for chat messages */
    }
    .sidebar .block-container {
        background-color: #f0f0f0; /* Light gray background for sidebar */
        border-radius: 10px; /* Add rounded corners */
        padding: 10px; /* Add some padding for spacing */
    }
    .top-right-image-container {
        position: fixed;
        top: 30px;
        right: 0;
        padding: 20px;
        background-color: white; /* White background for image container */
        border-radius: 0 0 0 10px; /* Add rounded corners to bottom left */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <img src='https://cdn-icons-png.flaticon.com/512/3815/3815321.png' width='50'>
        <h1 style='margin: 0;'> Gramedia Digital Skilvul</h1>
    </div>
""", unsafe_allow_html=True)


# Create functions to open each social media app



##################################################################################################
# BUAT PEMANGGILAN PREDIKSI MULAI DARI SINI

# Cara memanggil file model yang sudah dibuat sebelumnya ke dalam code website
# '''
# 1. Masukan file .pkl ke dalam folder agar bisa diakses di website
# 2. Panggil alamat file (file path) beserta nama filenya seperti code dibawah (./random_forest_model.pkl)
# '''

# ===============================================================================
# Load Model (Cara Memanggil Model)

with open("./book_recommendation_knn_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)
# ===============================================================================

# '''
# 3.Code dibawah ini adalah cara melakukan prediksi, sama seperti jika anda memanggil prediksi di jupyter
# '''

# ====================================================================
# Menggunakan model yang dimuat yang sudah disimpan pada variabel 'loaded_model'
# Sesuaikan dengan model prediksi kalian masing-masing

# Kolom input teks untuk customer
# Buat inputan untuk empat angka berjenis float
number1 = st.number_input("Masukkan angka pertama:", format="%.2f")
number2 = st.number_input("Masukkan angka kedua:", format="%.2f")
number3 = st.number_input("Masukkan angka ketiga:", format="%.2f")
number4 = st.number_input("Masukkan angka keempat:", format="%.2f")

# Simpan angka-angka tersebut ke dalam list
numbers_list = [number1, number2, number3, number4]

if st.button("Prediksi"):
    if all(num == 0.0 for num in numbers_list):
        st.warning("Mohon isi nilai terlebih dahulu.")
    else:
        st.info("Sedang melakukan prediksi...")
        # contoh input: [3.07,3.04,3.39,3.55] atau [3.17,3.02,3.28,2.96]
        predictions = loaded_model.predict([numbers_list])

        # Lakukan evaluasi model atau tindakan lain yang diperlukan
        if predictions[0] == 0:
            pred = "Tepat Waktu"
        else:
            pred = "Tidak Tepat Waktu"
            
        st.write("Hasil Prediksi")
        st.title(pred)
        st.success("Prediksi selesai!")
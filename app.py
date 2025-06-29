import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Deteksi Intrusi Jaringan",
    page_icon="🛡️",
    layout="wide"
)

# Fungsi untuk memuat model dan preprocessor
# Menggunakan cache agar tidak perlu memuat ulang setiap kali ada interaksi
@st.cache_resource
def load_assets():
    """Memuat model dan preprocessor dari file."""
    try:
        # Nama file harus sesuai dengan yang Anda simpan dari notebook
        model = joblib.load('gnb_model.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        return model, preprocessor
    except FileNotFoundError:
        st.error("File model 'gnb_model.joblib' atau 'preprocessor.joblib' tidak ditemukan.")
        st.error("Pastikan Anda sudah menjalankan Langkah 1 dari petunjuk dan meletakkan file-file tersebut di folder yang sama dengan app.py.")
        return None, None

# Memuat aset (model dan preprocessor)
gnb_model, preprocessor = load_assets()

# --- Antarmuka Aplikasi ---
# Hanya tampilkan UI jika model berhasil dimuat
if gnb_model is not None and preprocessor is not None:
    st.title("🛡️ Aplikasi Deteksi Intrusi Jaringan")
    st.write("Aplikasi ini menggunakan model *Machine Learning* (Gaussian Naive Bayes) untuk memprediksi apakah sebuah koneksi jaringan bersifat **normal** atau **anomali**.")

    # Membuat dua tab: satu untuk unggah file, satu untuk input manual
    tab1, tab2 = st.tabs(["Unggah File CSV", "Prediksi Manual Satu Data"])

    # === Tab 1: Unggah File CSV ===
    with tab1:
        st.header("Prediksi dari File CSV")
        st.info("Unggah file CSV yang memiliki kolom sama dengan data latih (kecuali kolom 'class').")

        uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

        if uploaded_file is not None:
            try:
                # Baca file yang diunggah
                df_test = pd.read_csv(uploaded_file)
                st.write("**Pratinjau Data yang Diunggah:**")
                st.dataframe(df_test.head())

                # Simpan kolom asli sebelum diproses untuk ditampilkan nanti
                df_display = df_test.copy()

                # Tombol untuk menjalankan prediksi
                if st.button("Jalankan Prediksi pada File", type="primary"):
                    with st.spinner("Memproses dan melakukan prediksi..."):
                        # Proses data menggunakan preprocessor yang sudah dilatih
                        X_processed = preprocessor.transform(df_test)
                        
                        # Lakukan prediksi
                        predictions = gnb_model.predict(X_processed)
                        
                        # Tambahkan hasil prediksi ke DataFrame untuk ditampilkan
                        df_display['predicted_class'] = predictions

                        st.success("Prediksi selesai!")
                        st.write("**Hasil Prediksi:**")
                        st.dataframe(df_display)

                        # Tampilkan visualisasi hasil
                        st.write("**Visualisasi Hasil Prediksi:**")
                        fig, ax = plt.subplots()
                        sns.countplot(x='predicted_class', data=df_display, ax=ax, palette='coolwarm', order=['normal', 'anomaly'])
                        ax.set_title('Distribusi Kelas Hasil Prediksi')
                        ax.set_xlabel('Kelas Prediksi')
                        ax.set_ylabel('Jumlah')
                        st.pyplot(fig)

                        # Opsi untuk mengunduh hasil
                        @st.cache_data
                        def convert_df_to_csv(df):
                            return df.to_csv(index=False).encode('utf-8')

                        csv_results = convert_df_to_csv(df_display)
                        st.download_button(
                            label="Unduh Hasil Prediksi (CSV)",
                            data=csv_results,
                            file_name="hasil_prediksi_intrusi.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")
    
    # === Tab 2: Input Manual ===
    with tab2:
        st.header("Prediksi dari Input Manual")
        st.info("Masukkan nilai fitur untuk satu koneksi jaringan dan lihat hasilnya secara langsung.")

        # Membuat kolom untuk input agar lebih rapi
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Fitur Dasar")
            duration = st.number_input('Duration', min_value=0, value=0)
            protocol_type = st.selectbox('Protocol Type', ['tcp', 'udp', 'icmp'])
            service = st.text_input('Service', 'http') # Input teks karena banyak kemungkinan
            flag = st.selectbox('Flag', ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'S2', 'S3', 'OTH'])
            src_bytes = st.number_input('Source Bytes', min_value=0, value=250)
            dst_bytes = st.number_input('Destination Bytes', min_value=0, value=1500)
            
        with col2:
            st.subheader("Fitur Koneksi")
            count = st.number_input('Count', min_value=0, value=2)
            srv_count = st.number_input('Srv Count', min_value=0, value=2)
            serror_rate = st.slider('Serror Rate', 0.0, 1.0, 0.0)
            srv_serror_rate = st.slider('Srv Serror Rate', 0.0, 1.0, 0.0)
            rerror_rate = st.slider('Rerror Rate', 0.0, 1.0, 0.0)
            srv_rerror_rate = st.slider('Srv Rerror Rate', 0.0, 1.0, 0.0)
            
        with col3:
            st.subheader("Fitur Host Tujuan")
            dst_host_count = st.number_input('Dst Host Count', min_value=0, value=255)
            dst_host_srv_count = st.number_input('Dst Host Srv Count', min_value=0, value=255)
            dst_host_same_srv_rate = st.slider('Dst Host Same Srv Rate', 0.0, 1.0, 1.0)
            dst_host_diff_srv_rate = st.slider('Dst Host Diff Srv Rate', 0.0, 1.0, 0.0)
            dst_host_same_src_port_rate = st.slider('Dst Host Same Src Port Rate', 0.0, 1.0, 0.0)
            dst_host_serror_rate = st.slider('Dst Host Serror Rate', 0.0, 1.0, 0.0)

        # Tombol untuk prediksi manual
        if st.button("Prediksi Koneksi Ini", type="primary"):
            # Kumpulkan semua fitur lain yang tidak ada di UI dengan nilai default (misalnya 0)
            # Ini penting karena preprocessor mengharapkan 41 kolom sesuai data latih.
            features = {
                'duration': duration, 'protocol_type': protocol_type, 'service': service, 'flag': flag,
                'src_bytes': src_bytes, 'dst_bytes': dst_bytes, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
                'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0, 'root_shell': 0,
                'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
                'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'count': count, 'srv_count': srv_count,
                'serror_rate': serror_rate, 'srv_serror_rate': srv_serror_rate, 'rerror_rate': rerror_rate,
                'srv_rerror_rate': srv_rerror_rate, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0,
                'srv_diff_host_rate': 0.0, 'dst_host_count': dst_host_count, 'dst_host_srv_count': dst_host_srv_count,
                'dst_host_same_srv_rate': dst_host_same_srv_rate, 'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
                'dst_host_same_src_port_rate': dst_host_same_src_port_rate,
                'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': dst_host_serror_rate,
                'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
            }
            
            # Buat DataFrame dari input tunggal
            input_df = pd.DataFrame([features])
            
            # Proses dan prediksi
            input_processed = preprocessor.transform(input_df)
            prediction = gnb_model.predict(input_processed)
            prediction_proba = gnb_model.predict_proba(input_processed)

            # Tampilkan hasil
            st.subheader("Hasil Prediksi:")
            if prediction[0] == 'anomaly':
                st.error(f"**Anomali Terdeteksi!** (Probabilitas: {prediction_proba[0][0]:.2f})", icon="🚨")
            else:
                st.success(f"**Koneksi Normal** (Probabilitas: {prediction_proba[0][1]:.2f})", icon="✅")


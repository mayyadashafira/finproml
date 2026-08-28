import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Fungsi preprocessing arsitektur yang dipakai di dalam model (layer Lambda saat training).
# Model final = EfficientNetB0, jadi kita pakai efficientnet.preprocess_input.
# Kalau kamu ganti model final ke arsitektur lain, ganti import & custom_objects di bawah:
#   MobileNetV2 -> tf.keras.applications.mobilenet_v2.preprocess_input
#   ResNet50    -> tf.keras.applications.resnet50.preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as architecture_preprocess_input

# ============ KONFIGURASI ============
st.set_page_config(page_title="Klasifikasi Sampah", page_icon="🗑️", layout="centered")

MODEL_PATH = "best_garbage_model.keras"
IMG_SIZE = (224, 224)

# ⚠️ Urutan kelas HARUS SAMA PERSIS dengan urutan class_indices saat training
# (cek output "train_generator.class_indices" di notebook modeling kamu untuk memastikan)
CLASS_LABELS = [
    "Botol_Plastik",
    "Kaca",
    "Karton_Minuman",
    "Kemasan_Logam",
    "Kertas_Karton",
    "Sampah_Rumah_Tangga",
]

CLASS_INFO = {
    "Botol_Plastik": ("♻️", "Buang di tempat sampah plastik/anorganik untuk didaur ulang."),
    "Kaca": ("🍾", "Daur ulang khusus kaca — hati-hati kemungkinan pecahan tajam."),
    "Karton_Minuman": ("🧃", "Bilas dahulu sebelum dibuang, masuk kategori karton/tetra pak."),
    "Kemasan_Logam": ("🥫", "Daur ulang di tempat sampah logam/kaleng."),
    "Kertas_Karton": ("📦", "Daur ulang di tempat sampah kertas."),
    "Sampah_Rumah_Tangga": ("🗑️", "Umumnya masuk kategori sampah organik/residu."),
}


# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"File model '{MODEL_PATH}' tidak ditemukan di folder aplikasi. "
            "Pastikan file model sudah ditambahkan ke repository (lihat README)."
        )
        st.stop()
    # safe_mode=False diperlukan karena model ini punya layer Lambda (untuk
    # preprocessing khusus arsitektur MobileNetV2/ResNet50/EfficientNetB0).
    # custom_objects diperlukan agar Keras tahu cara merekonstruksi fungsi
    # yang dibungkus Lambda tersebut saat model di-load kembali.
    # Aman digunakan karena model ini kita buat & latih sendiri, bukan dari sumber
    # yang tidak dipercaya.
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": architecture_preprocess_input},
        safe_mode=False,
    )


model = load_model()

# ============ UI ============
st.title("🗑️ Klasifikasi Sampah Otomatis")
st.write(
    "Upload foto sampah, dan model Machine Learning akan memprediksi kategorinya "
    "dari 6 kelas: Botol Plastik, Kaca, Karton Minuman, Kemasan Logam, Kertas Karton, "
    "dan Sampah Rumah Tangga."
)

uploaded_file = st.file_uploader("Pilih gambar sampah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Gambar yang diupload", width="stretch")

    # Preprocessing: HARUS SAMA dengan yang dipakai saat training
    # (generator training memakai rescale=1./255, model sudah punya layer
    # preprocessing arsitektur di dalamnya)
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Menganalisis gambar..."):
        pred_probs = model.predict(img_array, verbose=0)[0]

    pred_idx = int(np.argmax(pred_probs))
    pred_label = CLASS_LABELS[pred_idx]
    confidence = float(pred_probs[pred_idx]) * 100
    emoji, tip = CLASS_INFO.get(pred_label, ("🗑️", ""))

    with col2:
        st.subheader(f"{emoji} {pred_label.replace('_', ' ')}")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.caption(tip)

        if confidence < 60:
            st.warning(
                "⚠️ Model kurang yakin dengan prediksi ini. "
                "Coba foto dengan pencahayaan lebih baik atau objek lebih jelas."
            )

    st.divider()
    st.subheader("Detail Prediksi — Semua Kelas")
    sorted_idx = np.argsort(pred_probs)[::-1]
    for idx in sorted_idx:
        label = CLASS_LABELS[idx]
        prob = float(pred_probs[idx]) * 100
        st.write(f"**{label.replace('_', ' ')}** — {prob:.2f}%")
        st.progress(min(max(prob / 100, 0.0), 1.0))

else:
    st.info("👆 Silakan upload gambar untuk memulai prediksi.")

st.divider()
st.caption("Final Project Machine Learning — Klasifikasi Sampah (Computer Vision)")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow import keras

st.set_page_config(page_title="Roubo ou Furto? - SP", page_icon="🔎")

@st.cache_resource
def carrega():
    return keras.models.load_model("modelo_mlp.keras"), joblib.load("encoder.joblib")

modelo, enc = carrega()
cat_cols = ["BAIRRO", "DESCR_TIPOLOCAL", "periodo_dia"]
cats = dict(zip(cat_cols, enc.categories_))

DIAS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

st.title("🔎 Classificador de ocorrências: roubo ou furto")
st.write("MLP treinada com ocorrências de **celulares subtraídos na capital de São Paulo (2023, SSP-SP)**. "
         "Informe as características de uma ocorrência e veja a classe mais compatível.")

col1, col2 = st.columns(2)
bairro = col1.selectbox("Bairro", sorted(cats["BAIRRO"]))
local = col2.selectbox("Tipo de local", sorted(cats["DESCR_TIPOLOCAL"]))
periodo = col1.selectbox("Período do dia", list(cats["periodo_dia"]))
dia = col2.selectbox("Dia da semana", DIAS)

if st.button("Classificar", type="primary"):
    d = DIAS.index(dia)
    cat_df = pd.DataFrame([[bairro, local, periodo]], columns=cat_cols)
    x = np.hstack([
        enc.transform(cat_df),
        [[np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)]],
    ]).astype("float32")

    p_roubo = float(modelo.predict(x, verbose=0)[0][0])
    p_furto = 1 - p_roubo

    st.subheader("Classe estimada: " + ("ROUBO" if p_roubo >= 0.5 else "FURTO"))
    c1, c2 = st.columns(2)
    c1.metric("Probabilidade de roubo", f"{p_roubo:.1%}")
    c2.metric("Probabilidade de furto", f"{p_furto:.1%}")
    st.progress(p_roubo)

    st.caption("Interpretação: considerando que a ocorrência aconteceu com essas características, "
               "o modelo estima a probabilidade de ela ser roubo ou furto. "
               "Ele **não** calcula o risco de um crime acontecer nesse local ou horário.")
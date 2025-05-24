import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="All Care Vet - Análise Preditiva", layout="wide")

# Inicializa o vetor
vectorizer = TfidfVectorizer()

# ✅ Função atualizada com remoção de NaN e campos vazios
def treinar_modelo():
    if os.path.exists('data/novos_casos.csv'):
        df = pd.read_csv('data/novos_casos.csv', encoding='utf-8', names=['anamnese', 'decisao', 'dias_prev', 'maior_risco'])

        # ✅ Remover registros inválidos em anamnese e decisao
        df = df.dropna(subset=['anamnese', 'decisao'])
        df = df[(df['anamnese'].astype(str).str.strip() != '') & (df['decisao'].astype(str).str.strip() != '')]

        if len(df) >= 3:
            X = vectorizer.fit_transform(df['anamnese'].astype(str))
            y = df['decisao'].astype(str)

            modelo = LogisticRegression(max_iter=1000)
            modelo.fit(X, y)
            joblib.dump(modelo, 'modelo_classificador.pkl')

# Carregar banco de sintomas e riscos
sintomas_risco = pd.read_csv('data/sintomas_risco.csv', encoding='utf-8')
sintomas_dict = dict(zip(sintomas_risco['Sintoma'].str.lower(), sintomas_risco['Risco']))

# Função de cálculo automático
def calcular_decisao_e_dias(maior_risco):
    if maior_risco >= 80:
        dias = 10
    elif maior_risco >= 60:
        dias = 7
    elif maior_risco >= 40:
        dias = 5
    elif maior_risco >= 20:
        dias = 3
    else:
        dias = 1
    decisao = "Internar" if maior_risco >= 50 else "Medicar e enviar para casa"
    return decisao, dias

# Controle de estado
if 'analise_feita' not in st.session_state:
    st.session_state.analise_feita = False
    st.session_state.resultado = ""
    st.session_state.riscos_identificados = []
    st.session_state.anamnese = ""

# Layout com duas colunas
col1, col2 = st.columns([2, 1])

with col2:
    st.image('assets/logo.png', width=300)

with col1:
    st.title("All Care Vet - Análise Preditiva")

    if not st.session_state.analise_feita:
        with st.form("form_anamnese"):
            anamnese = st.text_area("Anamnese Completa", height=250, key="input_anamnese")
            submitted = st.form_submit_button("Analisar")

            if submitted:
                if not anamnese.strip():
                    st.error("Por favor, insira a anamnese completa.")
                else:
                    texto = anamnese.strip()
                    texto_lower = texto.lower()
                    riscos_identificados = [(sintoma, risco) for sintoma, risco in sintomas_dict.items() if sintoma in texto_lower]

                    maior_risco = max([r[1] for r in riscos_identificados], default=0)
                    decisao, dias_prev = calcular_decisao_e_dias(maior_risco)

                    resultado = f"**Decisão:** {decisao}\n\n"
                    resultado += f"**Dias previstos:** {dias_prev}\n\n"
                    resultado += f"**Chance de Eutanásia (base clínica):** {maior_risco:.1f}%\n\n"

                    if maior_risco >= 70:
                        comentario = "🔴 **Alta probabilidade de morte ou eutanásia após internação.**"
                    elif maior_risco >= 30:
                        comentario = "🟠 **Probabilidade moderada de morte após internação.**"
                    else:
                        comentario = "🟢 **Alta chance de recuperação após internação.**"

                    resultado += comentario

                    # Armazenar no estado
                    st.session_state.analise_feita = True
                    st.session_state.resultado = resultado
                    st.session_state.riscos_identificados = riscos_identificados
                    st.session_state.anamnese = anamnese

                    # Salvar no banco de dados
                    novos_dados = [texto, decisao, dias_prev, maior_risco]
                    os.makedirs('data', exist_ok=True)
                    with open('data/novos_casos.csv', mode='a', newline='', encoding='utf-8') as f:
                        f.write(",".join(map(str, novos_dados)) + "\n")

                    # ✅ Auto-retreinamento com segurança
                    treinar_modelo()

                    st.rerun()

    else:
        st.subheader("Resultado da Análise")
        st.markdown(st.session_state.resultado)

        st.subheader("Probabilidades baseadas nos sintomas identificados")
        if st.session_state.riscos_identificados:
            for sintoma, risco in st.session_state.riscos_identificados:
                st.write(f"- **{sintoma.capitalize()}**: {risco}%")
        else:
            st.write("Nenhum sintoma crítico identificado.")

        if st.button("Nova Análise"):
            st.session_state.analise_feita = False
            st.session_state.resultado = ""
            st.session_state.riscos_identificados = []
            st.session_state.anamnese = ""
            st.rerun()

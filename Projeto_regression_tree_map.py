import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1️⃣ Base de dados
dados = pd.DataFrame({
    'tipo': ['camiseta', 'jaqueta', 'tênis', 'bermuda', 'chinelo', 'cachecol'],
    'preco': [100, 200, 50, 50, 15, 10],
    'clima': ['quente', 'frio', 'quente', 'quente', 'quente', 'frio'],
    'quantidade': [4, 10, 6, 8, 9, 20],
    'tamanho': ['P', 'M', 'M', 'G', 'G', 'P']
})

dias = 30  # Previsão futura em dias
estoque_critico = 5  # Limite de alerta

st.title("🛒 Painel Interativo de Vendas")

# 2️⃣ Inputs
tipo = st.selectbox("Tipo do produto", dados['tipo'].unique())
preco = st.slider("Preço (R$)", 5, 300, 50, 5)
clima = st.radio("Clima", ['quente', 'frio'])
tamanho = st.selectbox("Tamanho", dados['tamanho'].unique())
modelo_selecionado = st.selectbox("Modelo de previsão", ['Linear Regression', 'Decision Tree'])

if st.button("Prever"):

    entrada = pd.DataFrame({
        'tipo': [tipo],
        'preco': [preco],
        'clima': [clima],
        'tamanho': [tamanho]
    })

    colunas_categoricas = ['tipo', 'clima', 'tamanho']
    colunas_numericas = ['preco']
    transformador = ColumnTransformer([
        ('cat', OneHotEncoder(), colunas_categoricas),
        ('num', 'passthrough', colunas_numericas)
    ])

    regressor = LinearRegression() if modelo_selecionado == 'Linear Regression' else DecisionTreeRegressor(random_state=42)
    modelo = Pipeline([('transformacao', transformador), ('regressor', regressor)])

    X = dados[['tipo', 'preco', 'clima', 'tamanho']]
    y = dados['quantidade']
    modelo.fit(X, y)

    # Previsão
    previsao = modelo.predict(entrada)[0]
    media_hist = dados[dados['tipo'] == tipo]['quantidade'].mean()
    percentual = (previsao - media_hist) / media_hist * 100

    st.subheader("📊 Relatório de Previsão")
    st.write(f"**Produto:** {tipo}")
    st.write(f"**Preço atual:** R$ {preco}")
    st.write(f"**Clima:** {clima}")
    st.write(f"**Tamanho:** {tamanho}")
    st.write(f"**Quantidade Prevista:** {previsao:.1f} unidades")
    st.write(f"**Média histórica:** {media_hist:.1f} unidades")
    st.write(f"**Variação esperada:** {percentual:.1f}%")

    if previsao < estoque_critico:
        st.warning(f"⚠️ Estoque crítico! Apenas {previsao:.1f} unidades previstas.")

    # Gráfico preço vs previsão
    precos = np.linspace(5, 300, 50)
    previsoes_precos = [modelo.predict(pd.DataFrame({'tipo':[tipo],'preco':[p],
                                                    'clima':[clima],'tamanho':[tamanho]}))[0]
                        for p in precos]

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(precos, previsoes_precos, color='blue', linewidth=2)
    ax.scatter(preco, previsao, color='red', s=100, label='Sua escolha')
    ax.set_xlabel('Preço (R$)')
    ax.set_ylabel('Quantidade Prevista')
    ax.set_title(f'Impacto do Preço nas Vendas - {tipo.capitalize()}')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Preço ideal
    preco_ideal = precos[np.argmax(previsoes_precos)]
    st.write(f"💰 Preço sugerido para maximizar vendas: R$ {preco_ideal:.1f}")

    # Ranking produtos
    ranking = []
    for t in dados['tipo'].unique():
        df_tmp = pd.DataFrame({'tipo':[t],'preco':[50],'clima':[clima],'tamanho':['M']})
        qtd = modelo.predict(df_tmp)[0]
        ranking.append((t, qtd))
    ranking.sort(key=lambda x: x[1], reverse=True)

    st.subheader("🏆 Ranking de produtos para o clima escolhido")
    for idx, (prod, qtd) in enumerate(ranking,1):
        st.write(f"{idx}. {prod}: {qtd:.1f} unidades")

    # Previsão futura 30 dias
    previsao_futura = pd.DataFrame()
    for t in dados['tipo'].unique():
        df_tmp = pd.DataFrame({'tipo':[t]*dias,
                               'preco':[50]*dias,
                               'clima':[clima]*dias,
                               'tamanho':['M']*dias})
        df_tmp['dia'] = range(1,dias+1)
        df_tmp['previsao'] = modelo.predict(df_tmp[['tipo','preco','clima','tamanho']])
        previsao_futura = pd.concat([previsao_futura, df_tmp])

    fig2, ax2 = plt.subplots(figsize=(8,5))
    for t in dados['tipo'].unique():
        df_plot = previsao_futura[previsao_futura['tipo']==t]
        ax2.plot(df_plot['dia'], df_plot['previsao'], label=t, linewidth=2)
    ax2.set_xlabel("Dia")
    ax2.set_ylabel("Quantidade Prevista")
    ax2.set_title(f"📊 Previsão futura de vendas para os próximos {dias} dias")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

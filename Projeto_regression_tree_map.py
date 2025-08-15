import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ===========================
# Base de dados ampliada
# ===========================
np.random.seed(42)

tipos = ['camiseta', 'jaqueta', 'tênis', 'bermuda', 'chinelo', 'cachecol',
         'moletom', 'boné', 'saia', 'blusa', 'calça', 'sandália', 'meia', 'luva', 'gorra']
climas = ['quente', 'frio']
tamanhos = ['P', 'M', 'G']

num_produtos = 70
dados = pd.DataFrame({
    'tipo': np.random.choice(tipos, num_produtos),
    'preco': np.random.randint(10, 301, num_produtos),
    'clima': np.random.choice(climas, num_produtos),
    'quantidade': np.random.randint(5, 26, num_produtos),
    'tamanho': np.random.choice(tamanhos, num_produtos)
})

dias = 30
estoque_critico = 5

# ===========================
# Interface Streamlit
# ===========================
st.title("🛒 Painel de Previsão de Vendas")

tipo = st.selectbox("Tipo de produto", dados['tipo'].unique())
preco = st.slider("Preço", 10, 300, 50, step=5)
clima = st.radio("Clima", ['quente', 'frio'])
tamanho = st.selectbox("Tamanho", dados['tamanho'].unique())
modelo_selecionado = st.selectbox("Modelo", ['Linear Regression', 'Decision Tree'])

if st.button("Prever"):
    # Entrada do usuário
    entrada = pd.DataFrame({'tipo':[tipo],'preco':[preco],'clima':[clima],'tamanho':[tamanho]})
    
    # Transformador
    colunas_categoricas = ['tipo', 'clima', 'tamanho']
    colunas_numericas = ['preco']
    transformador = ColumnTransformer([
        ('cat', OneHotEncoder(), colunas_categoricas),
        ('num', 'passthrough', colunas_numericas)
    ])
    
    # Modelo
    regressor = LinearRegression() if modelo_selecionado == 'Linear Regression' else DecisionTreeRegressor(random_state=42)
    modelo = Pipeline([('transformacao', transformador), ('regressor', regressor)])
    
    X = dados[['tipo', 'preco', 'clima', 'tamanho']]
    y = dados['quantidade']
    modelo.fit(X, y)
    
    # Previsão
    previsao = modelo.predict(entrada)[0]
    st.subheader(f"Quantidade prevista: {previsao:.1f} unidades")
    
    # Gráfico preço vs previsão
    precos = np.linspace(10, 300, 50)
    previsoes_precos = [modelo.predict(pd.DataFrame({'tipo':[tipo],'preco':[p],
                                                    'clima':[clima],'tamanho':[tamanho]}))[0]
                        for p in precos]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(precos, previsoes_precos, color='blue', linewidth=2)
    ax.scatter(preco, previsao, color='red', s=100, label='Sua escolha')
    ax.set_xlabel("Preço (R$)")
    ax.set_ylabel("Quantidade Prevista")
    ax.set_title(f"Impacto do Preço nas Vendas - {tipo.capitalize()}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    # Ranking de produtos
    ranking = []
    for t in dados['tipo'].unique():
        df_tmp = pd.DataFrame({'tipo':[t],'preco':[50],'clima':[clima],'tamanho':['M']})
        qtd = modelo.predict(df_tmp)[0]
        ranking.append((t, qtd))
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    st.subheader("🏆 Ranking de produtos")
    for idx, (prod, qtd) in enumerate(ranking,1):
        st.write(f"{idx}. {prod}: {qtd:.1f} unidades")
    
    # Previsão futura em gráfico de treemap-like
    import squarify  # pip install squarify
    previsao_futura = []
    for t in dados['tipo'].unique():
        df_tmp = pd.DataFrame({'tipo':[t]*dias,
                               'preco':[50]*dias,
                               'clima':[clima]*dias,
                               'tamanho':['M']*dias})
        df_tmp['previsao'] = modelo.predict(df_tmp[['tipo','preco','clima','tamanho']])
        previsao_futura.append(df_tmp.groupby('tipo')['previsao'].sum())
    previsao_futura = pd.concat(previsao_futura)
    
    st.subheader("📊 Previsão futura total por produto (Treemap)")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    squarify.plot(sizes=previsao_futura.values, label=previsao_futura.index, alpha=0.8)
    plt.axis('off')
    st.pyplot(fig2)

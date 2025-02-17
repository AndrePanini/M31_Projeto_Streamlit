import streamlit as st
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gower import gower_matrix
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Configuração inicial da página
st.set_page_config(
    page_title="EBAC | Módulo 31 | Projeto de Agrupamento hierárquico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Função para calcular a matriz de Gower com otimização
@st.cache_data(show_spinner=False)
def calcularGowerMatrix(data_x, cat_features):
    try:
        st.write("Calculando matriz de Gower... (Isso pode levar alguns minutos)")
        # Limita o tamanho dos dados para evitar problemas de memória
        if len(data_x) > 5000:
            st.warning("O conjunto de dados é grande. Usando uma amostra de 5000 linhas para o cálculo.")
            data_x = data_x.sample(n=5000, random_state=42)
        
        # Calcula a matriz de Gower
        dist_gower = gower_matrix(data_x=data_x, cat_features=cat_features)
        st.success("Matriz de Gower calculada com sucesso!")
        return dist_gower
    except Exception as e:
        st.error(f"Erro ao calcular a matriz de Gower: {e}")
        return None

# Função para criar o dendrograma
def dn(color_threshold: float, num_groups: int, Z: list) -> None:
    plt.figure(figsize=(24, 6))
    plt.ylabel(ylabel='Distância')
    plt.title(f'Dendrograma Hierárquico - {num_groups} Grupos')
    dn = dendrogram(
        Z=Z,
        p=6,
        truncate_mode='level',
        color_threshold=color_threshold,
        show_leaf_counts=True,
        leaf_font_size=8,
        leaf_rotation=45,
        show_contracted=True
    )
    plt.yticks(np.linspace(0, 0.6, num=31))
    plt.xticks([])
    st.pyplot(plt)
    for i in dn.keys():
        st.text(f'dendrogram.{i}: {len(dn[i])}')

# Função principal
def main():
    # Carregar dados
    df = pd.read_csv('https://raw.githubusercontent.com/AndrePanini/M31_Projeto_Streamlit/main/online_shoppers_intention.csv')

    # Sidebar com informações
    with st.sidebar.expander(label="Índice", expanded=False):
        st.markdown('''
            - [Entendimento do negócio](#intro)
            - [Visualização dos dados](#visualizacao)
            - [Análise descritiva](#descritiva)
            - [Feature selection](#feature_selection)
            - [Agrupamentos hierárquicos](#agrupamento)
            - [Construção, avaliação e análise dos grupos](#grupos)
            - [Conclusão](#final)
        ''', unsafe_allow_html=True)

    # Exibir dados
    st.markdown('''## Visualização dos Dados''', unsafe_allow_html=True)
    st.dataframe(df)

    # Análise descritiva
    st.markdown('''## Análise Descritiva''', unsafe_allow_html=True)
    st.info(f'''
        Quantidade de linhas: {df.shape[0]}
        Quantidade de colunas: {df.shape[1]}
        Quantidade de valores missing: {df.isna().sum().sum()}
    ''')
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # Feature selection
    st.markdown('''## Feature Selection''', unsafe_allow_html=True)
    numerical = ['ProductRelated', 'PageValues', 'SpecialDay']
    df_ = df[['Administrative', 'Informational', 'ProductRelated', 'PageValues', 'OperatingSystems', 'Browser', 'TrafficType', 'VisitorType', 'SpecialDay', 'Month', 'Weekend']]
    df_cat = df_.drop(columns=numerical)

    # Processamento de variáveis dummy
    df_dummies = pd.get_dummies(data=df_, drop_first=False)
    categorical_features = df_dummies.drop(columns=numerical).columns.values
    cat_features = [True if column in categorical_features else False for column in df_dummies]

    # Cálculo da matriz de Gower
    st.markdown('''## Cálculo da Matriz de Distância Gower''', unsafe_allow_html=True)
    dist_gower = calcularGowerMatrix(data_x=df_dummies, cat_features=cat_features)

    if dist_gower is not None:
        # Cálculo da matriz de ligação
        gdv = squareform(X=dist_gower, force='tovector')
        Z = linkage(y=gdv, method='complete')

        # Dendrogramas
        st.markdown('''## Dendrogramas''', unsafe_allow_html=True)
        for qtd, color_threshold in [(3, 0.53), (4, 0.5)]:
            st.info(f'{qtd} grupos:')
            dn(color_threshold=color_threshold, num_groups=qtd, Z=Z)

        # Agrupamento e análise
        st.markdown('''## Construção, Avaliação e Análise dos Grupos''', unsafe_allow_html=True)
        df['grupo_3'] = fcluster(Z=Z, t=3, criterion='maxclust')
        df['grupo_4'] = fcluster(Z=Z, t=4, criterion='maxclust')

        st.dataframe(pd.DataFrame({
            'Grupo': df.grupo_3.value_counts().index,
            'Quantidade': df.grupo_3.value_counts().values
        }).set_index('Grupo').style.format({'Quantidade': lambda x: '{:d}'.format(x)}))

        st.table(pd.crosstab(index=df.VisitorType, columns=[df.grupo_3, df.Revenue], normalize='index').applymap(lambda x: f'{x*100:.0f} %'))

    else:
        st.error("Não foi possível calcular a matriz de Gower. Verifique os logs para mais detalhes.")

if __name__ == '__main__':
    main()

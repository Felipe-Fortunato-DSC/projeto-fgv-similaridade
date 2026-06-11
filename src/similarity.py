import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from .data_process import padronizar_medida, preprocess_text, remove_stopwords, converter_medida
from .penalty import calcular_penalizacao

# Download do modelo de transformers SBERT pré-treinado com compatibilidade para a lingua PT-BR
sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def gerar_embeddings(df, colunas):
    df['bert_vectors'] = [sbert_model.encode(row, convert_to_numpy=True) 
                          for row in tqdm(df[colunas].astype(str).agg(' '.join, axis=1), desc="Processando embeddings")]
    return df
    
def treinar_knn(df, metric, n_neighbors=5):
    # Treina um modelo k-NN para busca de similaridade
    X_train = np.array(df['bert_vectors'].tolist())
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric=metric)
    knn_model.fit(X_train)
    return knn_model

def consultar_item(descricao, medida, knn_model, df, df_embeddings, marca='SEM MARCA'):
    # Consulta um insumo no modelo de similaridade baseado em SBERT e k-NN.

    consulta_original = f"{descricao}"
    print(f'Item consultado: {consulta_original}')
    
    descricao_processada = remove_stopwords(preprocess_text(descricao))
    marca_processada = remove_stopwords(preprocess_text(marca))
    medida_processada = padronizar_medida(converter_medida(medida))
    medida_processada = medida_processada.replace('.0', '')
    medida_processada = remove_stopwords(preprocess_text(medida_processada))

    consulta = f"{descricao_processada} {marca_processada} {medida_processada}"
    print(f'\nItem consultado: {consulta}\n')

    consulta_vector = sbert_model.encode(consulta, convert_to_numpy=True).reshape(1, -1)
    distances, indices = knn_model.kneighbors(consulta_vector)

    resultados = df.iloc[indices.flatten()]
    resultados['CONSULTA'] = consulta_original
    resultados["SCORE"] = (1 - distances.flatten()) * 100
    resultados["SCORE"] = resultados["SCORE"].apply(lambda x: f"{x:.6f}")

    resultados = resultados.loc[:, ~resultados.columns.str.contains('^Unnamed')]

    # Armazenar os valores originais de 'MEDIDA' e 'MARCA'
    medidas_originais = resultados['MEDIDA'].copy()
    marcas_originais = resultados['MARCA'].copy()
    descricao_original = resultados['INSUMO_DESCRICAO'].copy()

    # Aplicar as funções de padronização temporariamente para o cálculo
    resultados['MEDIDA'] = resultados['MEDIDA'].apply(lambda x: padronizar_medida(converter_medida(x)))
    resultados['MEDIDA'] = resultados['MEDIDA'].apply(lambda x: str(x).replace('.0', ''))
    resultados['MEDIDA'] = resultados['MEDIDA'].apply(lambda x: remove_stopwords(preprocess_text(x)))
    resultados['MARCA'] = resultados['MARCA'].apply(lambda x: remove_stopwords(preprocess_text(x)))
    resultados['INSUMO_DESCRICAO'] = resultados['INSUMO_DESCRICAO'].apply(lambda x: remove_stopwords(preprocess_text(x)))
    
     # Calculando penalização para cada resultado baseado nas distâncias de Levenshtein
    for i, row in resultados.iterrows():
        # Calcular a penalização com base na marca e medida
        penalizacao = calcular_penalizacao(marca_processada, row['MARCA'], medida_processada, row['MEDIDA'], descricao_processada, row['INSUMO_DESCRICAO'])
        
        # Ajustar o score pela penalização
        score_original = float(row["SCORE"])
        novo_score = score_original * (1 - penalizacao)  # Penaliza o score com base na distância

        # Atualizar o score com o novo valor
        resultados.at[i, "SCORE"] = f"{novo_score:.6f}"

        # Restaurar os valores originais de 'MEDIDA' e 'MARCA'
        resultados['MEDIDA'] = medidas_originais
        resultados['MARCA'] = marcas_originais
        resultados['INSUMO_DESCRICAO'] = descricao_original

        # Convertendo 'SCORE' para float para comparar corretamente com 50.0
        resultados['SCORE'] = resultados['SCORE'].astype(float)
        resultados = resultados.sort_values(by='SCORE', ascending=False)
        # Filtrando resultados com similaridade acima de 50%
        resultados = resultados[resultados['SCORE'] > 50.0]
        
    return resultados.head(knn_model.n_neighbors)
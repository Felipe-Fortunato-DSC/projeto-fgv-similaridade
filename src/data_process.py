import pandas as pd
import unicodedata
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')

def padronizar_medida(medida):
    # Padroniza unidades de medida para valores numéricos homogêneos, mas mantém as unidades não padronizáveis inalteradas.

    # Dicionários de conversão de unidades
    conversao_peso = {"KILOGRAMA": 1000, "GRAMA": 1, "MILIGRAMA": 0.001, "TONELADA": 1000000}
    conversao_volume = {"LITRO": 1000, "MILILITRO": 1, "CENTIMETRO CUBICO": 1, "METRO CUBICO": 1000000}
    conversao_comprimento = {"METRO": 1000, "CENTIMETRO": 10, "MILIMETRO": 1, "KILOMETRO": 1000000, "POLEGADA": 25.4, "PÉS": 304.8, "JARDAS": 914.4}

    medida = medida.upper().strip()
    
    # Expressão regular para capturar a medida e unidade
    match = re.search(r'(\d+[\.,]?\d*)\s*(KILOGRAMA|GRAMA|MILIGRAMA|TONELADA|LITRO|MILILITRO|CENTIMETRO CUBICO|METRO CUBICO|METRO|CENTIMETRO|MILIMETRO|KILOMETRO|POL|PÉS|JARDAS)', medida)

    if match:
        # Substituir vírgula por ponto, se necessário
        valor, unidade = match.groups()
        valor = float(valor.replace(",", "."))
        
        # Aplicando as conversões conforme a unidade
        if unidade in conversao_peso:
            return f"{valor * conversao_peso[unidade]}GRAMA"
        elif unidade in conversao_volume:
            return f"{valor * conversao_volume[unidade]}MILILITRO"  # Padronizado para mL
        elif unidade in conversao_comprimento:
            return f"{valor * conversao_comprimento[unidade]}MILIMETRO"
    
    # Se não for uma unidade convertível, retorna a medida original
    return medida

def preprocess_text(text):
    # Normaliza o texto removendo acentos, caracteres especiais e convertendo para minúsculas.
    if isinstance(text, str):
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    elif pd.isna(text):
        text = ''
    return text

def remove_stopwords(text):
    # Remove stopwords do texto.
    words = word_tokenize(text)
    stop_words = set(stopwords.words('portuguese')) - {'com', 'sem'}
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remover_palavras_duplicadas(df, colunas):
    # Remove palavras duplicadas de forma nas colunas especificadas de um DataFrame
    for coluna in colunas:
        # Remover palavras duplicadas de cada linha da coluna
        df[coluna] = df[coluna].apply(lambda x: ' '.join(sorted(set(x.split()), key=x.split().index)))
    return df

# Função para converter a sigla para a medida por extenso
def converter_medida(input_value):
    # Carregando df com medidas padronizadas
    from .config import MEDIDA_CORRELACAO_CSV
    df_medida = pd.read_csv(MEDIDA_CORRELACAO_CSV)
    # Remove todos os espaços
    input_value = input_value.replace(" ", "")
    # Usando regex para separar o número da sigla
    match = re.match(r"(\d+)([a-zA-Z]+)", input_value)

    sigla = ''
    if match:
        valor = match.group(1)  # Parte numérica
        sigla = match.group(2)  # Parte da sigla
    else:
        # Caso não encontre número, trata-se apenas de uma sigla
        valor = ''
        sigla = input_value  # Caso só tenha a sigla

    sigla = preprocess_text(sigla)
    df_medida['CD_MEDIDA'] = df_medida['CD_MEDIDA'].apply(preprocess_text)

    # Verifica se a sigla existe no DataFrame
    medida_por_extenso = df_medida[df_medida['CD_MEDIDA'] == sigla]['MEDIDA'].values
    if len(medida_por_extenso) > 0:
        return f"{valor}{medida_por_extenso[0]}"
    else:
        return input_value  # Retorna o próprio input caso não encontre a sigla
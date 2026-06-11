import Levenshtein

# Função para calcular a penalização com base na distância de Levenshtein
def calcular_penalizacao(consulta_marca, input_marca, consulta_medida, input_medida, consulta_descricao, input_descricao):
    # Calculando a distância de Levenshtein    
    distancia_marca = Levenshtein.distance(consulta_marca, input_marca)
    distancia_medida = Levenshtein.distance(consulta_medida, input_medida)
    distancia_descricao = Levenshtein.distance(consulta_descricao, input_descricao)
    
    # Calculando a similaridade, que é inversamente proporcional à distância
    similaridade_marca = 1 - (distancia_marca / max(len(consulta_marca), len(input_marca)))

    # Calculando a similaridade, que é inversam}'ente proporcional à distância
    similaridade_medida = 1 - (distancia_medida / max(len(consulta_medida), len(input_medida)))

    # Calculando a similaridade, que é inversamente proporcional à distância
    similaridade_descricao = 1 - (distancia_descricao / max(len(consulta_descricao), len(input_descricao)))
    
    # Penalização ajustada para o intervalo
    if consulta_marca == '' and consulta_medida == '':
        # Caso tanto a marca quanto a medida estejam vazias, aplicar penalidade maior na descrição
        penalizacao_marca = 0
        penalizacao_medida = 0
        penalizacao_descricao = (1 - similaridade_descricao) * 0.5
    elif consulta_marca == '':
        # Caso a marca esteja vazia, aplicar penalidade maior na descrição e na medida
        penalizacao_marca = 0
        penalizacao_medida = (1 - similaridade_medida) * 0.8
        penalizacao_descricao = (1 - similaridade_descricao) * 0.1
    elif consulta_medida == '':
        # Caso a medida esteja vazia, aplicar penalidade maior na marca e na descrição
        penalizacao_marca = (1 - similaridade_marca) * 0.8
        penalizacao_medida = 0
        penalizacao_descricao = (1 - similaridade_descricao) * 0.1
    else:
        # Caso nenhuma das condições anteriores se aplique, aplica penalidades padrão
        penalizacao_marca = (1 - similaridade_marca) * 0.4  # Penalidade intermediária para a marca
        penalizacao_medida = (1 - similaridade_medida) * 0.8  # Penalidade mais alta para medida
        penalizacao_descricao = (1 - similaridade_descricao) * 0.1  # Penalidade menor para a descrição

    penalizacao = penalizacao_marca + penalizacao_medida + penalizacao_descricao
    
    # Exibindo as penalidades
    print(f'Marca Consultada: {consulta_marca}\nMarca Original: {input_marca}')
    print(f"Similaridade da Marca: {similaridade_marca*100:.4f}%\nPenalização da Marca: {penalizacao_marca*100:.4f}%\n\n")
    
    print(f'Medida Consultada: {consulta_medida}\nMedida Original: {input_medida}')
    print(f"Similaridade da Medida: {similaridade_medida*100:.4f}%\nPenalização da Medida: {penalizacao_medida*100:.4f}%\n\n")
    
    print(f'Descrição Consultada: {consulta_descricao}\nDescrição Original: {input_descricao}')
    print(f"Similaridade da Descrição: {similaridade_descricao*100:.4f}%\nPenalização da Descrição: {penalizacao_descricao*100:.4f}%\n\n")

    print(f'Penalização Total: {penalizacao*100:.4f}%\n\n')

    return penalizacao
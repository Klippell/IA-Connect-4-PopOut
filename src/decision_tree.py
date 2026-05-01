import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 1. FUNÇÕES DE PRÉ-PROCESSAMENTO (Datasets e Limpeza)
# =====================================================================

def clean_conflicting_data(df, target_name="target_move"):
    """
    [Prioridade 2 - PopOut] Resolve conflitos do MCTS no dataset.
    Agrupa todas as linhas que têm o mesmo estado de tabuleiro.
    Se houver jogadas diferentes para o mesmo estado, mantém a mais frequente.
    """
    feature_cols = [col for col in df.columns if col != target_name]
    
    print(f"[SISTEMA] A limpar ambiguidades do MCTS... Tamanho original: {len(df)} linhas.")
    
    # Agrupa pelas features do tabuleiro e guarda apenas a jogada mais frequente (moda)
    df_clean = df.groupby(feature_cols)[target_name].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
    
    print(f"[SISTEMA] Limpeza concluída! Novo tamanho do dataset: {len(df_clean)} linhas.")
    return df_clean

def discretizar_largura_igual(df, colunas):
    """
    [Dataset Iris] - Abordagem Ingénua
    Divide a 'régua' numérica em 3 pedaços de tamanhos matematicamente iguais.
    (Pode deixar grupos vazios se os dados não estiverem bem distribuídos).
    """
    df_out = df.copy()
    labels = ["Pequeno", "Médio", "Grande"]
    for col in colunas:
        df_out[col] = pd.cut(df_out[col], bins=3, labels=labels)
    return df_out

def discretizar_frequencia_igual(df, colunas):
    """
    [Dataset Iris] - Abordagem Inteligente
    Divide os dados em 3 grupos com o mesmo NÚMERO de elementos (Quantis).
    Abordagem ideal para maximizar o Ganho de Informação na Árvore.
    """
    df_out = df.copy()
    labels = ["Pequeno", "Médio", "Grande"]
    for col in colunas:
        # qcut garante que cada categoria tem ~33% dos dados (evita grupos vazios)
        df_out[col] = pd.qcut(df_out[col], q=3, labels=labels, duplicates='drop')
    return df_out

# =====================================================================
# 2. FUNÇÃO DE VISUALIZAÇÃO DA ÁRVORE
# =====================================================================

def mostrar_arvore_visual(node, depth=0):
    """
    [Exigência do Guião]
    Percorre a árvore de decisão recursivamente e imprime-a no terminal
    com formatação hierárquica (estilo explorador de ficheiros).
    """
    espaco = "    " * depth
    
    # Se for uma folha (resultado final/folha da árvore)
    if node.result is not None:
        print(espaco + f"➔ [CLASSE FINAL]: {node.result}")
        return
        
    # Se for um nó de decisão (faz uma pergunta sobre uma feature)
    print(espaco + f"■ PERGUNTA: Qual é a característica da '{node.feature}'?")
    for valor_ramo, child_node in node.children.items():
        print(espaco + f"    ├── Se for '{valor_ramo}':")
        # Chama a si mesma para imprimir o próximo nível (recursão)
        mostrar_arvore_visual(child_node, depth + 1)

# =====================================================================
# 3. NÚCLEO DO ALGORITMO ID3
# =====================================================================

class Node:
    """Representa um nó na árvore de decisão."""
    def __init__(self, feature=None, value=None, result=None, majority_class=None):
        self.feature = feature      # A feature a ser testada neste nó
        self.value = value          # O valor da branch que levou a este nó
        self.result = result        # Se for folha, guarda a classe final
        self.majority_class = majority_class # <-- Fallback: Guarda a decisão de segurança
        self.children = {}          # Dicionário de nós filhos

class DecisionTreeID3:
    """Implementação matemática do algoritmo ID3 de raiz."""
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def _entropy(self, target_col):
        """Calcula a entropia (nível de desordem) de um conjunto de dados."""
        elements, counts = np.unique(target_col, return_counts=True)
        entropy = 0
        for i in range(len(elements)):
            prob = counts[i] / np.sum(counts)
            entropy -= prob * math.log2(prob)
        return entropy

    def _information_gain(self, data, split_attribute_name, target_name="target"):
        """Calcula o Ganho de Informação de uma determinada feature."""
        total_entropy = self._entropy(data[target_name])
        
        vals, counts = np.unique(data[split_attribute_name], return_counts=True)
        weighted_entropy = 0
        for i in range(len(vals)):
            subset = data[data[split_attribute_name] == vals[i]]
            prob = counts[i] / np.sum(counts)
            weighted_entropy += prob * self._entropy(subset[target_name])
            
        return total_entropy - weighted_entropy

    def fit(self, data, target_name="target", features=None, depth=0):
        """Treina a árvore de decisão usando o algoritmo ID3."""
        if features is None:
            features = [col for col in data.columns if col != target_name]
            
        # [Segurança] Calcular a classe maioritária deste sub-dataset antes de dividir
        current_majority_class = data[target_name].mode()[0]
            
        # CASO BASE 1: Se todos os targets forem iguais, retorna uma folha
        if len(np.unique(data[target_name])) <= 1:
            return Node(result=np.unique(data[target_name])[0])
            
        # CASO BASE 2: Se não houver mais features ou atingiu profundidade máxima
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(result=current_majority_class)
            
        # Encontrar a feature com maior Ganho de Informação
        item_values = [self._information_gain(data, feature, target_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # Criar o nó GUARDANDO a classe maioritária para segurança
        node = Node(feature=best_feature, majority_class=current_majority_class)
        
        remaining_features = [f for f in features if f != best_feature]
        
        # Criar os ramos (filhos)
        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value]
            if len(subset) == 0:
                node.children[value] = Node(result=current_majority_class)
            else:
                node.children[value] = self.fit(subset, target_name, remaining_features, depth + 1)
                
        self.root = node
        return node

    def _predict_single(self, node, row):
        """Faz a previsão para uma única linha descendo pela árvore."""
        if node.result is not None:
            return node.result
            
        feature_value = row[node.feature]
        
        # [Prioridade 1 RESOLVIDA]: Fallback para o valor mais comum se caminho for desconhecido
        if feature_value not in node.children:
            return node.majority_class 
            
        return self._predict_single(node.children[feature_value], row)

    def predict(self, test_data):
        """Aplica a árvore a novos dados."""
        predictions = []
        for _, row in test_data.iterrows():
            pred = self._predict_single(self.root, row)
            predictions.append(pred)
        return predictions
  
"""
Módulo de Machine Learning - Árvore de Decisão ID3
Contém ferramentas de pré-processamento de datasets discretos, a implementação
do algoritmo ID3 construído de raiz com cálculo de Entropia/Ganho de Informação,
e funções para visualização gráfica do modelo treinado.
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =====================================================================
# PRÉ-PROCESSAMENTO DE DADOS
# =====================================================================

def clean_conflicting_data(df, target_name="target_move"):
    """
    Remove ambiguidades do dataset agrupando linhas com o mesmo estado de tabuleiro.
    Em caso de conflito (múltiplas jogadas para o mesmo estado), preserva a jogada mais frequente.
    """
    feature_cols = [col for col in df.columns if col != target_name]
    
    print(f"[SISTEMA] A limpar ambiguidades... Tamanho original: {len(df)} linhas.")
    df_clean = df.groupby(feature_cols)[target_name].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
    print(f"[SISTEMA] Limpeza concluída. Novo tamanho: {len(df_clean)} linhas.")
    
    return df_clean

def discretizar_largura_igual(df, colunas):
    """
    Aplica discretização Equal-Width (2 divisões).
    Mapeia valores numéricos contínuos para categorias com notação matemática de intervalos.
    """
    df_out = df.copy()
    nomes_base = ["Pequeno", "Grande"]
    
    for col in colunas:
        serie_intervalos = pd.cut(df_out[col], bins=2)
        categorias = serie_intervalos.cat.categories
        
        mapa = {}
        for i, cat in enumerate(categorias):
            nome = nomes_base[i] if i < len(nomes_base) else f"Grupo {i}"
            abertura_esq = "[" if cat.closed in ["left", "both"] else "]"
            abertura_dir = "]" if cat.closed in ["right", "both"] else "["
            
            mapa[cat] = f"{nome}\n{abertura_esq}{cat.left:.1f}, {cat.right:.1f}{abertura_dir}"
            
        df_out[col] = serie_intervalos.map(mapa).astype(str)
        
    return df_out

def discretizar_frequencia_igual(df, colunas):
    """
    Aplica discretização Equal-Frequency via Quantis (3 divisões).
    Ideal para maximizar o Ganho de Informação ao garantir o mesmo volume de dados por ramo.
    """
    df_out = df.copy()
    nomes_base = ["Pequeno", "Médio", "Grande"]
    
    for col in colunas:
        serie_intervalos = pd.qcut(df_out[col], q=3, duplicates='drop')
        categorias = serie_intervalos.cat.categories
        
        mapa = {}
        for i, cat in enumerate(categorias):
            nome = nomes_base[i] if i < len(nomes_base) else f"Grupo {i}"
            abertura_esq = "[" if cat.closed in ["left", "both"] else "]"
            abertura_dir = "]" if cat.closed in ["right", "both"] else "["
            
            mapa[cat] = f"{nome}\n{abertura_esq}{cat.left:.1f}, {cat.right:.1f}{abertura_dir}"
            
        df_out[col] = serie_intervalos.map(mapa).astype(str)
        
    return df_out

# =====================================================================
# ALGORITMO ID3 (ÁRVORE DE DECISÃO)
# =====================================================================

def mostrar_arvore_visual(node, depth=0):
    """
    Imprime a estrutura da árvore de decisão no terminal de forma hierárquica.
    """
    espaco = "    " * depth
    
    if node.result is not None:
        print(espaco + f"➔ [CLASSE FINAL]: {node.result}")
        return
        
    print(espaco + f"■ PERGUNTA: Qual é a característica da '{node.feature}'?")
    for valor_ramo, child_node in node.children.items():
        print(espaco + f"    ├── Se for '{valor_ramo}':")
        mostrar_arvore_visual(child_node, depth + 1)


class Node:
    """
    Representa um nó (decisão ou folha) na Árvore de Decisão.
    """
    def __init__(self, feature=None, value=None, result=None, majority_class=None):
        self.feature = feature               
        self.value = value                   
        self.result = result                 
        self.majority_class = majority_class 
        self.children = {}                   
        self.branch_counts = {}              

class DecisionTreeID3:
    """
    Implementação do classificador Decision Tree suportado pelo algoritmo matemático ID3.
    """
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def _entropy(self, target_col):
        """Calcula a impureza (entropia de Shannon) de um conjunto de dados."""
        elements, counts = np.unique(target_col, return_counts=True)
        entropy = 0
        for i in range(len(elements)):
            prob = counts[i] / np.sum(counts)
            entropy -= prob * math.log2(prob)
        return entropy

    def _information_gain(self, data, split_attribute_name, target_name="target"):
        """Calcula a redução de entropia esperada ao particionar os dados por um atributo."""
        total_entropy = self._entropy(data[target_name])
        
        vals, counts = np.unique(data[split_attribute_name], return_counts=True)
        weighted_entropy = 0
        
        for i in range(len(vals)):
            subset = data[data[split_attribute_name] == vals[i]]
            prob = counts[i] / np.sum(counts)
            weighted_entropy += prob * self._entropy(subset[target_name])
            
        return total_entropy - weighted_entropy

    def fit(self, data, target_name="target", features=None, depth=0):
        """
        Constrói recursivamente a árvore de decisão selecionando features com
        o maior Ganho de Informação.
        """
        if features is None:
            features = [col for col in data.columns if col != target_name]
            
        current_majority_class = data[target_name].mode()[0]
            
        # Caso Base 1: Nó puro (apenas uma classe restante)
        if len(np.unique(data[target_name])) <= 1:
            return Node(result=np.unique(data[target_name])[0])
            
        # Caso Base 2: Fim de linha (sem features ou profundidade máxima atingida)
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(result=current_majority_class)
            
        # Determinar o melhor atributo para dividir
        item_values = [self._information_gain(data, feature, target_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        node = Node(feature=best_feature, majority_class=current_majority_class)
        remaining_features = [f for f in features if f != best_feature]
        
        # Gerar os ramos
        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value]
            node.branch_counts[value] = len(subset)
            
            if len(subset) == 0:
                node.children[value] = Node(result=current_majority_class)
            else:
                node.children[value] = self.fit(subset, target_name, remaining_features, depth + 1)
                
        self.root = node
        return node

    def _predict_single(self, node, row):
        """Atravessa a árvore para classificar uma única observação."""
        if node.result is not None:
            return node.result
            
        feature_value = row[node.feature]
        
        # Fallback de segurança para valores de features não vistos no treino
        if feature_value not in node.children:
            return node.majority_class 
            
        return self._predict_single(node.children[feature_value], row)

    def predict(self, test_data):
        """Retorna uma lista de previsões para o dataframe fornecido."""
        return [self._predict_single(self.root, row) for _, row in test_data.iterrows()]


# =====================================================================
# AVALIAÇÃO E RENDERIZAÇÃO GRÁFICA
# =====================================================================

def calcular_metricas(arvore, df_teste, nome_da_coluna_alvo):
    """
    Avalia a exatidão (Accuracy) da árvore num conjunto de dados de teste.
    """
    previsoes = arvore.predict(df_teste)
    valores_reais = df_teste[nome_da_coluna_alvo].tolist()
    acertos = sum(1 for p, r in zip(previsoes, valores_reais) if p == r)
    return {'total': len(valores_reais), 'corretas': acertos, 'acuracia': (acertos/len(valores_reais))*100}

def plotar_arvore_decisao(node, titulo="Árvore de Decisão", metricas=None):
    """
    Gera um diagrama visual 2D detalhado da árvore de decisão treinada
    recorrendo à biblioteca NetworkX.
    """
    G = nx.DiGraph()
    labels = {}
    node_colors = {}
    
    def percorrer_arvore(no_atual, id_atual="0"):
        G.add_node(id_atual)
        if no_atual.result is not None:
            labels[id_atual] = f"Classe:\n{no_atual.result}"
            node_colors[id_atual] = '#d5f5e3'
        else:
            labels[id_atual] = f"{no_atual.feature}?"
            node_colors[id_atual] = '#ebf5fb'
            
            for i, (valor_ramo, filho) in enumerate(no_atual.children.items()):
                id_filho = f"{id_atual}_{i}_{valor_ramo}"
                qtd_amostras = no_atual.branch_counts.get(valor_ramo, 0)
                label_completa = f"{valor_ramo}\n[n={qtd_amostras}]"
                
                G.add_edge(id_atual, id_filho, label=label_completa)
                percorrer_arvore(filho, id_filho)

    percorrer_arvore(node)
    
    # Algoritmo de posicionamento anti-sobreposição (Leaf-based Layout)
    def layout_arvore_perfeita(G, root="0"):
        pos = {}
        leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
        for i, leaf in enumerate(leaves):
            pos[leaf] = [i * 2.0, 0] 

        def assign_x(n):
            children = list(G.successors(n))
            if not children: return pos[n][0]
            child_xs = [assign_x(c) for c in children]
            x = sum(child_xs) / len(child_xs)
            pos[n] = [x, 0]
            return x
        assign_x(root)

        def assign_y(n, depth):
            pos[n][1] = -depth * 1.5 
            for child in G.successors(n):
                assign_y(child, depth + 1)
        assign_y(root, 0)
        return pos

    pos = layout_arvore_perfeita(G)

    plt.figure(figsize=(22, 12)) 
    nx.draw_networkx_edges(G, pos, edge_color='#95a5a6', arrows=True, arrowsize=20, width=2.0)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='#c0392b', 
                                 font_size=12, font_weight='bold', 
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.5))
    
    for node_id, (x, y) in pos.items():
        cor = node_colors[node_id]
        plt.text(x, y, labels[node_id], ha='center', va='center', fontsize=12, fontweight='bold', color='#2c3e50',
                 bbox=dict(facecolor=cor, edgecolor='#95a5a6', boxstyle='round,pad=1', linewidth=2), zorder=5)
        
    if metricas:
        texto_metricas = (
            f"DADOS DO MODELO\n"
            f"---------------------------------\n"
            f"Profundidade: {metricas.get('profundidade', '?')} níveis\n" # <--- LINHA NOVA
            f"Total de Amostras: {metricas.get('total', '?')}\n"
            f"Previsões Corretas: {metricas.get('corretas', '?')}\n"
            f"Acurácia Global: {metricas.get('acuracia', 0):.2f}%"
        )
        plt.text(0.02, 0.98, texto_metricas, transform=plt.gca().transAxes,
                 fontsize=14, fontweight='bold', color='#154360',
                 bbox=dict(facecolor='#fcf3cf', edgecolor='#f1c40f', boxstyle='round,pad=1', linewidth=2, alpha=0.9),
                 verticalalignment='top')

    plt.title(titulo, fontsize=22, fontweight='bold', color='#2c3e50', pad=30)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
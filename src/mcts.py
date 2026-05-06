"""
Módulo MCTS (Monte Carlo Tree Search)
Implementa a lógica do agente inteligente para jogar PopOut Connect 4,
utilizando procura em árvore combinada com a fórmula UCT e heurísticas de 1-Ply (Heavy Playout).
"""

import math
import random

class Node:
    """
    Representa um nó na árvore de procura MCTS.
    Armazena o estado do jogo e estatísticas vitais para o cálculo UCT.
    """
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self._get_valid_moves(self.state)

    def _get_valid_moves(self, game_state):
        """
        Calcula todas as jogadas (drop ou pop) legais a partir do estado atual.
        """
        moves = []
        board_full = game_state.is_board_full()
        for col in range(7):
            if not board_full and game_state.board[0][col] == 0:
                moves.append((col, 'd'))
            if game_state.board[5][col] == game_state.current_player:
                moves.append((col, 'p'))
        return moves

class MCTS:
    """
    Controlador do Algoritmo Monte Carlo Tree Search.
    Gere as fases de seleção, expansão, simulação e retropropagação.
    """
    def __init__(self, iterations=1000, c=1.41, max_children=None, max_depth=60):
        self.iterations = iterations
        self.c = c
        self.max_children = max_children
        self.max_depth = max_depth

    def search(self, initial_state):
        """
        Inicia a procura iterativa e retorna a melhor jogada encontrada.
        (A verificação de vitória imediata (1-ply) na raiz foi movida 
        para o motor do jogo (game.py) para evitar redundância).
        """
        # Vai direto para a criação da raiz, pois o game.py já filtrou as jogadas óbvias
        root = Node(state=initial_state.clone())

        for _ in range(self.iterations):
            node = self._select(root)
            winner = self._simulate(node.state)
            self._backpropagate(node, winner)

        if not root.children: 
            return None
            
        return max(root.children, key=lambda n: n.visits).move

    def _select(self, node):
        """
        Percorre a árvore desde a raiz até encontrar um nó expansível
        ou um estado final, baseando-se no valor UCT de cada filho.
        """
        while not self._is_terminal(node.state):
            if len(node.untried_moves) > 0:
                if self.max_children is None or len(node.children) < self.max_children:
                    return self._expand(node)
            
            if not node.children: 
                break
                
            node = self._best_child_uct(node)
            
        return node

    def _expand(self, node):
        """
        Expande a árvore instanciando um novo filho através de uma jogada aleatória não testada.
        """
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        new_state = node.state.clone()
        col, m_type = move
        
        if m_type == 'd': 
            new_state.drop_piece(col, new_state.current_player)
        else: 
            new_state.pop_piece(col, new_state.current_player)
        
        new_state.current_player = 2 if new_state.current_player == 1 else 1
        
        child = Node(state=new_state, parent=node, move=move)
        node.children.append(child)
        
        return child

    def _simulate(self, state):
        """
        Conduz uma simulação do jogo até um estado final (rollout).
        MANTÉM heurísticas simples (1-ply) para vitórias diretas ou bloqueios
        para não simular o futuro de forma "cega".
        """
        temp_state = state.clone()
        depth = 0
        
        while True:
            # Proteção contra simulação excessivamente longa (Loops infinitos de PopOut)
            if self.max_depth is not None and depth >= self.max_depth: 
                return None 

            last_p = 2 if temp_state.current_player == 1 else 1
            winner = temp_state.check_winner_after_move(last_p)
            
            if winner is not None: 
                return winner
            
            moves = Node(temp_state).untried_moves
            if not moves: 
                return None 
            
            p_current = temp_state.current_player
            p_opponent = 2 if p_current == 1 else 1
            
            # Heavy Playout: Tenta ganhar na própria imaginação
            m = temp_state.get_winning_move(p_current)
            
            if m is None:
                m_defend = temp_state.get_winning_move(p_opponent, check_pop=False)
                if m_defend is not None and m_defend[1] == 'd':
                    if (m_defend[0], 'd') in moves:
                        m = (m_defend[0], 'd')
            
            if m is None:
                m = random.choice(moves)
            
            if m[1] == 'd': 
                temp_state.drop_piece(m[0], p_current, record=False)
            else: 
                temp_state.pop_piece(m[0], p_current, record=False)
            
            temp_state.current_player = p_opponent
            depth += 1

    def _backpropagate(self, node, winner):
        """
        Propaga o resultado da simulação (vitória/derrota/empate) de volta pela árvore
        até à raiz, atualizando os contadores de visitas e vitórias.
        """
        while node is not None:
            node.visits += 1
            player_who_just_moved = 2 if node.state.current_player == 1 else 1
            
            if winner == player_who_just_moved:
                node.wins += 1
            elif winner is None:
                node.wins += 0.5
                
            node = node.parent

    def _best_child_uct(self, node):
        """
        Seleciona o filho com o maior valor na fórmula Upper Confidence Bound for Trees (UCT).
        """
        def uct(child):
            return (child.wins / child.visits) + self.c * math.sqrt(math.log(node.visits) / child.visits)
        return max(node.children, key=uct)

    def _is_terminal(self, state):
        """
        Verifica se o estado atual representa o fim do jogo (vitória de alguém ou empate).
        """
        last_p = 2 if state.current_player == 1 else 1
        if state.check_winner_after_move(last_p) is not None: 
            return True
        if not Node(state).untried_moves: 
            return True
        return False
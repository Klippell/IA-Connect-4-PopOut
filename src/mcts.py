import math
import random
import copy

class Node:
    """
    Cada Nó representa um estado do tabuleiro. 
    Guarda as estatísticas para a fórmula UCT.
    """
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move  # A jogada que gerou este estado: (coluna, tipo)
        self.children = []
        self.wins = 0    # w_i
        self.visits = 0  # n_i
        
        # Lista de jogadas que o MCTS ainda não explorou a partir deste nó
        self.untried_moves = self._get_valid_moves(self.state)

    def _get_valid_moves(self, game_state):
        """Retorna todas as jogadas permitidas pelas regras do PopOut."""
        moves = []
        board_full = game_state.is_board_full()
        for col in range(7):
            # DROP: Se a coluna não estiver cheia
            if not board_full and game_state.board[0][col] == 0:
                moves.append((col, 'd'))
            # POP: Se a peça na base for do jogador atual
            if game_state.board[5][col] == game_state.current_player:
                moves.append((col, 'p'))
        return moves

class MCTS:
    """
    Algoritmo Monte Carlo Tree Search.
    CONFIGURAÇÕES FÁCEIS:
    - iterations: Quanto o PC 'pensa' (ex: 500, 1000, 5000)
    - c: Peso da exploração (UCT). 1.41 é o padrão.
    - max_children: Limite de ramos por nó (Exigência do guião).
    """
    def __init__(self, iterations=1000, c=1.41, max_children=None):
        self.iterations = iterations 
        self.c = c                   
        self.max_children = max_children 

    def search(self, initial_state):
        """Inicia a procura da melhor jogada."""
        p_current = initial_state.current_player
        p_opponent = 2 if p_current == 1 else 1

        # 1-Ply Lookahead na Raiz: Garante ataque ou defesa imediata antes de simular
        m_win = initial_state.get_winning_move(p_current)
        if m_win is not None:
            return m_win
            
        m_defend = initial_state.get_winning_move(p_opponent)
        if m_defend is not None:
            # Para bloquear o oponente, o mais provável/seguro é um drop na mesma coluna
            return (m_defend[0], 'd')

        root = Node(state=initial_state.clone())

        for _ in range(self.iterations):
            # 1. SELEÇÃO E 2. EXPANSÃO
            node = self._select(root)
            # 3. SIMULAÇÃO (Rollout)
            winner = self._simulate(node.state)
            # 4. RETROPROPAGAÇÃO
            self._backpropagate(node, winner)

        # A melhor jogada é a do filho que foi MAIS VISITADO
        if not root.children: return None
        return max(root.children, key=lambda n: n.visits).move

    def _select(self, node):
        """Desce pela árvore escolhendo os melhores nós via UCT."""
        while self._is_terminal(node.state) == False:
            # Se ainda houver jogadas para tentar e não batemos no limite de filhos...
            if len(node.untried_moves) > 0:
                if self.max_children is None or len(node.children) < self.max_children:
                    return self._expand(node)
            
            # Se o nó estiver 'cheio', escolhemos o melhor filho via UCT para continuar a descer
            if not node.children: break
            node = self._best_child_uct(node)
        return node

    def _expand(self, node):
        """Cria um novo nó filho aplicando uma jogada nova."""
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        new_state = node.state.clone()
        col, m_type = move
        if m_type == 'd': new_state.drop_piece(col, new_state.current_player)
        else: new_state.pop_piece(col, new_state.current_player)
        
        # Muda o turno para o próximo jogador na simulação
        new_state.current_player = 2 if new_state.current_player == 1 else 1
        
        child = Node(state=new_state, parent=node, move=move)
        node.children.append(child)
        return child

    def _simulate(self, state):
        """Joga (guiado por heurística de 1-ply) até ao fim para avaliar o nó."""
        temp_state = state.clone()
        while True:
            # Verifica se alguém ganhou na jogada anterior
            last_p = 2 if temp_state.current_player == 1 else 1
            winner = temp_state.check_winner_after_move(last_p)
            if winner is not None: return winner
            
            # Se não houver mais jogadas, é empate
            moves = Node(temp_state).untried_moves
            if not moves: return None 
            
            p_current = temp_state.current_player
            p_opponent = 2 if p_current == 1 else 1
            
            # 1-Ply Heurística de Domínio
            # 1. Ataque
            m = temp_state.get_winning_move(p_current)
            
            # 2. Defesa (Bloqueio Simples)
            if m is None:
                m_defend = temp_state.get_winning_move(p_opponent)
                if m_defend is not None:
                    if (m_defend[0], 'd') in moves:
                        m = (m_defend[0], 'd')
            
            # 3. Aleatório
            if m is None:
                m = random.choice(moves)
            
            # Executa a jogada escolhida com record=False por eficiência
            if m[1] == 'd': temp_state.drop_piece(m[0], p_current, record=False)
            else: temp_state.pop_piece(m[0], p_current, record=False)
            
            temp_state.current_player = p_opponent

    def _backpropagate(self, node, winner):
        """Sobe a árvore atualizando visitas e vitórias."""
        while node is not None:
            node.visits += 1
            # Se o vencedor foi quem fez a jogada para chegar a este nó, +1
            player_who_just_moved = 2 if node.state.current_player == 1 else 1
            if winner == player_who_just_moved:
                node.wins += 1
            elif winner is None:
                node.wins += 0.5
            node = node.parent

    def _best_child_uct(self, node):
        """Fórmula UCT: Exploração vs Aproveitamento."""
        def uct(child):
            # Exploração (Aproveitamento) + C * sqrt( ln(Total_Pai) / n_Filho )
            return (child.wins / child.visits) + self.c * math.sqrt(math.log(node.visits) / child.visits)
        return max(node.children, key=uct)

    def _is_terminal(self, state):
        """Verifica se o jogo acabou."""
        last_p = 2 if state.current_player == 1 else 1
        if state.check_winner_after_move(last_p) is not None: return True
        if not Node(state).untried_moves: return True
        return False
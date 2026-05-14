"""
Módulo MCTS (Monte Carlo Tree Search)
Implementa o agente inteligente para jogar PopOut Connect 4 em Python puro.
"""

import math
import random

# =====================================================================
# 1. HEURÍSTICAS E OTIMIZAÇÕES DE TABULEIRO (Rápidas)
# =====================================================================

def _build_lines():
    lines = []
    for r in range(6):
        for c in range(4): lines.append(((r,c), (r,c+1), (r,c+2), (r,c+3))) # Horizontais
    for c in range(7):
        for r in range(3): lines.append(((r,c), (r+1,c), (r+2,c), (r+3,c))) # Verticais
    for r in range(3):
        for c in range(4): lines.append(((r,c), (r+1,c+1), (r+2,c+2), (r+3,c+3))) # Diagonais ↘
    for r in range(3, 6):
        for c in range(4): lines.append(((r,c), (r-1,c+1), (r-2,c+2), (r-3,c+3))) # Diagonais ↗
    return lines

_LINES = _build_lines()
_CELL_LI = [[ [i for i, ln in enumerate(_LINES) if (r,c) in ln] for c in range(7)] for r in range(6)]
_COL_LI = [ list(set(i for r in range(6) for i in _CELL_LI[r][c])) for c in range(7) ]

def _win(board, piece, indices):
    """
    Verificação otimizada de vitória testando apenas as linhas afetadas pela jogada.
    """
    for i in indices:
        a, b, c, d = _LINES[i]
        if board[a[0], a[1]] == piece and board[b[0], b[1]] == piece and \
           board[c[0], c[1]] == piece and board[d[0], d[1]] == piece:
            return True
    return False

def get_valid_moves(board, player):
    """
    Retorna as jogadas legais analisando o estado atual da matriz.
    """
    full = not (board[0] == 0).any()
    return [(c, 'd') for c in range(7) if not full and board[0,c] == 0] + \
           [(c, 'p') for c in range(7) if board[5,c] == player]

def get_winning_move(board, piece, moves, check_pop=True):
    """
    Simula os movimentos possíveis para encontrar uma jogada vencedora imediata.
    """
    for col, m_type in moves:
        if m_type == 'd':
            for r in range(5, -1, -1):
                if board[r, col] == 0:
                    board[r, col] = piece
                    won = _win(board, piece, _CELL_LI[r][col])
                    board[r, col] = 0
                    if won: return (col, 'd'), r
                    break
        elif check_pop and m_type == 'p':
            col_save = board[:, col].copy()
            board[1:, col] = board[:-1, col]; board[0, col] = 0
            won = _win(board, piece, _COL_LI[col])
            board[:, col] = col_save
            if won: return (col, 'p'), -1
    return None, -1


# =====================================================================
# 2. ALGORITMO MCTS
# =====================================================================

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = get_valid_moves(state.board, state.current_player)

class MCTS:
    def __init__(self, iterations=10000, c=1.41, max_children=None, max_depth=60, pure_mode=False):
        self.iterations = iterations
        self.c = c
        self.max_children = max_children
        self.max_depth = max_depth
        self.pure_mode = pure_mode

    def search(self, initial_state):
        p = initial_state.current_player
        moves = get_valid_moves(initial_state.board, p)
        
        if not self.pure_mode:
            m_win, _ = get_winning_move(initial_state.board, p, moves)
            if m_win: return m_win
            
            m_def, _ = get_winning_move(initial_state.board, 2 if p==1 else 1, [(c,'d') for c, t in moves if t=='d'], False)
            if m_def: return m_def

        root = Node(state=initial_state.clone())
        
        for _ in range(self.iterations):
            node = self._select(root)
            winner = self._simulate(node.state)
            self._backpropagate(node, winner)

        return max(root.children, key=lambda n: n.visits).move if root.children else None

    def _select(self, node):
        while not self._is_terminal(node.state):
            if node.untried_moves and (self.max_children is None or len(node.children) < self.max_children):
                return self._expand(node)
            if not node.children: 
                break
            # Seleciona o melhor filho usando o algoritmo UCB1
            node = max(node.children, key=lambda c: (c.wins/c.visits) + self.c * math.sqrt(math.log(node.visits)/c.visits))
        return node

    def _expand(self, node):
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        s = node.state.clone()
        if move[1] == 'd': s.drop_piece(move[0], s.current_player)
        else: s.pop_piece(move[0], s.current_player)
        s.current_player = 2 if s.current_player == 1 else 1
        
        child = Node(state=s, parent=node, move=move)
        node.children.append(child)
        return child

    def _simulate(self, state):
        b = state.board.copy()
        p = state.current_player
        depth = 0
        opp = 2 if p == 1 else 1
        
        # Retorna se o estado simulado inicial já for um estado de vitória
        if _win(b, opp, range(len(_LINES))): return opp

        while self.max_depth is None or depth < self.max_depth:
            moves = get_valid_moves(b, p)
            if not moves: return None
            
            opp = 2 if p == 1 else 1
            m, m_row = None, -1
            
            if not self.pure_mode:
                m, m_row = get_winning_move(b, p, moves)
                if not m:
                    m_def, r_def = get_winning_move(b, opp, [(c,'d') for c, t in moves if t=='d'], False)
                    if m_def: m, m_row = m_def, r_def

            if not m:
                m = random.choice(moves)
                if m[1] == 'd':
                    for r in range(5, -1, -1):
                        if b[r, m[0]] == 0: 
                            m_row = r
                            break

            if m[1] == 'd':
                b[m_row, m[0]] = p
                if _win(b, p, _CELL_LI[m_row][m[0]]): return p
            else:
                b[1:, m[0]] = b[:-1, m[0]]
                b[0, m[0]] = 0
                w_p, w_opp = _win(b, p, _COL_LI[m[0]]), _win(b, opp, _COL_LI[m[0]])
                if w_p or w_opp: return p if (w_p and w_opp) or w_p else opp

            p = opp
            depth += 1
            
        return None

    def _backpropagate(self, node, winner):
        while node:
            node.visits += 1
            mover = 2 if node.state.current_player == 1 else 1
            node.wins += 1 if winner == mover else (0.5 if winner is None else 0)
            node = node.parent

    def _is_terminal(self, state):
        last_p = 2 if state.current_player == 1 else 1
        if state.check_winner_after_move(last_p) is not None: 
            return True
        return len(get_valid_moves(state.board, state.current_player)) == 0
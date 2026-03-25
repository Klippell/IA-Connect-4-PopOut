import numpy as np
import copy
from src.ui import draw_board
from src.mcts import MCTS

# =================================================================
# CONFIGURAÇÕES E CONSTANTES DO JOGO
# =================================================================
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1  
PLAYER2 = 2  

class PopOutGame:
    """Classe que gere o estado lógico e as regras do PopOut."""
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = PLAYER1
        self.state_history = {} 
        self._record_state()

    def get_state_key(self):
        return tuple(map(tuple, self.board))

    def _record_state(self):
        key = self.get_state_key()
        self.state_history[key] = self.state_history.get(key, 0) + 1

    def drop_piece(self, col, piece):
        for r in range(ROWS-1, -1, -1):
            if self.board[r][col] == EMPTY:
                self.board[r][col] = piece 
                self._record_state()
                return True
        return False 

    def pop_piece(self, col, piece):
        if self.board[ROWS-1][col] == piece:
            for r in range(ROWS-1, 0, -1):
                self.board[r][col] = self.board[r-1][col]
            self.board[0][col] = EMPTY
            self._record_state()
            return True
        return False

    def check_win(self, piece):
        for c in range(COLS-3):
            for r in range(ROWS):
                if all(self.board[r][c+i] == piece for i in range(4)): return True
        for c in range(COLS):
            for r in range(ROWS-3):
                if all(self.board[r+i][c] == piece for i in range(4)): return True
        for c in range(COLS-3):
            for r in range(ROWS-3):
                if all(self.board[r+i][c+i] == piece for i in range(4)): return True
        for c in range(COLS-3):
            for r in range(3, ROWS):
                if all(self.board[r-i][c+i] == piece for i in range(4)): return True
        return False

    def check_winner_after_move(self, player_who_moved):
        p1 = self.check_win(PLAYER1)
        p2 = self.check_win(PLAYER2)
        if p1 and p2: return player_who_moved 
        if p1: return PLAYER1
        if p2: return PLAYER2
        return None

    def is_board_full(self):
        return all(self.board[0][c] != EMPTY for c in range(COLS))

    def check_repetition(self):
        return self.state_history.get(self.get_state_key(), 0) >= 3

    def clone(self):
        return copy.deepcopy(self)

# =================================================================
# CONTROLADOR DE JOGO E INTERFACE DE INPUT
# =================================================================

def get_human_move(game):
    p_name = "X" if game.current_player == PLAYER1 else "O"
    if game.check_repetition():
        print(f"\n[AVISO] Este estado repetiu-se 3 vezes.")
        if input(f"Jogador {p_name}, aceita o empate? (s/n): ").lower() == 's': return "DRAW"

    while True:
        try:
            cmd = input(f"[{p_name}] Jogada (ex: 3 d) ou 'q': ").lower().strip()
            if cmd in ['q', 'sair']: return "QUIT"
            parts = cmd.split()
            col, m_type = int(parts[0]), parts[1]

            if game.is_board_full() and m_type == 'd':
                print("[!] Tabuleiro cheio! Apenas 'p' é permitido.")
                continue

            if m_type == 'd' and game.drop_piece(col, game.current_player): return "MOVED"
            if m_type == 'p' and game.pop_piece(col, game.current_player): return "MOVED"
            print("[!] Movimento inválido (coluna cheia ou peça errada).")
        except:
            print("[!] Erro de formato! Use: 'COLUNA TIPO' (ex: 3 d).")

# =================================================================
# CONTROLADOR DE JOGO (PAINEL DE CONFIGURAÇÃO)
# =================================================================

def play_game(mode, ia_std=None, ia_p1=None, ia_p2=None):
    game = PopOutGame()
    IT_HUMAN_VS_IA, C_HUMAN_VS_IA, MAX_C_HUMAN_IA = 1500, 1.41, None
    IT_P1, C_P1, MAX_C_P1 = 500, 1.41, None
    IT_P2, C_P2, MAX_C_P2 = 10000, 1.41, None

    if ia_std is None: ia_std = MCTS(iterations=IT_HUMAN_VS_IA, c=C_HUMAN_VS_IA, max_children=MAX_C_HUMAN_IA)
    else: IT_HUMAN_VS_IA = ia_std.iterations 

    if ia_p1 is None: ia_p1 = MCTS(iterations=IT_P1, c=C_P1, max_children=MAX_C_P1)
    if ia_p2 is None: ia_p2 = MCTS(iterations=IT_P2, c=C_P2, max_children=MAX_C_P2)

    human_p = PLAYER1
    if mode == 2:
        print("\n--- Configuração de Partida ---")
        print("1. Ser Jogador 1 (X - Vermelho) [Começa]")
        print("2. Ser Jogador 2 (O - Amarelo) [IA começa]")
        human_p = PLAYER1 if input("Escolha (1/2): ").strip() == "1" else PLAYER2

    while True:
        draw_board(game)
        move = None

        if mode == 1: 
            if get_human_move(game) in ["QUIT", "DRAW"]: break
        elif mode == 2: 
            if game.current_player == human_p:
                if get_human_move(game) in ["QUIT", "DRAW"]: break
            else:
                print(f"\n[IA] A calcular ({IT_HUMAN_VS_IA} it)...")
                move = ia_std.search(game)
        elif mode == 3: 
            active_ia = ia_p1 if game.current_player == PLAYER1 else ia_p2
            print(f"\n[IA {game.current_player}] A calcular ({active_ia.iterations} it)...")
            move = active_ia.search(game)

        if move:
            col, m_type = move
            if m_type == 'd': game.drop_piece(col, game.current_player)
            else: game.pop_piece(col, game.current_player)

        winner = game.check_winner_after_move(game.current_player)
        if winner:
            draw_board(game)
            vencedor_nome = "X (Vermelho)" if winner == PLAYER1 else "O (Amarelo)"
            print(f"\n*** VITÓRIA DO JOGADOR {vencedor_nome.upper()} ***")
            break
        
        game.current_player = PLAYER2 if game.current_player == PLAYER1 else PLAYER1

def main_menu(ia_std=None, ia_p1=None, ia_p2=None):
    while True:
        print("\n" + "="*30 + "\n      POPOUT AI SYSTEM 2026\n" + "="*30)
        print("1. Iniciar Humano vs Humano\n2. Desafiar IA (Humano vs MCTS)\n3. Modo Observador (IA vs IA Visual)\n4. Sair")
        op = input("\nEscolha uma opção: ").strip()
        if op == "1": play_game(1)
        elif op == "2": play_game(2, ia_std=ia_std)
        elif op == "3": play_game(3, ia_p1=ia_p1, ia_p2=ia_p2)
        elif op == "4": print("A encerrar o sistema..."); break
        else: print("[!] Opção inválida. Tente novamente.")
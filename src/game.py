import numpy as np
import copy
import pandas as pd
import random
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

    def drop_piece(self, col, piece, record=True):
        for r in range(ROWS-1, -1, -1):
            if self.board[r][col] == EMPTY:
                self.board[r][col] = piece 
                if record: self._record_state()
                return True
        return False 

    def pop_piece(self, col, piece, record=True):
        if self.board[ROWS-1][col] == piece:
            for r in range(ROWS-1, 0, -1):
                self.board[r][col] = self.board[r-1][col]
            self.board[0][col] = EMPTY
            if record: self._record_state()
            return True
        return False

    def get_winning_move(self, player):
        """Verifica se há alguma jogada de vitória imediata (1-Ply Lookahead)."""
        board_full = self.is_board_full()
        original_board = np.copy(self.board)
        
        for col in range(COLS):
            # Testar DROP
            if not board_full and self.board[0][col] == EMPTY:
                self.drop_piece(col, player, record=False)
                if self.check_win(player):
                    self.board = np.copy(original_board)
                    return (col, 'd')
                self.board = np.copy(original_board)
                
            # Testar POP
            if self.board[ROWS-1][col] == player:
                self.pop_piece(col, player, record=False)
                if self.check_win(player):
                    self.board = np.copy(original_board)
                    return (col, 'p')
                self.board = np.copy(original_board)
                
        return None

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
# CONTROLADORES DE INPUT E IA
# =================================================================

def get_human_move(game):
    p_name = "X" if game.current_player == PLAYER1 else "O"
    
    # Regra 3 do guião: Empate por Repetição
    if game.check_repetition():
        print(f"\n[AVISO] Este estado repetiu-se 3 vezes.")
        if input(f"Jogador {p_name}, aceita o empate? (s/n): ").lower() == 's': return "DRAW"

    # Regra 2 do guião: Empate por Tabuleiro Cheio
    if game.is_board_full():
        print(f"\n[AVISO] O tabuleiro está cheio.")
        print(f"Jogador {p_name}, o seu único movimento possível é um 'Pop'.")
        if input(f"Deseja declarar empate em vez de jogar? (s/n): ").lower() == 's': 
            return "DRAW"

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
            print("[!] Movimento inválido (coluna cheia ou peça errada na base).")
        except:
            print("[!] Erro de formato! Use: 'COLUNA TIPO' (ex: 3 d).")

def _get_ai_move(ia_obj, game):
    """Função auxiliar que extrai a jogada detetando se é Árvore ou MCTS, com segurança global."""
    p_current = game.current_player
    p_opponent = 2 if p_current == 1 else 1

    # --- VALIDAÇÃO DE SEGURANÇA GLOBAL ---
    # 1. Ataque letal imediato
    m_win = game.get_winning_move(p_current)
    if m_win is not None:
        return m_win

    # 2. Defesa imediata
    m_defend = game.get_winning_move(p_opponent)
    if m_defend is not None:
        # Se a ameaça for Drop e a coluna não estiver cheia, nós bloqueamos
        if m_defend[1] == 'd' and not game.is_board_full() and game.board[0][m_defend[0]] == EMPTY:
            return (m_defend[0], 'd')
    # -------------------------------------------------------------

    if hasattr(ia_obj, 'predict'): # Se for a Árvore de Decisão
        flat_board = game.board.flatten().tolist()
        df = pd.DataFrame([flat_board + [game.current_player]], columns=[f"pos_{i}" for i in range(42)] + ["p_turn"])
        previsao = ia_obj.predict(df)[0]
        
        col = int(previsao.split("_")[0])
        m_type = previsao.split("_")[1]
        
        # --- REALITY CHECK: O movimento da Árvore é legal? ---
        is_valid = False
        if m_type == 'd' and not game.is_board_full() and game.board[0][col] == EMPTY:
            is_valid = True
        elif m_type == 'p' and game.board[ROWS-1][col] == p_current:
            is_valid = True
            
        if is_valid:
            return (col, m_type)
        else:
            # A Árvore "alucinou" uma jogada impossível. 
            # Forçamos uma jogada aleatória legal para evitar o loop infinito.
            valid_moves = []
            for c in range(COLS):
                if not game.is_board_full() and game.board[0][c] == EMPTY:
                    valid_moves.append((c, 'd'))
                if game.board[ROWS-1][c] == p_current:
                    valid_moves.append((c, 'p'))
            
            if not valid_moves: return None # Prevenção extrema de crash
            return random.choice(valid_moves)

    else: # Se for o MCTS
        return ia_obj.search(game)

# =================================================================
# MOTOR PRINCIPAL DO JOGO
# =================================================================

def play_game(mode, ia_std=None, ia_p1=None, ia_p2=None, nome_p1="Humano (X)", nome_p2="Humano (O)"):
    game = PopOutGame()
    human_p = PLAYER1
    
    if mode == 2:
        print("\n--- Configuração de Partida ---")
        print("1. Ser Jogador 1 (X - Vermelho) [Começa]")
        print("2. Ser Jogador 2 (O - Amarelo) [IA começa]")
        if input("Escolha (1/2): ").strip() == "1":
            human_p = PLAYER1
            nome_p1 = "Humano (X)"
        else:
            human_p = PLAYER2
            nome_p2 = "Humano (O)"

    while True:
        draw_board(game)
        move = None

        if mode == 1: 
            human_action = get_human_move(game)
            if human_action == "QUIT": break
            if human_action == "DRAW":
                print("\n*** JOGO TERMINOU EMPATADO! ***")
                break
            
        elif mode == 2: 
            if game.current_player == human_p:
                human_action = get_human_move(game)
                if human_action == "QUIT": break
                if human_action == "DRAW":
                    print("\n*** JOGO TERMINOU EMPATADO! ***")
                    break
            else:
                nome_atual = nome_p1 if game.current_player == PLAYER1 else nome_p2
                print(f"\n[{nome_atual}] A calcular jogada...")
                move = _get_ai_move(ia_std, game)
                
        elif mode == 3: 
            active_ia = ia_p1 if game.current_player == PLAYER1 else ia_p2
            nome_atual = nome_p1 if game.current_player == PLAYER1 else nome_p2
            print(f"\n[{nome_atual}] A calcular jogada...")
            move = _get_ai_move(active_ia, game)

        if move:
            col, m_type = move
            if m_type == 'd': game.drop_piece(col, game.current_player)
            else: game.pop_piece(col, game.current_player)

        winner = game.check_winner_after_move(game.current_player)
        if winner:
            draw_board(game)
            # Aqui exibimos o nome específico da IA ou Humano que venceu
            vencedor_final = nome_p1 if winner == PLAYER1 else nome_p2
            print(f"\n*** VITÓRIA: {vencedor_final.upper()} VENCEU! ***")
            break
        
        game.current_player = PLAYER2 if game.current_player == PLAYER1 else PLAYER1

# =================================================================
# MENU PRINCIPAL (Configurado via Jupyter)
# =================================================================

def main_menu(mcts1=None, mcts2=None, tree1=None, tree2=None):
    """
    Menu dinâmico que recebe as IAs configuradas no Jupyter Notebook
    e organiza os duelos baseados nas escolhas do utilizador.
    """
    if mcts1 is None: mcts1 = MCTS(iterations=1000, c=1.41)
    if mcts2 is None: mcts2 = MCTS(iterations=10000, c=1.41)

    while True:
        print("\n" + "="*30 + "\n      POPOUT AI SYSTEM 2026\n" + "="*30)
        print("1. Iniciar Humano vs Humano")
        print("2. Desafiar IA")
        print("3. Modo Observador (IA vs IA Visual)")
        print("4. Sair")
        
        op = input("\nEscolha uma opção: ").strip()
        
        if op == "1": 
            play_game(1)
            
        elif op == "2": 
            print("\n--- Escolher IA para o Desafio ---")
            print("1. Monte Carlo (mcts1)")
            print("2. Árvore de Decisão (tree1)")
            escolha = input("Escolha (1/2): ").strip()
            
            if escolha == "1":
                play_game(2, ia_std=mcts1, nome_p2="mcts1")
            elif escolha == "2":
                if tree1 is None:
                    print("[!] Árvore não enviada! A usar mcts1.")
                    play_game(2, ia_std=mcts1, nome_p2="mcts1")
                else:
                    play_game(2, ia_std=tree1, nome_p2="tree1")
            else:
                print("[!] Opção inválida.")
                
        elif op == "3": 
            print("\n--- Configurar Duelo ---")
            print("1. MCTS vs MCTS (mcts1 x mcts2)")
            print("2. Árvore vs Árvore (tree1 x tree2)")
            print("3. MCTS vs Árvore (mcts1 x tree1)")
            print("4. Árvore vs MCTS (tree1 x mcts1)")
            escolha = input("Escolha o duelo: ").strip()
            
            if escolha == "1":
                play_game(3, ia_p1=mcts1, ia_p2=mcts2, nome_p1="mcts1", nome_p2="mcts2")
            elif escolha == "2":
                if tree1 is None or tree2 is None:
                    print("[!] Falta configurar tree1 ou tree2 no Jupyter!")
                else:
                    play_game(3, ia_p1=tree1, ia_p2=tree2, nome_p1="tree1", nome_p2="tree2")
            elif escolha == "3":
                if tree1 is None: print("[!] Árvore não foi enviada!")
                else: play_game(3, ia_p1=mcts1, ia_p2=tree1, nome_p1="mcts1", nome_p2="tree1")
            elif escolha == "4":
                if tree1 is None: print("[!] Árvore não foi enviada!")
                else: play_game(3, ia_p1=tree1, ia_p2=mcts1, nome_p1="tree1", nome_p2="mcts1")
            else:
                print("[!] Opção inválida.")
                
        elif op == "4": 
            print("A encerrar o sistema...")
            break
        else: 
            print("[!] Opção inválida. Tente novamente.")
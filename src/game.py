"""
Módulo Principal - PopOut Connect 4 (OTIMIZADO)
Contém o motor de regras do jogo, verificação de condições de vitória,
e a orquestração dos modos de jogo (Humano vs Humano, Humano vs IA, IA vs IA).
"""

import numpy as np
import pandas as pd
import random
from src.ui import draw_board
from src.mcts import MCTS

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1  
PLAYER2 = 2  

class PopOutGame:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.intp)
        self.current_player = PLAYER1
        self.state_history = {} 
        self._record_state()

    def get_state_key(self):
        return tuple(map(tuple, self.board))

    def _record_state(self):
        key = self.get_state_key()
        self.state_history[key] = self.state_history.get(key, 0) + 1

    # --- NOVOS MÉTODOS DE OTIMIZAÇÃO DE PERFORMANCE ---
    def clone(self):
        """
        Retorna uma cópia rápida do estado atual do jogo.
        Ota-se por não copiar o histórico de estados para otimizar memória e processamento nas simulações MCTS.
        """
        new_game = PopOutGame()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        return new_game

    def get_valid_moves(self):
        """Calcula movimentos válidos diretamente no estado."""
        moves = []
        board_full = self.is_board_full()
        for col in range(COLS):
            if not board_full and self.board[0][col] == EMPTY:
                moves.append((col, 'd'))
            if self.board[ROWS-1][col] == self.current_player:
                moves.append((col, 'p'))
        return moves

    # ----------------------------------------------------

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


def get_human_move(game):
    """
    Solicita a entrada do jogador humano e processa o formato da jogada.
    """
    p_name = "X" if game.current_player == PLAYER1 else "O"
    
    if game.check_repetition():
        print(f"\n[AVISO] Este estado repetiu-se 3 vezes.")
        if input(f"Jogador {p_name}, aceita o empate? (s/n): ").lower() == 's': return "DRAW"

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
    """
    Solicita a jogada da Inteligência Artificial (compatível com MCTS ou Árvore de Decisão).
    """
    p_current = game.current_player

    if hasattr(ia_obj, 'predict'):
        flat_board = game.board.flatten().tolist()
        normalized_board = []
        for piece in flat_board:
            if piece == EMPTY: normalized_board.append(0)
            elif piece == p_current: normalized_board.append(1)
            else: normalized_board.append(-1)
            
        df = pd.DataFrame([normalized_board], columns=[f"pos_{i}" for i in range(42)])
        previsao = ia_obj.predict(df)[0]
        
        col = int(previsao.split("_")[0])
        m_type = previsao.split("_")[1]
        
        is_valid = False
        if m_type == 'd' and not game.is_board_full() and game.board[0][col] == EMPTY:
            is_valid = True
        elif m_type == 'p' and game.board[ROWS-1][col] == p_current:
            is_valid = True
            
        if is_valid:
            return (col, m_type)
        else:
            valid_moves = game.get_valid_moves()
            if not valid_moves: return None
            return random.choice(valid_moves)
    else: 
        return ia_obj.search(game)

def play_game(mode, ia_std=None, ia_p1=None, ia_p2=None, nome_p1="Humano (X)", nome_p2="Humano (O)"):
    """
    Controla o ciclo principal de um jogo interativo (CLI).
    Modes: 1 (Humano vs Humano), 2 (Humano vs IA), 3 (IA vs IA Visual).
    """
    game = PopOutGame()
    human_p = PLAYER1
    num_moves = 0
    
    if mode == 2:
        nome_ia = nome_p2
        print("\n--- Configuração de Partida ---")
        print("1. Ser Jogador 1 (X - Vermelho) [Começa]")
        print("2. Ser Jogador 2 (O - Amarelo) [IA começa]")
        if input("Escolha (1/2): ").strip() == "1":
            human_p = PLAYER1
            nome_p1 = "Humano (X)"
            nome_p2 = f"{nome_ia} (O)"
        else:
            human_p = PLAYER2
            nome_p1 = f"{nome_ia} (X)"
            nome_p2 = "Humano (O)"

    while True:
        draw_board(game)
        
        if mode == 3 and (game.check_repetition() or num_moves > 150):
            print("\n*** JOGO TERMINOU EMPATADO (Repetição ou Limite de 150 turnos)! ***")
            break

        move = None
        ia_failed = False

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
                if move is None: ia_failed = True
                
        elif mode == 3: 
            active_ia = ia_p1 if game.current_player == PLAYER1 else ia_p2
            nome_atual = nome_p1 if game.current_player == PLAYER1 else nome_p2
            print(f"\n[{nome_atual}] A calcular jogada...")
            move = _get_ai_move(active_ia, game)
            if move is None: ia_failed = True

        if ia_failed:
            print("\n*** JOGO TERMINOU EMPATADO (A IA declarou impasse)! ***")
            break

        if move:
            col, m_type = move
            if m_type == 'd': game.drop_piece(col, game.current_player)
            else: game.pop_piece(col, game.current_player)

        num_moves += 1
        winner = game.check_winner_after_move(game.current_player)
        
        if winner:
            draw_board(game)
            vencedor_final = nome_p1 if winner == PLAYER1 else nome_p2
            print(f"\n*** VITÓRIA: {vencedor_final.upper()} VENCEU! ***")
            break
        
        game.current_player = PLAYER2 if game.current_player == PLAYER1 else PLAYER1

def _descrever_ia(nome, ia_obj):
    """Retorna uma string com a descrição/configuração de um agente IA."""
    if isinstance(ia_obj, MCTS):
        return (f"{nome}  [MCTS: iter={ia_obj.iterations}, c={ia_obj.c}, "
                f"children={ia_obj.max_children}, depth={ia_obj.max_depth}, pure={ia_obj.pure_mode}]")
    else:
        return f"{nome}  [Árvore de Decisão]"


def _escolher_ia(agentes, mensagem="Escolha uma IA"):
    """Mostra lista numerada de agentes e retorna (nome, objeto) escolhido."""
    nomes = list(agentes.keys())
    print(f"\n--- {mensagem} ---")
    for i, nome in enumerate(nomes, 1):
        print(f"  {i}. {_descrever_ia(nome, agentes[nome])}")

    while True:
        try:
            idx = int(input(f"Escolha (1-{len(nomes)}): ").strip()) - 1
            if 0 <= idx < len(nomes):
                escolhido = nomes[idx]
                print(f"  → Selecionado: {_descrever_ia(escolhido, agentes[escolhido])}")
                return escolhido, agentes[escolhido]
            print("[!] Número fora do intervalo.")
        except ValueError:
            print("[!] Introduza um número válido.")


def main_menu(mcts_configs=None, arvores=None,
              mcts1=None, mcts2=None, tree1=None, tree2=None):
    """
    Menu principal do sistema PopOut AI.
    Aceita dois formatos de argumentos:
      - Novo: mcts_configs (dict) e arvores (dict)
      - Legado: mcts1, mcts2, tree1, tree2 (para compatibilidade com o notebook)
    """
    # ── Construir dicionário unificado de agentes ────────────────────────
    agentes = {}

    if mcts_configs:
        agentes.update(mcts_configs)
    else:
        # Modo legado (compatibilidade com notebook)
        if mcts1 is None:
            mcts1 = MCTS(iterations=10000, c=1.41)
        if mcts2 is None:
            mcts2 = MCTS(iterations=10000, c=1.41)
        agentes["mcts1"] = mcts1
        agentes["mcts2"] = mcts2

    if arvores:
        agentes.update(arvores)
    else:
        # Modo legado
        if tree1 is not None:
            agentes["tree1"] = tree1
        if tree2 is not None:
            agentes["tree2"] = tree2

    # ── Loop do menu ─────────────────────────────────────────────────────
    while True:
        print("\n" + "="*40)
        print("        POPOUT AI SYSTEM 2026")
        print("="*40)
        print(f"  Agentes disponíveis: {len(agentes)}")
        for nome, ia in agentes.items():
            print(f"    • {_descrever_ia(nome, ia)}")
        print("-"*40)
        print("1. Iniciar Humano vs Humano")
        print("2. Desafiar IA")
        print("3. Modo Observador (IA vs IA Visual)")
        print("4. Sair")

        op = input("\nEscolha uma opção: ").strip()

        if op == "1":
            play_game(1)

        elif op == "2":
            nome, ia = _escolher_ia(agentes, "Escolha a IA adversária")
            play_game(2, ia_std=ia, nome_p2=nome)

        elif op == "3":
            print("\n=== Configurar Duelo IA vs IA ===")
            nome1, ia1 = _escolher_ia(agentes, "Escolha o Jogador 1 (X)")
            nome2, ia2 = _escolher_ia(agentes, "Escolha o Jogador 2 (O)")
            play_game(3, ia_p1=ia1, ia_p2=ia2, nome_p1=nome1, nome_p2=nome2)

        elif op == "4":
            print("A encerrar o sistema...")
            break
        else:
            print("[!] Opção inválida. Tente novamente.")

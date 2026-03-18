import numpy as np
import copy
import csv
import os
from src.ui import draw_board
from src.mcts import MCTS  # Importa o cérebro estratégico (Algoritmo Monte Carlo)

# =================================================================
# CONFIGURAÇÕES E CONSTANTES DO JOGO
# =================================================================
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1  # Representado visualmente por 'X' (Vermelho)
PLAYER2 = 2  # Representado visualmente por 'O' (Amarelo)

class PopOutGame:
    """
    Classe que gere o estado lógico e as regras do PopOut.
    Implementa as jogadas Drop/Pop e as 3 regras especiais do projeto.
    """
    def __init__(self):
        # Matriz NumPy 6x7 para representar o tabuleiro físico.
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = PLAYER1
        # Dicionário para rastrear a repetição de estados (Regra Especial 3).
        self.state_history = {} 
        self._record_state()

    def get_state_key(self):
        """Converte a matriz num tuplo imutável para ser usado como chave de dicionário."""
        return tuple(map(tuple, self.board))

    def _record_state(self):
        """Regista quantas vezes o tabuleiro atual já apareceu no jogo."""
        key = self.get_state_key()
        self.state_history[key] = self.state_history.get(key, 0) + 1

    def drop_piece(self, col, piece):
        """Executa a inserção de peça no topo. Retorna False se a coluna estiver cheia."""
        for r in range(ROWS-1, -1, -1):
            if self.board[r][col] == EMPTY:
                self.board[r][col] = piece 
                self._record_state()
                return True
        return False 

    def pop_piece(self, col, piece):
        """Remove peça da base (Regra PopOut). Todas as peças acima caem uma linha."""
        if self.board[ROWS-1][col] == piece:
            for r in range(ROWS-1, 0, -1):
                self.board[r][col] = self.board[r-1][col]
            self.board[0][col] = EMPTY
            self._record_state()
            return True
        return False

    def check_win(self, piece):
        """Verifica se existem 4 peças alinhadas (Horizontal, Vertical ou Diagonais)."""
        # Horizontal
        for c in range(COLS-3):
            for r in range(ROWS):
                if all(self.board[r][c+i] == piece for i in range(4)): return True
        # Vertical
        for c in range(COLS):
            for r in range(ROWS-3):
                if all(self.board[r+i][c] == piece for i in range(4)): return True
        # Diagonal Positiva (/)
        for c in range(COLS-3):
            for r in range(ROWS-3):
                if all(self.board[r+i][c+i] == piece for i in range(4)): return True
        # Diagonal Negativa (\)
        for c in range(COLS-3):
            for r in range(3, ROWS):
                if all(self.board[r-i][c+i] == piece for i in range(4)): return True
        return False

    def check_winner_after_move(self, player_who_moved):
        """Regra Especial 1: Se um 'Pop' der vitória aos dois, quem jogou vence."""
        p1 = self.check_win(PLAYER1)
        p2 = self.check_win(PLAYER2)
        if p1 and p2: return player_who_moved 
        if p1: return PLAYER1
        if p2: return PLAYER2
        return None

    def is_board_full(self):
        """Regra Especial 2: Deteta se o topo do tabuleiro está totalmente preenchido."""
        return all(self.board[0][c] != EMPTY for c in range(COLS))

    def check_repetition(self):
        """Verifica se o estado atual já ocorreu 3 vezes (Regra 3)."""
        return self.state_history.get(self.get_state_key(), 0) >= 3

    def clone(self):
        """Cria uma cópia profunda para as simulações do MCTS."""
        return copy.deepcopy(self)

# =================================================================
# GERAÇÃO AUTOMÁTICA DE DATASETS (BATCHES)
# =================================================================

def save_move_to_dataset(game_state, move, filename):
    """Grava o tabuleiro e a decisão da IA num ficheiro CSV de lote."""
    flat_board = game_state.board.flatten().tolist()
    # Estrutura: [42 posições] + [Jogador] + [Coluna] + [Tipo d/p]
    row = flat_board + [game_state.current_player, move[0], move[1]]
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = [f"pos_{i}" for i in range(42)] + ["p_turn", "col", "type"]
            writer.writerow(headers)
        writer.writerow(row)

def run_batch_simulation(num_games, ia_1, ia_2, batch_label):
    """Executa jogos PC vs PC em segundo plano para encher o dataset."""
    filename = f"batch_{batch_label}_P1_it{ia_1.iterations}_vs_P2_it{ia_2.iterations}.csv"
    print(f"\n[SISTEMA] A gerar lote: {batch_label}...")
    for i in range(num_games):
        game = PopOutGame()
        while True:
            curr_ia = ia_1 if game.current_player == PLAYER1 else ia_2
            move = curr_ia.search(game)
            if move is None: break 
            save_move_to_dataset(game, move, filename)
            if move[1] == 'd': game.drop_piece(move[0], game.current_player)
            else: game.pop_piece(move[0], game.current_player)
            if game.check_winner_after_move(game.current_player): break
            game.current_player = 2 if game.current_player == 1 else 1
        print(f"> Jogo {i+1}/{num_games} guardado.")

# =================================================================
# CONTROLADOR DE JOGO E INTERFACE DE INPUT
# =================================================================

def get_human_move(game):
    """Lida com o input do utilizador, validações e regras de empate."""
    p_name = "X" if game.current_player == PLAYER1 else "O"
    
    # Tratamento da Regra 3 (Repetição)
    if game.check_repetition():
        print(f"\n[AVISO] Este estado repetiu-se 3 vezes.")
        if input(f"Jogador {p_name}, aceita o empate? (s/n): ").lower() == 's': return "DRAW"

    while True:
        try:
            cmd = input(f"[{p_name}] Jogada (ex: 3 d) ou 'q': ").lower().strip()
            if cmd in ['q', 'sair']: return "QUIT"
            
            parts = cmd.split()
            col, m_type = int(parts[0]), parts[1]

            # Validação: Se tabuleiro cheio, Drop é proibido (Regra 2)
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

def play_game(mode):
    """
    Controlador principal dos modos de jogo.
    Edita os níveis e comportamentos da IA diretamente nas variáveis abaixo.
    """
    game = PopOutGame()

    # ---------------------------------------------------------
    # PAINEL DE CONFIGURAÇÃO MANUAL (MODIFICA TUDO AQUI)
    # ---------------------------------------------------------
    
    # --- MODO 2: HUMANO VS IA ---
    # Configuração para o desafio contra o utilizador
    IT_HUMAN_VS_IA = 1500    # Quantidade de simulações
    C_HUMAN_VS_IA  = 1.41    # Equilíbrio entre exploração e vitória
    MAX_C_HUMAN_IA = None    # Limite de colunas a explorar (None = 7)

    # --- MODO 3: IA VS IA (OBSERVADOR VISUAL) ---
    # Jogador 1 (X) - Ex: IA muito forte
    IT_P1 = 5000            
    C_P1  = 1.41            
    MAX_C_P1 = None         

    # Jogador 2 (O) - Ex: IA God Mode com restrição de largura
    IT_P2 = 10000           
    C_P2  = 1.1             # Mais focada em vitórias conhecidas
    MAX_C_P2 = 3            # Testa apenas 3 colunas por nó (Requisito do Guião)
    
    # ---------------------------------------------------------

    # Inicialização dos motores MCTS com os parâmetros definidos acima (Corrigido)
    ia_std = MCTS(iterations=IT_HUMAN_VS_IA, c=C_HUMAN_VS_IA, max_children=MAX_C_HUMAN_IA)
    
    ia_vs_ia_p1 = MCTS(iterations=IT_P1, c=C_P1, max_children=MAX_C_P1)
    ia_vs_ia_p2 = MCTS(iterations=IT_P2, c=C_P2, max_children=MAX_C_P2)

    # Lógica de escolha de cor para o Modo Humano vs IA
    human_p = PLAYER1
    if mode == 2:
        print("\n--- Configuração de Partida ---")
        print("1. Ser Jogador 1 (X - Vermelho) [Começa]")
        print("2. Ser Jogador 2 (O - Amarelo) [IA começa]")
        human_p = PLAYER1 if input("Escolha (1/2): ").strip() == "1" else PLAYER2

    # Ciclo Principal de Jogo
    while True:
        draw_board(game)
        move = None

        if mode == 1: # Humano vs Humano
            if get_human_move(game) in ["QUIT", "DRAW"]: break
        
        elif mode == 2: # Humano vs IA
            if game.current_player == human_p:
                if get_human_move(game) in ["QUIT", "DRAW"]: break
            else:
                print(f"\n[IA] A calcular ({IT_HUMAN_VS_IA} it)...")
                move = ia_std.search(game)
        
        elif mode == 3: # IA vs IA (Observador)
            # Escolhe a IA configurada para o jogador atual
            active_ia = ia_vs_ia_p1 if game.current_player == PLAYER1 else ia_vs_ia_p2
            print(f"\n[IA {game.current_player}] A calcular ({active_ia.iterations} it)...")
            move = active_ia.search(game)

        # Execução da jogada decidida (seja pela IA ou Humano no status)
        if move:
            col, m_type = move
            if m_type == 'd': 
                game.drop_piece(col, game.current_player)
            else: 
                game.pop_piece(col, game.current_player)

        # Verificação de Vitória (Regra Especial 1 aplicada internamente)
        winner = game.check_winner_after_move(game.current_player)
        if winner:
            draw_board(game)
            vencedor_nome = "X (Vermelho)" if winner == PLAYER1 else "O (Amarelo)"
            print(f"\n*** VITÓRIA DO JOGADOR {vencedor_nome.upper()} ***")
            break
        
        # Troca de Turno
        game.current_player = PLAYER2 if game.current_player == PLAYER1 else PLAYER1

def main_menu():
    """Interface inicial do programa via terminal."""
    while True:
        print("\n" + "="*30)
        print("      POPOUT AI SYSTEM 2026")
        print("="*30)
        print("1. Iniciar Humano vs Humano")
        print("2. Desafiar IA (Humano vs MCTS)")
        print("3. Modo Observador (IA vs IA Visual)")
        print("4. Sair")
        
        op = input("\nEscolha uma opção: ").strip()
        
        if op == "1":
            play_game(1)
        elif op == "2":
            play_game(2)
        elif op == "3":
            play_game(3)
        elif op == "4":
            print("A encerrar o sistema...")
            break
        else:
            print("[!] Opção inválida. Tente novamente.")
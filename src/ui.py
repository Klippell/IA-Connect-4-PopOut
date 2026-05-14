"""
Módulo de renderização da interface do jogo no terminal (PopOut Connect 4).
Gere as cores e o desenho da grelha de tabuleiro.
"""

import os

COLOR_P1 = '\033[91m'
COLOR_P2 = '\033[93m'
COLOR_BOARD = '\033[94m'
RESET_COLOR = '\033[0m'

def clear_screen():
    """Limpa o ecrã do terminal de acordo com o sistema operativo atual."""
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_board(game):
    """
    Renderiza o estado atual do tabuleiro de jogo no terminal.
    
    Parâmetros:
        game (Game): Instância do objeto de jogo, que contém a matriz 'board'.
    """
    print("\n" + " " * 8 + "=== PopOut AI ===")
    print("   0   1   2   3   4   5   6")
    print(COLOR_BOARD + " ╔═══╦═══╦═══╦═══╦═══╦═══╦═══╗" + RESET_COLOR)
    
    for r in range(6):
        row_cells = []
        for c in range(7):
            piece = game.board[r][c]
            if piece == 0:
                char = "   "
            elif piece == 1:
                char = f" {COLOR_P1}X{RESET_COLOR} "
            else:
                char = f" {COLOR_P2}O{RESET_COLOR} "
            row_cells.append(char)
            
        row_str = (COLOR_BOARD + " ║" + RESET_COLOR + 
                   (COLOR_BOARD + "║" + RESET_COLOR).join(row_cells) + 
                   COLOR_BOARD + "║" + RESET_COLOR)
        print(row_str)
        
        if r < 5:
            print(COLOR_BOARD + " ╠═══╬═══╬═══╬═══╬═══╬═══╬═══╣" + RESET_COLOR)
            
    print(COLOR_BOARD + " ╚═══╩═══╩═══╩═══╩═══╩═══╩═══╝" + RESET_COLOR)
    print("   ^   ^   ^   ^   ^   ^   ^")
    print("   P   O   P   M   O   V   E   S\n")
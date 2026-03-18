import os

# Definição de constantes para as cores no terminal usando códigos ANSI.
# Estas variáveis permitem colorir o texto para melhor distinção visual,
# mantendo o aspeto profissional sem recorrer a bibliotecas complexas.
COLOR_P1 = '\033[91m'    # Código ANSI para a cor Vermelha (Jogador 1 - X)
COLOR_P2 = '\033[93m'    # Código ANSI para a cor Amarela (Jogador 2 - O)
COLOR_BOARD = '\033[94m' # Código ANSI para a cor Azul (Limites da grelha do tabuleiro)
RESET_COLOR = '\033[0m'  # Código ANSI para repor a cor padrão do terminal

def clear_screen():
    """
    Limpa o conteúdo anterior do terminal.
    Verifica o sistema operativo ('nt' para Windows, caso contrário assume Unix/Linux/Mac)
    e executa o comando apropriado ('cls' ou 'clear').
    Isto mantém o tabuleiro sempre na mesma posição do ecrã, evitando que o histórico
    role infinitamente para baixo.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_board(game):
    """
    Renderiza o estado atual do tabuleiro no terminal.
    Utiliza caracteres Unicode de desenho de caixas para criar uma grelha fechada
    e substitui os valores numéricos pelas letras 'X' e 'O'.
    Recebe o objeto 'game', que contém a matriz do tabuleiro atual.
    """
    clear_screen()
    print("\n" + " " * 8 + "=== PopOut AI ===")
    
    # Imprime os índices das colunas perfeitamente alinhados com o centro de cada célula,
    # para guiar o utilizador na escolha da jogada.
    print("   0   1   2   3   4   5   6")
    
    # Desenha o topo da grelha do tabuleiro
    print(COLOR_BOARD + " ╔═══╦═══╦═══╦═══╦═══╦═══╦═══╗" + RESET_COLOR)
    
    # Itera sobre cada linha da matriz (de cima para baixo, total de 6 linhas).
    for r in range(6):
        # Inicia a string da linha com a borda lateral esquerda
        row_str = COLOR_BOARD + " ║" + RESET_COLOR
        
        # Itera sobre cada coluna da matriz (da esquerda para a direita, total de 7 colunas).
        for c in range(7):
            piece = game.board[r][c]
            
            # Substitui os valores numéricos por 'X' ou 'O' coloridos.
            if piece == 0:
                char = "   " # Espaço vazio para melhorar a leitura
            elif piece == 1:
                char = f" {COLOR_P1}X{RESET_COLOR} " # Peça do Jogador 1
            else:
                char = f" {COLOR_P2}O{RESET_COLOR} " # Peça do Jogador 2
                
            # Adiciona a peça e o separador vertical da grelha à string da linha
            row_str += char + COLOR_BOARD + "║" + RESET_COLOR
            
        # Imprime a linha completa
        print(row_str)
        
        # Desenha as linhas intermédias da grelha, garantindo que não desenha
        # na última linha (pois essa terá o fecho da base).
        if r < 5:
            print(COLOR_BOARD + " ╠═══╬═══╬═══╬═══╬═══╬═══╬═══╣" + RESET_COLOR)
            
    # Desenha a base da grelha do tabuleiro
    print(COLOR_BOARD + " ╚═══╩═══╩═══╩═══╩═══╩═══╩═══╝" + RESET_COLOR)
    
    # Imprime os indicadores visuais na base para lembrar que as jogadas 'Pop' (remover da base)
    # são permitidas nas colunas correspondentes.
    print("   ^   ^   ^   ^   ^   ^   ^")
    print("   P   O   P   M   O   V   E   S\n")
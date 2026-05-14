"""
Módulo de Geração de Datasets (Simulação Automática)

Permite executar simulações automáticas de IA vs IA (MCTS) e registar 
o histórico de cada estado de jogo, jogada e vencedor num ficheiro CSV, 
gerando dados para treino da Árvore de Decisão.
"""

import os
import csv
import uuid
from src.game import PopOutGame, PLAYER1

def save_game_to_dataset(game_history, winner, game_id, filename):
    """
    Grava os movimentos e o resultado de um jogo num ficheiro CSV de forma sequencial (append).
    
    Se o ficheiro não existir, gera os cabeçalhos automaticamente.
    
    Args:
        game_history (list): Lista com o histórico de jogadas e estados.
        winner (int): ID do jogador vencedor (ou 0 para empate).
        game_id (str): Identificador único da partida.
        filename (str): Caminho do ficheiro CSV de destino.
    """
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            headers = ["game_id"] + [f"pos_{i}" for i in range(42)] + ["p_turn", "col", "type", "winner"]
            writer.writerow(headers)
        
        for row in game_history:
            final_row = [game_id] + row + [winner]
            writer.writerow(final_row)

def run_batch_simulation(num_games, ia_1, ia_2):
    """
    Executa um lote de simulações automáticas de jogos entre duas IAs.
    
    Args:
        num_games (int): Número de jogos a simular.
        ia_1 (MCTS): Agente IA correspondente ao jogador 1.
        ia_2 (MCTS): Agente IA correspondente ao jogador 2.
    """
    os.makedirs("datasets", exist_ok=True)
    
    mc1 = ia_1.max_children if getattr(ia_1, 'max_children', None) is not None else "All"
    mc2 = ia_2.max_children if getattr(ia_2, 'max_children', None) is not None else "All"
    
    d1 = getattr(ia_1, 'max_depth', "None")
    d2 = getattr(ia_2, 'max_depth', "None")
    
    pm1 = "pure" if getattr(ia_1, 'pure_mode', False) else "opt"
    pm2 = "pure" if getattr(ia_2, 'pure_mode', False) else "opt"
    
    config_p1 = f"P1_it{ia_1.iterations}_c{ia_1.c}_mc{mc1}_d{d1}_{pm1}"
    config_p2 = f"P2_it{ia_2.iterations}_c{ia_2.c}_mc{mc2}_d{d2}_{pm2}"
    filename = f"datasets/{config_p1}_vs_{config_p2}.csv"
    
    print(f"\n[INFO] A gerar lote de {num_games} jogos...")
    print(f"[INFO] Ficheiro de destino: {filename}")
    
    for i in range(num_games):
        game = PopOutGame()
        game_history = [] 
        winner = 0 
        num_moves = 0
        
        current_game_id = uuid.uuid4().hex[:8]
        
        while True:
            if game.check_repetition() or num_moves > 150:
                winner = 0
                break
                
            curr_ia = ia_1 if game.current_player == PLAYER1 else ia_2
            move = curr_ia.search(game)
            
            if move is None: 
                break 
            
            flat_board = game.board.flatten().tolist()
            row = flat_board + [game.current_player, move[0], move[1]]
            game_history.append(row)
            
            if move[1] == 'd': 
                game.drop_piece(move[0], game.current_player)
            else: 
                game.pop_piece(move[0], game.current_player)
                
            num_moves += 1
            
            vencedor_atual = game.check_winner_after_move(game.current_player)
            if vencedor_atual:
                winner = vencedor_atual
                break
            
            game.current_player = 2 if game.current_player == 1 else 1
            
        save_game_to_dataset(game_history, winner, current_game_id, filename)
        
        status_vencedor = "Empate" if winner == 0 else f"P{winner}"
        print(f"> Jogo {i + 1}/{num_games} guardado (ID: {current_game_id} | Vencedor: {status_vencedor})")

    print(f"[INFO] Lote concluído. Ficheiros guardados em {filename}.")
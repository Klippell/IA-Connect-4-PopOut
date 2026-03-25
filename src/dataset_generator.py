import os
import csv
import uuid
from src.game import PopOutGame, PLAYER1

def save_game_to_dataset(game_history, winner, game_id, filename):
    """Grava todos os movimentos de um jogo no CSV com Vencedor e ID único."""
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
    """Executa jogos PC vs PC, guarda em memória e grava no fim do jogo."""
    os.makedirs("datasets", exist_ok=True)
    
    mc1 = ia_1.max_children if ia_1.max_children is not None else 7
    mc2 = ia_2.max_children if ia_2.max_children is not None else 7
    config_p1 = f"P1_it{ia_1.iterations}_c{ia_1.c}_mc{mc1}"
    config_p2 = f"P2_it{ia_2.iterations}_c{ia_2.c}_mc{mc2}"
    filename = f"datasets/{config_p1}_vs_{config_p2}.csv"
    
    print(f"\n[SISTEMA] A gerar lote de {num_games} jogos...")
    print(f"[ARQUIVO] {filename}")
    
    for i in range(num_games):
        game = PopOutGame()
        game_history = [] 
        winner = 0 
        
        # Gera um ID único e seguro para este jogo (ex: 'f8a2b3c1')
        current_game_id = uuid.uuid4().hex[:8]
        
        while True:
            curr_ia = ia_1 if game.current_player == PLAYER1 else ia_2
            move = curr_ia.search(game)
            if move is None: break 
            
            flat_board = game.board.flatten().tolist()
            row = flat_board + [game.current_player, move[0], move[1]]
            game_history.append(row)
            
            if move[1] == 'd': game.drop_piece(move[0], game.current_player)
            else: game.pop_piece(move[0], game.current_player)
            
            vencedor_atual = game.check_winner_after_move(game.current_player)
            if vencedor_atual:
                winner = vencedor_atual
                break
            
            game.current_player = 2 if game.current_player == 1 else 1
            
        # Guarda usando o UUID único
        save_game_to_dataset(game_history, winner, current_game_id, filename)
        
        # Print humano para saberes em que jogo a simulação vai
        status_vencedor = "Empate" if winner == 0 else f"P{winner}"
        print(f"> Jogo {i + 1}/{num_games} guardado (ID: {current_game_id} | Vencedor: {status_vencedor})")
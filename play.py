import time
import os
import pickle
from src.mcts import MCTS
from src.game import main_menu

MODELOS_DIR = "modelos_treinados"


def carregar_todas_arvores():
    """Carrega automaticamente todas as árvores .pkl da pasta modelos_treinados."""
    arvores = {}
    if not os.path.isdir(MODELOS_DIR):
        print(f"[ERRO] Pasta '{MODELOS_DIR}' não encontrada!")
        return arvores

    for ficheiro in sorted(os.listdir(MODELOS_DIR)):
        if ficheiro.endswith(".pkl"):
            nome = ficheiro.replace(".pkl", "")
            caminho = os.path.join(MODELOS_DIR, ficheiro)
            try:
                with open(caminho, 'rb') as f:
                    arvores[nome] = pickle.load(f)
                print(f"  [OK] {nome}", flush=True)
            except Exception as e:
                print(f"  [ERRO] Falha ao carregar {nome}: {e}", flush=True)

    return arvores


if __name__ == "__main__":
    print("A carregar modelos de IA do disco...", flush=True)
    arvores = carregar_todas_arvores()

    if not arvores:
        print("\n[!] Nenhuma árvore encontrada! Apenas MCTS estará disponível.")

    mcts1 = MCTS(iterations=10000, c=1.41, max_children=None, max_depth=None, pure_mode=False)
    mcts2 = MCTS(iterations=10000, c=1.41, max_children=None, max_depth=None, pure_mode=True)

    mcts_configs = {
        "mcts1": mcts1,
        "mcts2": mcts2,
    }

    print(f"\n[INFO] {len(arvores)} árvore(s) e {len(mcts_configs)} MCTS carregados com sucesso!", flush=True)
    print("[INFO] A iniciar o sistema...", flush=True)
    time.sleep(2)

    main_menu(mcts_configs=mcts_configs, arvores=arvores)
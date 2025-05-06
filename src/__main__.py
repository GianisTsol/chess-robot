from . import move_detector

from . import chess_engine

import threading
if __name__ == "__main__":
    detector = move_detector.ChessboardDetector(camera_index=2)

    manager = chess_engine.ChessGameManager(stockfish_path="/home/gtsol/Downloads/stockfish/stockfish-ubuntu-x86-64-avx2")  # adjust path

    t1 = threading.Thread(target=detector.run())
    t1.start()
    try:
        while True:
            board_array = get_board_array()  # Your OpenCV output
            result = manager.update_board_from_array(board_array)

            if result:
                player_move, ai_move = result
                print(f"Player moved: {player_move}, Stockfish responded with: {ai_move}")
                manager.print_board()
    except KeyboardInterrupt:
        print("EXITING")
        t1.join()
    except Exception as e:
        raise e

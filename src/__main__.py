import cv2
from . import move_detector
from . import chess_engine
import time
if __name__ == "__main__":
    detector = move_detector.ChessboardDetector(camera_index=1)
    detector.start()
    manager = chess_engine.ChessGameManager(stockfish_path="G:\My Drive\CODING\KORNHLIOS\stockfish\stockfish-windows-x86-64-avx2.exe")  # adjust path

    try:
        while True:
            time.sleep(0.1)
            if detector.consume():
                board_array = detector.board_state

                result = manager.update_board_from_array(board_array)

                if result:
                    player_move, ai_move = result
                    print(f"Player moved: {player_move}, Stockfish responded with: {ai_move}")
                    manager.print_board()
                else:
                    print("no move!")

                detector.update_game_state(manager.get_bool_array())

    except KeyboardInterrupt:
        print("EXITING")

        detector.stop()
    except Exception as e:
        raise e

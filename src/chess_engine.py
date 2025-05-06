import os
import cv2
import numpy as np
import chess
import chess.engine
from move_detector import square_has_piece

# === Settings ===
square_len_px = 50

# Βάλε εδώ την πλήρη διαδρομή προς το Stockfish .exe
STOCKFISH_PATH = r"C:\Users\micn\AppData\Local\Programs\Python\Python313\Lib\site-packages\stockfish"

if not os.path.exists(STOCKFISH_PATH):
    raise FileNotFoundError(f"Δεν βρέθηκε το Stockfish στο: {STOCKFISH_PATH}")

# === Αρχική διάταξη πιονιών ===
starting_board = [
    ["b_rook", "b_knight", "b_bishop", "b_queen", "b_king", "b_bishop", "b_knight", "b_rook"],
    ["b_pawn"] * 8,
    ["empty"] * 8,
    ["empty"] * 8,
    ["empty"] * 8,
    ["empty"] * 8,
    ["w_pawn"] * 8,
    ["w_rook", "w_knight", "w_bishop", "w_queen", "w_king", "w_bishop", "w_knight", "w_rook"]
]

def initialize_board(gray_top, square_size):
    board_state = [["empty" for _ in range(8)] for _ in range(8)]
    for row in range(8):
        for col in range(8):
            x, y = col * square_size, row * square_size
            if square_has_piece(gray_top, (x, y, square_size, square_size)):
                board_state[row][col] = starting_board[row][col]
    return board_state

def board_state_to_fen(board_state):
    piece_map = {
        "w_pawn": "P", "w_rook": "R", "w_knight": "N", "w_bishop": "B", "w_queen": "Q", "w_king": "K",
        "b_pawn": "p", "b_rook": "r", "b_knight": "n", "b_bishop": "b", "b_queen": "q", "b_king": "k"
    }
    fen_rows = []
    for row in board_state:
        fen_row = ""
        empty_count = 0
        for piece in row:
            if piece == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece_map.get(piece, "?")
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    fen = "/".join(fen_rows) + " w KQkq - 0 1"
    return fen

def draw_board_state(image, board_state, square_size):
    for row in range(8):
        for col in range(8):
            piece = board_state[row][col]
            if piece != "empty":
                x, y = col * square_size, row * square_size
                center = (x + square_size // 2, y + square_size // 2)
                cv2.putText(image, piece, center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    return image

if __name__ == "__main__":
    # Διάβασε εικόνα (π.χ. από warpPerspective)
    frame = cv2.imread("top_down_chessboard.jpg")
    if frame is None:
        raise FileNotFoundError("Δεν βρέθηκε η εικόνα 'top_down_chessboard.jpg'")
    
    gray_top = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Εντόπισε πιονάκια
    board_state = initialize_board(gray_top, square_len_px)

    # Σχεδίασε αποτελέσματα
    output = draw_board_state(frame.copy(), board_state, square_len_px)
    cv2.imshow("Detected Pieces", output)

    # Μετατροπή σε FEN
    fen = board_state_to_fen(board_state)
    print("FEN:", fen)

    # Φόρτωσε σκακιέρα και μηχανή
    board = chess.Board(fen)
    print("Σκακιέρα:\n", board)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    result = engine.play(board, chess.engine.Limit(time=1))
    print("Η AI παίζει:", result.move)

    board.push(result.move)
    print("Μετά την κίνηση:\n", board)

    engine.quit()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

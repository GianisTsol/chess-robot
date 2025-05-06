from move_detector import square_has_piece
import cv2
import numpy as np

# Board and square size setup
square_len_px = 50
chessboard_size = (8, 8)

# Standard chess layout from black's perspective
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

# Assign pieces to positions by board layout
def initialize_board(gray_top, square_size):
    board_state = [["empty" for _ in range(8)] for _ in range(8)]
    for row in range(8):
        for col in range(8):
            x, y = col * square_size, row * square_size
            if square_has_piece(gray_top, (x, y, square_size, square_size)):
                board_state[row][col] = starting_board[row][col]
    return board_state

# Visualization helper
def draw_board_state(image, board_state, square_size):
    for row in range(8):
        for col in range(8):
            piece = board_state[row][col]
            if piece != "empty":
                x, y = col * square_size, row * square_size
                center = (x + square_size // 2, y + square_size // 2)
                cv2.putText(image, piece, center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    return image

# Example usage with a warped top-down board
if __name__ == "__main__":
    # Replace with real image or frame
    frame = cv2.imread("top_down_chessboard.jpg")  # Already warped 8x8 top-down image
    gray_top = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    board_state = initialize_board(gray_top, square_len_px)

    # Display result
    output = draw_board_state(frame.copy(), board_state, square_len_px)
    cv2.imshow("Initial Board State", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

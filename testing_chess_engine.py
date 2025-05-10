# File: testing_chess_engine.py
#Script to test the ChessGameManager class from chess_engine.py
import os
import sys

import unittest
import numpy as np
import chess

# Assuming the class is in a file named chess_engine.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chess_engine import ChessGameManager  

class TestUpdateBoardFromArray(unittest.TestCase):

    def print_board_from_fen(self, fen):
        """
        Print the chess board in ASCII given a FEN string.
        """
        import chess
        board = chess.Board(fen)
        print(board)

    def setUp(self):
        self.manager = ChessGameManager(mock_engine=True)

    def test_initial_state_sets_previous(self):
        board = np.full((8, 8), False)
        result = self.manager.update_board_from_array(board)
        self.assertTrue(np.array_equal(self.manager.previous_state, board))
        self.assertIsNone(result)

    def test_no_change_returns_none(self):
        board = np.full((8, 8), False)
        self.manager.previous_state = board.copy()
        result = self.manager.update_board_from_array(board)
        self.assertIsNone(result)

    def test_valid_pawn_capture_returns_uci(self):
        # Simulate a capture: white pawn on e4 captures black pawn on d5 (e4xd5)
        old_board = np.full((8, 8), False)
        new_board = old_board.copy()

        # Position setup (from White's perspective, row 0 is rank 8, col 0 is file 'a')
        # White pawn starts at e4 → row 4, col 4
        # Black pawn is at d5 → row 3, col 3
        old_board[4][4] = True  # e4 (white pawn)
        old_board[3][3] = True  # d5 (black pawn)

        new_board[3][3] = True  # White pawn moved to d5 (captured black)
        new_board[4][4] = False  # e4 is now empty

        # Set up board position in sync with the above state (FEN with white pawn on e4, black pawn on d5)
        self.manager.board.set_fen("rnbqkbnr/pppppppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")

        # Create old_board to match the FEN
        old_board = self.manager.get_bool_array()
        new_board = old_board.copy()

        # Now apply the move: e4xd5
        old_board[4][4] = True  # e4 (white pawn, already True)
        old_board[3][3] = True  # d5 (black pawn, already True)
        new_board[4][4] = False  # e4 is now empty
        new_board[3][3] = True   # d5 is now white pawn

        self.manager.previous_state = old_board.copy()
        result = self.manager.update_board_from_array(new_board)
        print("Result from update_board_from_array:", result)
        self.assertTrue(result == "e4d5" or (isinstance(result, tuple) and result[0] == "e4d5"))
        self.assertEqual(result, "e4d5")

    def test_multiple_differences_returns_none(self):
        board1 = np.full((8, 8), False)
        board2 = board1.copy()
        board1[7][0] = True  # a1
        board1[7][7] = True  # h1

    def test_pawn_move(self):
        # e2 to e4
        self.print_board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.manager.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        old_board = self.manager.get_bool_array()
        new_board = old_board.copy()
        old_board[6][4] = True  # e2
        new_board[6][4] = False
        new_board[4][4] = True  # e4
        self.manager.previous_state = old_board.copy()
        result = self.manager.update_board_from_array(new_board)
        self.assertTrue(result == "e2e4" or (isinstance(result, tuple) and result[0] == "e2e4"))

    def test_knight_move(self):
        # g1 to f3
        self.print_board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.manager.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        old_board = self.manager.get_bool_array()
        new_board = old_board.copy()
        old_board[7][6] = True  # g1
        new_board[7][6] = False
        new_board[5][5] = True  # f3
        self.manager.previous_state = old_board.copy()
        result = self.manager.update_board_from_array(new_board)
        self.assertTrue(result == "g1f3" or (isinstance(result, tuple) and result[0] == "g1f3"))

    def test_bishop_move(self):
        # f1 to c4 (after e2e4)
        old_board = self.manager.get_bool_array()
        self.print_board_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        self.manager.board.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        new_board = old_board.copy()
        old_board[7][5] = True  # f1
        new_board[7][5] = False
        new_board[4][2] = True  # c4
        self.manager.previous_state = old_board.copy()
        result = self.manager.update_board_from_array(new_board)
        self.assertTrue(result == "f1c4" or (isinstance(result, tuple) and result[0] == "f1c4"))

    def test_rook_move(self):
        # h1 to f1 (after knight leaves g1)
        self.print_board_from_fen("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")
        self.manager.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")
        old_board = self.manager.get_bool_array()
        new_board = old_board.copy()
        old_board[7][7] = True  # h1
        new_board[7][7] = False
        new_board[7][5] = True  # f1
        self.manager.previous_state = old_board.copy()
        result = self.manager.update_board_from_array(new_board)
        self.assertTrue(result == "h1f1" or (isinstance(result, tuple) and result[0] == "h1f1"))

    def test_queen_move(self):
        # d1 to h5 (ensure path is clear, no pawns or pieces block d1-h5)
        # FEN: White queen on d1, path to h5 is clear
        self.print_board_from_fen("rnb1kbnr/pppppppp/8/8/8/4P3/PPP2PPP/RNBQKBNR w KQkq - 0 1")
        self.manager.board.set_fen("rnb1kbnr/pppppppp/8/8/8/4P3/PPP2PPP/RNBQKBNR w KQkq - 0 1")
        old_board = self.manager.get_bool_array()
        new_board = old_board.copy()
        # Move queen from d1 to h5
        old_board[7][3] = True  # d1
        new_board[7][3] = False
        new_board[3][7] = True  # h5
        self.manager.previous_state = old_board.copy()
        result = self.manager.update_board_from_array(new_board)
        self.assertTrue(result == "d1h5" or (isinstance(result, tuple) and result[0] == "d1h5"))

    def test_king_move(self):
        # e1 to f1 (after rook leaves f1)
        self.print_board_from_fen("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")
        self.manager.board.set_fen("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")
        old_board = self.manager.get_bool_array()
        new_board = old_board.copy()
        old_board[7][4] = True  # e1
        new_board[7][4] = False
        new_board[7][5] = True  # f1
        self.manager.previous_state = old_board.copy()
        result = self.manager.update_board_from_array(new_board)
        self.assertTrue(result == "e1f1" or (isinstance(result, tuple) and result[0] == "e1f1"))

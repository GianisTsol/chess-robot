import unittest
import numpy as np
import chess
from chess_engine import ChessGameManager  # Assuming chess_engine.py is in the same directory

class TestChessGameManager(unittest.TestCase):
    def setUp(self):
        # Initialize the ChessGameManager with a mock Stockfish path
        self.manager = ChessGameManager(stockfish_path="mock_stockfish")

    def test_no_changes(self):
        """Test when the board state does not change."""
        initial_state = self.manager.get_bool_array()
        result = self.manager.update_board_from_array(initial_state)
        self.assertIsNone(result, "No changes should return None")

    def test_valid_move(self):
        """Test a valid move from one square to another."""
        initial_state = self.manager.get_bool_array()
        # Simulate a move: e2 to e4
        current_state = initial_state.copy()
        current_state[6][4] = False  # e2 (rank 6, file 4) is now empty
        current_state[4][4] = True   # e4 (rank 4, file 4) now has a piece

        result = self.manager.update_board_from_array(current_state)
        self.assertIsNotNone(result, "A valid move should return a result")
        self.assertEqual(result[0], "e2e4", "The move should be e2e4")

    def test_ambiguous_move(self):
        """Test when multiple differences are detected."""
        initial_state = self.manager.get_bool_array()
        # Simulate multiple changes
        current_state = initial_state.copy()
        current_state[6][4] = False  # e2 (rank 6, file 4) is now empty
        current_state[4][4] = True   # e4 (rank 4, file 4) now has a piece
        current_state[6][3] = False  # d2 (rank 6, file 3) is now empty
        current_state[4][3] = True   # d4 (rank 4, file 3) now has a piece

        result = self.manager.update_board_from_array(current_state)
        self.assertIsNone(result, "Ambiguous moves should return None")

    def test_illegal_move(self):
        """Test when an illegal move is detected."""
        initial_state = self.manager.get_bool_array()
        # Simulate an illegal move: a piece "appears" in an invalid position
        current_state = initial_state.copy()
        current_state[6][4] = False  # e2 (rank 6, file 4) is now empty
        current_state[4][5] = True   # f4 (rank 4, file 5) now has a piece (illegal)

        result = self.manager.update_board_from_array(current_state)
        self.assertIsNone(result, "Illegal moves should return None")

if __name__ == "__main__":
    unittest.main()
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

    def test_valid_move_returns_uci(self):
        # A piece moves from e2 to e4
        old_board = np.full((8, 8), False)
        new_board = old_board.copy()

        # e2 = rank 6, file 4 | e4 = rank 4, file 4
        old_board[6][4] = True
        new_board[4][4] = True

         # Synchronize the chess.Board object with the old_board state
        self.manager.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/4P3/RNBQKBNR w KQkq - 0 1")

        self.manager.previous_state = old_board.copy()
        result = self.manager.update_board_from_array(new_board)

        self.assertEqual(result, "e2e4")

    def test_multiple_differences_returns_none(self):
        board1 = np.full((8, 8), False)
        board2 = board1.copy()
        board1[7][0] = True  # a1
        board1[7][7] = True  # h1

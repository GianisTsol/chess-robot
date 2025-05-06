import os
import chess
from stockfish import Stockfish


class ChessGameManager():
    def __init__(self, stockfish_path="stockfish", skill_level=5):
        self.board = chess.Board()

        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Did not find stockfish in: {stockfish_path}")

        self.stockfish = Stockfish(path=stockfish_path)
        self.previous_state = None  # 8x8 boolean board

    def update_board_from_array(self, current_state):
        """
        Update game state based on 8x8 boolean board array (True=piece, False=empty).
        """
        if self.previous_state is None:
            self.previous_state = current_state
            return None  # No move yet

        move = self._infer_move(self.previous_state, current_state)
        self.previous_state = current_state

        if move and chess.Move.from_uci(move) in self.board.legal_moves:
            self.board.push_uci(move)
            self.stockfish.set_position(self.board.fen())
            response_move = self.stockfish.best_move()
            self.board.push_uci(response_move)
            return move, response_move

        return None

    def _infer_move(self, prev, curr):
        """
        Try to infer a move from the difference between two boolean arrays.
        Returns a UCI string (like 'e2e4') or None if it fails.
        """
        from_square = None
        to_square = None

        for rank in range(8):
            for file in range(8):
                index = rank * 8 + file
                if prev[rank][file] and not curr[rank][file]:
                    from_square = chess.square(file, 7 - rank)
                if not prev[rank][file] and curr[rank][file]:
                    to_square = chess.square(file, 7 - rank)

        if from_square is not None and to_square is not None:
            return chess.Move(from_square, to_square).uci()

        return None

    def get_board_fen(self):
        return self.board.fen()

    def print_board(self):
        print(self.board)

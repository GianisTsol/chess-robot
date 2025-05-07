import os
import chess, chess.engine
import numpy as np

class ChessGameManager():
    def __init__(self, stockfish_path="stockfish", skill_level=5):
        self.board = chess.Board()

        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Did not find stockfish in: {stockfish_path}")

        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.previous_state = self.get_bool_array()  # 8x8 boolean board

    def get_bool_array(self):
        # Create an 8x8 boolean array initialized to False
        piece_presence = np.zeros((8, 8), dtype=bool)

        # Loop through all squares
        for square in chess.SQUARES:
            if self.board.piece_at(square):
                rank = 7 - chess.square_rank(square)  # 0-indexed from top to bottom
                file = chess.square_file(square)      # 0-indexed from left to right
                piece_presence[rank][file] = True
        return piece_presence

    def update_board_from_array(self, current_state):
        """
        Update game state based on 8x8 boolean board array (True=piece, False=empty).
        """
        if self.previous_state is None:
            self.previous_state = current_state.copy()
            return None
        diff = self.previous_state.copy() != self.get_bool_array()
        if np.any(diff):
            coords = np.argwhere(diff)
            print("Differences found at:")
            for y, x in coords:
                print(f"Wrong state - Position ({y}, {x})")
            self.previous_state = None
            return None

        if not np.any(self.previous_state.copy() != current_state):
            return None

        move = self._infer_move(self.previous_state, current_state)
        self.previous_state = current_state
        if move:
            if not chess.Move.from_uci(move) in self.board.legal_moves:
                print("illegal move put it back")
                return None
            self.board.push_uci(move)
            resp = self.engine.play(self.board, chess.engine.Limit(time=0.1))
            self.board.push(resp.move)
            return move, resp.move

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

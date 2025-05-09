import os
import chess, chess.engine
import numpy as np

class ChessGameManager():
    def __init__(self, stockfish_path="stockfish", skill_level=5, mock_engine=False):
        self.board = chess.Board()

        if not mock_engine:
            if not os.path.exists(stockfish_path):
                raise FileNotFoundError(f"Did not find stockfish in: {stockfish_path}")

            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        else:
            self.engine = None

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
        Returns a string with the move in UCI format if successful, or None if no move detected.
        When using an engine, returns a tuple (player_move, engine_move).
        """
        if self.previous_state is None:
            # Initialize previous state if not already set
            self.previous_state = current_state.copy()
            return None

        # Detect differences between previous and current state
        diff = self.previous_state != current_state
        
        # If no differences, return None
        if not np.any(diff):
            return None
        
        # Separate differences into from_squares and to_squares
        from_squares = []
        to_squares = []
        coords = np.argwhere(diff)
        
        for rank, file in coords:
            if self.previous_state[rank][file] and not current_state[rank][file]:
                from_squares.append((rank, file))
            elif not self.previous_state[rank][file] and current_state[rank][file]:
                to_squares.append((rank, file))

        # Store current state before any move processing
        new_previous_state = current_state.copy()
        
        # Standard move (one piece moved from one square to another)
        if len(from_squares) == 1 and len(to_squares) == 1:
            from_rank, from_file = from_squares[0]
            to_rank, to_file = to_squares[0]
            from_square = chess.square(from_file, 7 - from_rank)
            to_square = chess.square(to_file, 7 - to_rank)
            
            # Check for promotion (pawn moving to the last rank)
            piece = self.board.piece_at(from_square)
            is_promotion = False
            
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and chess.square_rank(to_square) == 7) or \
                (piece.color == chess.BLACK and chess.square_rank(to_square) == 0):
                    is_promotion = True
            
            # Try to make a move
            if is_promotion:
                # Try promotion pieces in order: Queen, Rook, Bishop, Knight
                for promotion_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    move = chess.Move(from_square, to_square, promotion=promotion_piece)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        self.previous_state = new_previous_state
                        
                        move_uci = move.uci()
                        if self.engine:
                            resp = self.engine.play(self.board, chess.engine.Limit(time=0.1))
                            self.board.push(resp.move)
                            return move_uci, resp.move
                        return move_uci
            else:
                # Standard move
                move = chess.Move(from_square, to_square)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.previous_state = new_previous_state
                    
                    move_uci = move.uci()
                    if self.engine:
                        resp = self.engine.play(self.board, chess.engine.Limit(time=0.1))
                        self.board.push(resp.move)
                        return move_uci, resp.move
                    return move_uci
        
        # Castling (king and rook moved)
        elif len(from_squares) == 2 and len(to_squares) == 2:
            # Try to identify king position
            king_move = None
            
            for from_rank, from_file in from_squares:
                from_square = chess.square(from_file, 7 - from_rank)
                piece = self.board.piece_at(from_square)
                
                if piece and piece.piece_type == chess.KING:
                    # Find where the king went
                    for to_rank, to_file in to_squares:
                        to_square = chess.square(to_file, 7 - to_rank)
                        move = chess.Move(from_square, to_square)
                        
                        # If this is a legal castling move
                        if move in self.board.legal_moves and abs(from_file - to_file) == 2:
                            self.board.push(move)
                            self.previous_state = new_previous_state
                            
                            move_uci = move.uci()
                            if self.engine:
                                resp = self.engine.play(self.board, chess.engine.Limit(time=0.1))
                                self.board.push(resp.move)
                                return move_uci, resp.move
                            return move_uci
        
        # En passant (one piece disappeared, one moved)
        elif len(from_squares) == 2 and len(to_squares) == 1:
            for from_rank, from_file in from_squares:
                from_square = chess.square(from_file, 7 - from_rank)
                to_rank, to_file = to_squares[0]
                to_square = chess.square(to_file, 7 - to_rank)
                
                move = chess.Move(from_square, to_square)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    # Check if this matches the current state
                    test_state = self.get_bool_array()
                    if np.array_equal(test_state, current_state):
                        self.previous_state = new_previous_state
                        
                        move_uci = move.uci()
                        if self.engine:
                            resp = self.engine.play(self.board, chess.engine.Limit(time=0.1))
                            self.board.push(resp.move)
                            return move_uci, resp.move
                        return move_uci
                    else:
                        # Undo the move and continue checking
                        self.board.pop()
        
        # General case: try all legal moves and see if any match the current state
        if len(from_squares) > 0:
            # Get all legal moves from the 'from' squares
            for from_rank, from_file in from_squares:
                from_square = chess.square(from_file, 7 - from_rank)
                
                for move in self.board.legal_moves:
                    if move.from_square == from_square:
                        # Try this move
                        self.board.push(move)
                        test_state = self.get_bool_array()
                        
                        # Check if the resulting state matches the current state
                        if np.array_equal(test_state, current_state):
                            self.previous_state = new_previous_state
                            
                            move_uci = move.uci()
                            if self.engine:
                                resp = self.engine.play(self.board, chess.engine.Limit(time=0.1))
                                self.board.push(resp.move)
                                return move_uci, resp.move
                            return move_uci
                        
                        # Undo the move and try the next one
                        self.board.pop()
        
        # If all else fails, update the previous state and return None
        self.previous_state = new_previous_state
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
                    print(f"Detected piece moved from rank {rank}, file {file} -> square {from_square}")
                if not prev[rank][file] and curr[rank][file]:
                    to_square = chess.square(file, 7 - rank)
                    print(f"Detected piece moved to rank {rank}, file {file} -> square {to_square}")

        if from_square is not None and to_square is not None:
            print(f"From square: {from_square}, To square: {to_square}")
            return chess.Move(from_square, to_square).uci()

        return None

    def get_board_fen(self):
        return self.board.fen()

    def print_board(self):
        print(self.board)

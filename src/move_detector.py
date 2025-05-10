import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
from .hsv_detector import HSVPieceDetector

class ChessboardDetector():
    def __init__(self, camera_index=2, model_path="chess_square_classifier.h5", use_hsv=True):
        """Initialize the chessboard detector with camera and model settings."""
        # Configuration
        self.chessboard_inner = (7, 7)  # Inner corners for 8x8 board
        self.square_len_px = 64         # Size of squares in final warped view
        self.board_size = 8             # 8x8 chessboard
        self.use_hsv = use_hsv          # Whether to use HSV filtering

        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}")

        self.thread = threading.Thread(target=self.run)
        self.running = False
        
        # Model setup
        self.model = load_model(model_path)
        self.img_size = 64
        self.categories = ["empty", "piece"]  # Index 0 = empty, 1 = piece

        # HSV detector
        self.hsv_detector = HSVPieceDetector()

        # State variables
        self.board_state = np.zeros((8, 8), dtype=bool)
        self.game_state = np.zeros((8, 8), dtype=bool) # game state from chess_engine to display on the cv2 view
        self.piece_colors = np.zeros((8, 8), dtype=np.int8)  # 0: empty, 1: white, -1: black

        self.last_valid_M = None  # Store last valid transformation matrix
        self.frame_counter = 0
        self.detection_interval = 10  # Detect pieces every 10 frames
        self.board_state_ready = False  # Flag to indicate if board state is ready
        
        # Calibration state
        self.calibration_mode = False

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

    def set_ready(self):
        self.board_state_ready = True

    def consume(self):
        if self.board_state_ready:
            self.board_state_ready = False
            return True
        return False

    def order_points_clockwise(self, pts):
        """Order corners: [top-left, top-right, bottom-right, bottom-left]"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]      # top-left has the smallest sum
        rect[2] = pts[np.argmax(s)]      # bottom-right has the largest sum
        rect[1] = pts[np.argmin(diff)]   # top-right has the smallest difference
        rect[3] = pts[np.argmax(diff)]   # bottom-left has the largest difference

        return rect

    def prepare_image(self, img):
        """Prepare image for model prediction."""
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0  # Normalize
        return img

    def detect_pieces_batch(self, warped_image):
        """Detect pieces on all squares using batch prediction for efficiency."""
        if self.use_hsv:
            # Use HSV-based detection
            color_state = self.hsv_detector.detect_pieces(warped_image, self.square_len_px)
            self.piece_colors = color_state
            self.board_state = (color_state != 0)  # Convert to boolean (True if piece present)
        else:
            # Use the CNN model
            squares = []
            positions = []

            # Extract all squares
            for row in range(self.board_size):
                for col in range(self.board_size):
                    y, x = col * self.square_len_px, row * self.square_len_px
                    square = warped_image[y:y+self.square_len_px, x:x+self.square_len_px]
                    square = self.prepare_image(square)
                    squares.append(square)
                    positions.append((row, col))

            # Batch predict
            if squares:
                batch = np.array(squares)
                predictions = self.model.predict(batch, verbose=0)  # Suppress verbose output

                # Update board state
                for i, (row, col) in enumerate(positions):
                    has_piece = np.argmax(predictions[i]) == 1  # 1 = piece
                    self.board_state[row][col] = has_piece

    def detect_chessboard_corners(self, gray):
        """Detect chessboard corners in the image."""
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_inner, flags)

        if not found:
            flags += cv2.CALIB_CB_FAST_CHECK
            found, corners = cv2.findChessboardCorners(gray, self.chessboard_inner, flags)

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners.reshape(self.chessboard_inner[1], self.chessboard_inner[0], 2)

        return None

    def calculate_perspective_transform(self, corners):
        """Calculate perspective transform matrix from detected corners."""
        # Estimate outer corners
        top_row_vec = corners[0, -1] - corners[0, 0]
        left_col_vec = corners[-1, 0] - corners[0, 0]

        # Extrapolate outer corners with robust calculation
        scale_x = 1.0 / (self.chessboard_inner[0] - 1)
        scale_y = 1.0 / (self.chessboard_inner[1] - 1)

        outer_pts = [
            corners[0, 0] - top_row_vec * scale_x - left_col_vec * scale_y,  # top-left
            corners[0, -1] + top_row_vec * scale_x - left_col_vec * scale_y,  # top-right
            corners[-1, -1] + top_row_vec * scale_x + left_col_vec * scale_y, # bottom-right
            corners[-1, 0] - top_row_vec * scale_x + left_col_vec * scale_y   # bottom-left
        ]

        # Perspective warp
        dst_size = (self.board_size * self.square_len_px, self.board_size * self.square_len_px)
        dst_pts = np.float32([
            [0, 0],
            [dst_size[0]-1, 0],
            [dst_size[0]-1, dst_size[1]-1],
            [0, dst_size[1]-1]
        ])

        ordered_pts = self.order_points_clockwise(np.array(outer_pts))
        src_pts = np.float32(ordered_pts)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return M, dst_size, outer_pts

    def draw_debug_info(self, frame, outer_pts=None, has_valid_board=False):
        """Draw debug information on the frame."""
        display = frame.copy()

        if outer_pts and has_valid_board:
            # Draw corner points
            for i, pt in enumerate(outer_pts):
                pt_int = tuple(np.int32(pt))
                cv2.circle(display, pt_int, 10, (0, 0, 255), 2)
                cv2.putText(display, f'{i}', pt_int, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        elif not has_valid_board:
            cv2.putText(display, "Chessboard not detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show detection method
        method = "HSV Detection" if self.use_hsv else "CNN Detection"
        cv2.putText(display, method, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return display

    def draw_board_state(self, warped_image):
        """Draw the current board state on the warped image."""
        result = warped_image.copy()
        square_size = self.square_len_px

        # Draw pieces
        for row in range(self.board_size):
            for col in range(self.board_size):
                y, x = col * square_size, row * square_size
                center = (x + square_size//2, y + square_size//2)
                
                if self.use_hsv:
                    # HSV detection - show piece colors
                    if self.piece_colors[row][col] == 1:  # White piece
                        cv2.circle(result, center, square_size//4, (0, 255, 255), 2)
                        cv2.putText(result, "W", (x + 5, y + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    elif self.piece_colors[row][col] == -1:  # Black piece
                        cv2.circle(result, center, square_size//4, (255, 0, 255), 2)
                        cv2.putText(result, "B", (x + 5, y + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                else:
                    # CNN detection - just show piece presence
                    if self.board_state[row][col]:
                        cv2.circle(result, center, square_size//4, (0, 0, 255), 2)
                
                # Draw game state from chess engine
                if self.game_state[row][col]:
                    cv2.circle(result, center, square_size//6, (158, 0, 255), 2)

        # Draw grid lines
        for i in range(1, self.board_size):
            cv2.line(result, (i*square_size, 0), (i*square_size, self.board_size*square_size), (0, 255, 0), 1)
            cv2.line(result, (0, i*square_size), (self.board_size*square_size, i*square_size), (0, 255, 0), 1)

        return result

    def update_game_state(self, state):
        self.game_state = state

    def toggle_detection_method(self):
        """Toggle between HSV and CNN detection methods."""
        self.use_hsv = not self.use_hsv
        print(f"Using {'HSV' if self.use_hsv else 'CNN'} detection")

    def enter_hsv_calibration(self):
        """Enter HSV calibration mode."""
        self.calibration_mode = True
        print("Entering HSV calibration mode. Press ESC when done.")

    def run(self):
        """Main processing loop."""
        print("Press 'q' to quit, 'c' to detect corners, 'm' to toggle detection method, 'h' for HSV calibration.")
        corners = None
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_counter += 1

            # Preprocess image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            has_valid_board = False
            outer_pts = None
            warped_image = None
            dst_size = None

            # Check for key inputs
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break
            elif k & 0xFF == ord('c'):
                corners = self.detect_chessboard_corners(gray)
            elif k & 0xFF == ord('p'):
                self.set_ready()
            elif k & 0xFF == ord('t'):
                self.toggle_detection_method()
            elif k & 0xFF == ord('h'):
                self.enter_hsv_calibration()

            if corners is not None:
                # Calculate perspective transform
                M, dst_size, outer_pts = self.calculate_perspective_transform(corners)
                self.last_valid_M = M
                has_valid_board = True
                # Warp perspective to get top-down view
                warped_image = cv2.warpPerspective(frame, M, dst_size)
            elif self.last_valid_M is not None:
                # Use the last valid transformation if no corners detected
                M = self.last_valid_M
                dst_size = (self.board_size * self.square_len_px, self.board_size * self.square_len_px)
                has_valid_board = True
                # Warp perspective to get top-down view
                warped_image = cv2.warpPerspective(frame, M, dst_size)

            # Display results
            display = self.draw_debug_info(frame, outer_pts, has_valid_board)
            cv2.imshow("Camera Feed", display)

            if has_valid_board and warped_image is not None:
                # Check for calibration mode
                if self.calibration_mode:
                    # Run HSV calibration
                    self.hsv_detector.calibrate_from_image(warped_image, self.square_len_px)
                    self.calibration_mode = False
                
                # Detect pieces at specified intervals for performance
                if self.frame_counter >= self.detection_interval:
                    self.detect_pieces_batch(warped_image)
                    self.frame_counter = 0

                # Draw the board state
                result = self.draw_board_state(warped_image)
                cv2.imshow("Chessboard Detection", result)


        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ChessboardDetector(camera_index=2)
    detector.run()
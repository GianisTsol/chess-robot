import cv2
import numpy as np

class HSVPieceDetector:
    """
    Chess piece detector using HSV color filtering to distinguish between white and black pieces,
    and empty squares based on their color characteristics.
    """
    def __init__(self):
        # Define HSV ranges for white pieces, black pieces, and empty squares
        # These values will need calibration for your specific lighting and pieces
        self.white_lower = np.array([0, 0, 180])  # High value (bright)
        self.white_upper = np.array([180, 30, 255])
        
        self.black_lower = np.array([0, 0, 0])  # Low value (dark)
        self.black_upper = np.array([180, 255, 60])
        
        # Parameters for piece detection
        self.min_piece_area = 100  # Minimum area to consider as a piece
        self.board_state = np.zeros((8, 8), dtype=np.int8)  # 0: empty, 1: white, -1: black
        
        # Debug mode
        self.debug = False

    def set_hsv_ranges(self, white_lower, white_upper, black_lower, black_upper):
        """Update HSV ranges for piece detection."""
        self.white_lower = np.array(white_lower)
        self.white_upper = np.array(white_upper)
        self.black_lower = np.array(black_lower)
        self.black_upper = np.array(black_upper)
    
    def analyze_square(self, square_img):
        """
        Analyze a single chess square to determine if it contains a piece and what color.
        Returns: 0 for empty, 1 for white piece, -1 for black piece
        """
        # Convert to HSV
        hsv = cv2.cvtColor(square_img, cv2.COLOR_BGR2HSV)
        
        # Create masks for white and black pieces
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        black_mask = cv2.inRange(hsv, self.black_lower, self.black_upper)
        
        # Calculate area of white and black regions
        white_area = cv2.countNonZero(white_mask)
        black_area = cv2.countNonZero(black_mask)
        
        # Calculate total area of the square
        total_area = square_img.shape[0] * square_img.shape[1]
        
        # Calculate percentage of white and black regions
        white_percentage = white_area / total_area
        black_percentage = black_area / total_area
        
        # Debug information
        if self.debug:
            print(f"White: {white_percentage:.2f}, Black: {black_percentage:.2f}")
            cv2.imshow("Square", square_img)
            cv2.imshow("White Mask", white_mask)
            cv2.imshow("Black Mask", black_mask)
            cv2.waitKey(1)
        
        # Determine piece type based on dominant color
        # These thresholds will need tuning
        if white_percentage > 0.3:
            return 1  # White piece
        elif black_percentage > 0.2:
            return -1  # Black piece
        else:
            return 0  # Empty square
    
    def detect_pieces(self, warped_image, square_size):
        """
        Detect pieces on the entire board and update the board state.
        Returns an 8x8 array with 0 (empty), 1 (white), or -1 (black)
        """
        result = np.zeros((8, 8), dtype=np.int8)
        
        for row in range(8):
            for col in range(8):
                # Extract the square
                y, x = col * square_size, row * square_size
                square = warped_image[y:y+square_size, x:x+square_size]
                
                # Analyze the square
                result[row][col] = self.analyze_square(square)
        
        self.board_state = result
        return result
    
    def get_piece_presence(self):
        """Convert the detailed board state to a simple boolean array (True=piece, False=empty)."""
        return self.board_state != 0
    
    def calibrate_from_image(self, warped_image, square_size):
        """
        Interactive calibration tool to set HSV ranges based on selected squares.
        """
        def nothing(x):
            pass
        
        cv2.namedWindow('Calibration')
        # Create trackbars for HSV ranges
        cv2.createTrackbar('White H Min', 'Calibration', self.white_lower[0], 180, nothing)
        cv2.createTrackbar('White H Max', 'Calibration', self.white_upper[0], 180, nothing)
        cv2.createTrackbar('White S Min', 'Calibration', self.white_lower[1], 255, nothing)
        cv2.createTrackbar('White S Max', 'Calibration', self.white_upper[1], 255, nothing)
        cv2.createTrackbar('White V Min', 'Calibration', self.white_lower[2], 255, nothing)
        cv2.createTrackbar('White V Max', 'Calibration', self.white_upper[2], 255, nothing)
        
        cv2.createTrackbar('Black H Min', 'Calibration', self.black_lower[0], 180, nothing)
        cv2.createTrackbar('Black H Max', 'Calibration', self.black_upper[0], 180, nothing)
        cv2.createTrackbar('Black S Min', 'Calibration', self.black_lower[1], 255, nothing)
        cv2.createTrackbar('Black S Max', 'Calibration', self.black_upper[1], 255, nothing)
        cv2.createTrackbar('Black V Min', 'Calibration', self.black_lower[2], 255, nothing)
        cv2.createTrackbar('Black V Max', 'Calibration', self.black_upper[2], 255, nothing)
        
        selected_square = None
        
        while True:
            # Get current trackbar positions
            w_h_min = cv2.getTrackbarPos('White H Min', 'Calibration')
            w_h_max = cv2.getTrackbarPos('White H Max', 'Calibration')
            w_s_min = cv2.getTrackbarPos('White S Min', 'Calibration')
            w_s_max = cv2.getTrackbarPos('White S Max', 'Calibration')
            w_v_min = cv2.getTrackbarPos('White V Min', 'Calibration')
            w_v_max = cv2.getTrackbarPos('White V Max', 'Calibration')
            
            b_h_min = cv2.getTrackbarPos('Black H Min', 'Calibration')
            b_h_max = cv2.getTrackbarPos('Black H Max', 'Calibration')
            b_s_min = cv2.getTrackbarPos('Black S Min', 'Calibration')
            b_s_max = cv2.getTrackbarPos('Black S Max', 'Calibration')
            b_v_min = cv2.getTrackbarPos('Black V Min', 'Calibration')
            b_v_max = cv2.getTrackbarPos('Black V Max', 'Calibration')
            
            # Update HSV ranges
            self.white_lower = np.array([w_h_min, w_s_min, w_v_min])
            self.white_upper = np.array([w_h_max, w_s_max, w_v_max])
            self.black_lower = np.array([b_h_min, b_s_min, b_v_min])
            self.black_upper = np.array([b_h_max, b_s_max, b_v_max])
            
            # Display the warped chessboard
            display = warped_image.copy()
            
            # Draw grid
            for i in range(1, 8):
                cv2.line(display, (i*square_size, 0), (i*square_size, 8*square_size), (0, 255, 0), 1)
                cv2.line(display, (0, i*square_size), (8*square_size, i*square_size), (0, 255, 0), 1)
            
            # Detect and show pieces with current settings
            board_state = self.detect_pieces(warped_image, square_size)
            
            # Overlay piece detection results
            for row in range(8):
                for col in range(8):
                    y, x = col * square_size, row * square_size
                    center = (x + square_size//2, y + square_size//2)
                    
                    if board_state[row][col] == 1:  # White piece
                        cv2.circle(display, center, square_size//4, (0, 255, 255), 2)
                    elif board_state[row][col] == -1:  # Black piece
                        cv2.circle(display, center, square_size//4, (255, 0, 255), 2)
            
            cv2.imshow('Calibration', display)
            
            # Handle mouse clicks for square selection
            def mouse_callback(event, x, y, flags, param):
                nonlocal selected_square
                if event == cv2.EVENT_LBUTTONDOWN:
                    row = x // square_size
                    col = y // square_size
                    selected_square = (row, col)
                    
                    # Extract the square
                    y_pos, x_pos = col * square_size, row * square_size
                    square = warped_image[y_pos:y_pos+square_size, x_pos:x_pos+square_size]
                    
                    # Show HSV histograms for the selected square
                    hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(hsv)
                    
                    hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
                    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
                    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
                    
                    # Normalize for display
                    cv2.normalize(hist_h, hist_h, 0, 255, cv2.NORM_MINMAX)
                    cv2.normalize(hist_s, hist_s, 0, 255, cv2.NORM_MINMAX)
                    cv2.normalize(hist_v, hist_v, 0, 255, cv2.NORM_MINMAX)
                    
                    # Create histogram images
                    hist_img_h = np.zeros((256, 180, 3), np.uint8)
                    hist_img_s = np.zeros((256, 256, 3), np.uint8)
                    hist_img_v = np.zeros((256, 256, 3), np.uint8)
                    
                    for i in range(180):
                        cv2.line(hist_img_h, (i, 255), (i, 255 - int(hist_h[i])), (255, 0, 0), 1)
                    
                    for i in range(256):
                        cv2.line(hist_img_s, (i, 255), (i, 255 - int(hist_s[i])), (0, 255, 0), 1)
                        cv2.line(hist_img_v, (i, 255), (i, 255 - int(hist_v[i])), (0, 0, 255), 1)
                    
                    cv2.imshow('Selected Square', square)
                    cv2.imshow('H Histogram', hist_img_h)
                    cv2.imshow('S Histogram', hist_img_s)
                    cv2.imshow('V Histogram', hist_img_v)
            
            cv2.setMouseCallback('Calibration', mouse_callback)
            
            # Exit on ESC
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        
        print("Calibration complete. HSV ranges:")
        print(f"White: H[{self.white_lower[0]}-{self.white_upper[0]}], "
              f"S[{self.white_lower[1]}-{self.white_upper[1]}], "
              f"V[{self.white_lower[2]}-{self.white_upper[2]}]")
        print(f"Black: H[{self.black_lower[0]}-{self.black_upper[0]}], "
              f"S[{self.black_lower[1]}-{self.black_upper[1]}], "
              f"V[{self.black_lower[2]}-{self.black_upper[2]}]")
        
        return self.white_lower, self.white_upper, self.black_lower, self.black_upper
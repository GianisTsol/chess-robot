import cv2
import numpy as np

# Setup
chessboard_inner = (7, 7)  # Inner corners, not squares (for 8x8 squares)
square_len_px = 50         # Size of squares in the final warped view

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Press 'q' to quit.")

empty_avg_colors = np.zeros((8, 8, 3))
def get_square_avg_color(image, row, col, square_size):
    """
    Returns the average BGR color of a square on the top-down chessboard image.

    Parameters:
        image (np.ndarray): Top-down view of the chessboard.
        row (int): Row index (0 at top).
        col (int): Column index (0 at left).
        square_size (int): Pixel size of one square.

    Returns:
        tuple: (B, G, R) average color of the square.
    """
    y_start = int(row * square_size)
    y_end = int(y_start + square_size)
    x_start = int(col * square_size)
    x_end = int(x_start + square_size)

    square = image[y_start:y_end, x_start:x_end]
    avg_color = cv2.mean(square)[:3]  # (B, G, R)
    return avg_color

def square_has_piece(gray_img, square_coords, threshold=380):
    x, y, w, h = square_coords
    square = gray_img[y:y+h, x:x+w]

    # Option 1: Use intensity standard deviation
    #stddev = np.std(square)

    # Option 2 (alternative): Use edge detection
    edges = cv2.Canny(square, 50, 150)
    edge_pixels = np.sum(edges > 0)

    return edge_pixels > threshold

o = False
c = True
M, dst_size = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, chessboard_inner, flags)

    display = frame.copy()

    if found:
        o = True
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners = corners.reshape(chessboard_inner[1], chessboard_inner[0], 2)

        # Estimate vectors along rows and columns
        top_row_vec = corners[0, -1] - corners[0, 0]
        left_col_vec = corners[-1, 0] - corners[0, 0]

        # Extrapolate outer corners
        top_left = corners[0, 0] - top_row_vec / (chessboard_inner[0] - 1) - left_col_vec / (chessboard_inner[1] - 1)
        top_right = corners[0, -1] + top_row_vec / (chessboard_inner[0] - 1) - left_col_vec / (chessboard_inner[1] - 1)
        bottom_right = corners[-1, -1] + top_row_vec / (chessboard_inner[0] - 1) + left_col_vec / (chessboard_inner[1] - 1)
        bottom_left = corners[-1, 0] - top_row_vec / (chessboard_inner[0] - 1) + left_col_vec / (chessboard_inner[1] - 1)


        # Debug draw points
        outer_pts = [top_left, top_right, bottom_right, bottom_left]
        for i, pt in enumerate(outer_pts):
            pt_int = tuple(np.int32(pt))
            cv2.circle(display, pt_int, 10, (0, 0, 255), 2)
            cv2.putText(display, f'{i}', pt_int, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Perspective warp
        dst_size = (8 * square_len_px, 8 * square_len_px)
        dst_pts = np.float32([
            [0, 0],
            [dst_size[0] - 1, 0],
            [dst_size[0] - 1, dst_size[1] - 1],
            [0, dst_size[1] - 1]
        ])
        src_pts = np.float32(outer_pts)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    if o:
        top_down = cv2.warpPerspective(frame, M, dst_size)
        gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)
        square_l = (top_down.shape[0] + top_down.shape[1] ) / 16

        for j in range(0, 7):
            for k in range(0, 7):
                has_piece = square_has_piece(gray, np.int32([square_l * j, square_l * k, square_l, square_l]))
                if has_piece:
                    pp = tuple(np.int32([(square_l*j) + (square_l/2), (square_l*k) + (square_l/2)]))
                    cv2.circle(gray, pp, 10, (0, 0, 255), 2)


        cv2.imshow("Top-Down Chessboard (Full)", gray)
    c = False
    # Show camera with debug overlay
    cv2.imshow("Camera Feed with Chessboard Overlay", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

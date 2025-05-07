import cv2
import cv2 as cv
import numpy as np

# Setup
chessboard_inner = (7, 7)  # Inner corners for 8x8 board
square_len_px = 50         # Size of squares in final warped view

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Press 'q' to quit.")


calibrated_edges = np.zeros((8, 8, 1))

def order_points_clockwise(pts):
    # Order corners: [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left has the smallest sum
    rect[2] = pts[np.argmax(s)]      # bottom-right has the largest sum
    rect[1] = pts[np.argmin(diff)]   # top-right has the smallest difference
    rect[3] = pts[np.argmax(diff)]   # bottom-left has the largest difference

    return rect


def is_corner_square(row, col):
    """Check if square is one of the four corners"""
    return (row == 0 and col == 0) or (row == 0 and col == 7) or \
           (row == 7 and col == 0) or (row == 7 and col == 7)


def detect_chessboard_corners(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, chessboard_inner, flags)
    if not found:
        flags += cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, chessboard_inner, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners.reshape(chessboard_inner[1], chessboard_inner[0], 2)
    return None

valid_matrix = False
M, dst_size = None, None
last_valid_M = None  # Store last valid transformation matrix


def calibrate_square_edges(img, row, col):
    (x, y) = row * square_len_px, col * square_len_px
    square = img[y:y+square_len_px, x:x+square_len_px]
    edges = cv2.Canny(square, 50, 150)
    edge_pixels = np.sum(edges > 0)
    calibrated_edges[row][col] = edge_pixels

cal = False
kk = 150
corners = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    display = frame.copy()
    if cal:
        cal = False
        corners = detect_chessboard_corners(gray)

    if corners is not None:
        # Estimate outer corners
        top_row_vec = corners[0, -1] - corners[0, 0]
        left_col_vec = corners[-1, 0] - corners[0, 0]

        # Extrapolate outer corners with more robust calculation
        scale_x = 1.0 / (chessboard_inner[0] - 1)
        scale_y = 1.0 / (chessboard_inner[1] - 1)

        outer_pts = [
            corners[0, 0] - top_row_vec * scale_x - left_col_vec * scale_y,  # top-left
            corners[0, -1] + top_row_vec * scale_x - left_col_vec * scale_y,  # top-right
            corners[-1, -1] + top_row_vec * scale_x + left_col_vec * scale_y, # bottom-right
            corners[-1, 0] - top_row_vec * scale_x + left_col_vec * scale_y   # bottom-left
        ]

        # Perspective warp
        dst_size = (8 * square_len_px, 8 * square_len_px)
        dst_pts = np.float32([[0, 0], [dst_size[0]-1, 0],
                            [dst_size[0]-1, dst_size[1]-1], [0, dst_size[1]-1]])
        ordered_pts = order_points_clockwise(np.array(outer_pts))
        src_pts = np.float32(ordered_pts)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        last_valid_M = M  # Store the last valid transformation
        valid_matrix = True

        # Draw debug points
        for i, pt in enumerate(outer_pts):
            pt_int = tuple(np.int32(pt))
            cv2.circle(display, pt_int, 10, (0, 0, 255), 2)
            cv2.putText(display, f'{i}', pt_int, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        # If no corners detected but we have a previous valid matrix, use that
        if last_valid_M is not None:
            M = last_valid_M
            valid_matrix = True
        else:
            valid_matrix = False

    if valid_matrix:
        top_down = cv2.warpPerspective(frame, M, dst_size)
        gray_top = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)
        blur_image = cv.blur(gray_top,(3,3))

        sobel_image = blur_image
        square_size = dst_size[0] // 8

        # Detect pieces, skipping corner squares if needed
        f = cv2.waitKey(1)
        if f & 0xFF == ord('c'):
            cal = True
        if f & 0xFF == ord('e'):
            print("GETTING EMPTY IMAGES")
            for row in range(2, 6):
                for col in range(2, 6):
                    x, y = row * square_len_px, col * square_len_px

                    # Crop the current box
                    box = top_down[y:y + square_len_px, x:x + square_len_px]

                    # Save the box as an image
                    cv2.imwrite(f"dataset/empty2/chess_square_{kk}_{row}_{col}.png", box)
            kk += 1
        if f & 0xFF == ord('t'):
            print("GETTING PIEECE IMAGES")
            for row in range(2, 6):
                for col in range(2, 6):
                    x, y = row * square_len_px, col * square_len_px

                    # Crop the current box
                    box = top_down[y:y + square_len_px, x:x + square_len_px]

                    # Save the box as an image
                    cv2.imwrite(f"dataset/piece2/chess_square_{kk}_{row}_{col}.png", box)
            kk += 1
        cv2.imshow("Chessboard Detection", top_down)
    else:
        cv2.putText(display, "Chessboard not detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camera Feed", display)
    f = cv2.waitKey(1)
    if f & 0xFF == ord('c'):
        cal = True
    if f & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

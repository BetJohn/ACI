import cv2
import numpy as np

# ==== CONFIG ====
WINDOW_NAME = "Image (Left-click = include, Shift+drag = region)"
PREVIEW_NAME = "Track Preview"
PANEL_NAME = "Selected Colors"
FLOOD_PREVIEW = "Flood Preview"
FLOOD_PANEL = "Flood Points"
FILENAME = "./images/raw_image_3_formula_1.png"


# ==== INIT ====
img = cv2.imread(FILENAME)
if img is None:
    raise ValueError("Failed to load image. Please check the filename!")

included_colors = []
mask = np.zeros(img.shape[:2], dtype=np.uint8)
result = img.copy()

tol_include = 20
ker_include = 5
morph_order = "open_close"
region_mask = None
use_region_restrict = False
ker_open = 3
ker_close = 5

flood_ker_open = 3
flood_ker_close = 5

# Region selection
drawing_region = False
region_start = None
region_end = None

# Panel setup
panel_height = 200
swatch_size = 40
font = cv2.FONT_HERSHEY_SIMPLEX

# ==== FLOOD MODULE ====
flood_tolerance = 25
flood_seeds = []  # list of (x, y)
flood_mode_fixed = False
flood_result_mask = np.zeros(img.shape[:2], np.uint8)
flood_ker = 3


# ==== CORE FUNCTIONS ====
def build_mask():
    """Combine included colors and region restriction."""
    global mask, result
    include_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Include colors
    for color in included_colors:
        b, g, r = color
        lower = np.array([max(b - tol_include, 0), max(g - tol_include, 0), max(r - tol_include, 0)], np.uint8)
        upper = np.array([min(b + tol_include, 255), min(g + tol_include, 255), min(r + tol_include, 255)], np.uint8)
        include_mask |= cv2.inRange(img, lower, upper)

    mask = include_mask

    if use_region_restrict and region_mask is not None:
        mask = cv2.bitwise_and(mask, region_mask)

    # Morphological filters
    # Morphological filters
    # Morphological filters (separate open/close)
    ko = max(1, ker_open)
    kc = max(1, ker_close)
    if ko % 2 == 0: ko += 1
    if kc % 2 == 0: kc += 1
    kernel_open = np.ones((ko, ko), np.uint8)
    kernel_close = np.ones((kc, kc), np.uint8)

    if morph_order == "open_close":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    result = cv2.bitwise_and(img, img, mask=mask)


def update_preview():
    build_mask()
    cv2.imshow(PREVIEW_NAME, result)
    update_panel()


def update_panel():
    panel = np.ones((panel_height, 300, 3), dtype=np.uint8) * 255
    cv2.putText(panel, "INCLUDE COLORS", (20, 30), font, 0.7, (0, 180, 0), 2)

    for i, color in enumerate(included_colors):
        x = 20 + (i % 6) * (swatch_size + 5)
        y = 60 + (i // 6) * (swatch_size + 5)
        cv2.rectangle(panel, (x, y), (x + swatch_size, y + swatch_size), color.tolist(), -1)
        cv2.rectangle(panel, (x, y), (x + swatch_size, y + swatch_size), (0, 0, 0), 1)

    cv2.putText(panel, f"Morph: {morph_order}", (20, panel_height - 15), font, 0.5, (0, 0, 0), 1)
    cv2.putText(panel, f"Region: {'ON' if use_region_restrict else 'OFF'}", (150, panel_height - 15), font, 0.5, (0, 0, 0), 1)
    cv2.imshow(PANEL_NAME, panel)


def on_trackbar_change(val=None):
    global tol_include, ker_include
    tol_include = cv2.getTrackbarPos("Tol Inc", PREVIEW_NAME)
    ker_include = cv2.getTrackbarPos("Ker Inc", PREVIEW_NAME)
    update_preview()


# ==== FLOOD FILL MODULE ====
def flood_select_from_result():
    """Perform flood-fill based on the processed result image."""
    global flood_result_mask
    h, w = result.shape[:2]
    flood_result_mask = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    lo = (flood_tolerance,) * 3
    hi = (flood_tolerance,) * 3

    for (sx, sy) in flood_seeds:
        mask_ff = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | (255 << 8)
        if flood_mode_fixed:
            flags |= cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(gray.copy(), mask_ff, (sx, sy), 255, lo, hi, flags)
        flood_result_mask |= mask_ff[1:-1, 1:-1]

    apply_flood_morph()  # Do morph after flood
    update_flood_preview()


def apply_flood_morph():
    """Apply post-flood morphological operations and re-run flood once."""
    global flood_result_mask
    # Apply separate open/close kernels for flood result
    ko = max(1, flood_ker_open)
    kc = max(1, flood_ker_close)
    if ko % 2 == 0: ko += 1
    if kc % 2 == 0: kc += 1
    kernel_open = np.ones((ko, ko), np.uint8)
    kernel_close = np.ones((kc, kc), np.uint8)

    flood_result_mask = cv2.morphologyEx(flood_result_mask, cv2.MORPH_OPEN, kernel_open)
    flood_result_mask = cv2.morphologyEx(flood_result_mask, cv2.MORPH_CLOSE, kernel_close)


def update_flood_preview():
    flood_vis = cv2.bitwise_and(result, result, mask=flood_result_mask)
    cv2.imshow(FLOOD_PREVIEW, flood_vis)
    update_flood_panel()


def update_flood_panel():
    panel = np.ones((panel_height, 400, 3), dtype=np.uint8) * 255
    cv2.putText(panel, f"Mode: {'FIXED' if flood_mode_fixed else 'FLOAT'}", (20, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(panel, "Seeds:", (20, 60), font, 0.6, (0, 0, 0), 1)
    for i, (x, y) in enumerate(flood_seeds):
        sx = 20 + (i % 6) * (swatch_size + 5)
        sy = 80 + (i // 6) * (swatch_size + 5)
        cv2.rectangle(panel, (sx, sy), (sx + swatch_size, sy + swatch_size), (0, 200, 200), -1)
        cv2.putText(panel, f"{i+1}", (sx + 10, sy + 25), font, 0.6, (0, 0, 0), 1)
    cv2.imshow(FLOOD_PANEL, panel)


def flood_panel_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for i, (sx, sy) in enumerate(flood_seeds):
        bx = 20 + (i % 6) * (swatch_size + 5)
        by = 80 + (i // 6) * (swatch_size + 5)
        if bx <= x <= bx + swatch_size and by <= y <= by + swatch_size:
            flood_seeds.pop(i)
            print("💧 Removed flood seed")
            flood_select_from_result()
            return


def on_flood_tol_change(val):
    global flood_tolerance
    flood_tolerance = val
    flood_select_from_result()


def on_flood_ker_change(val):
    global flood_ker
    flood_ker = max(1, val)
    flood_select_from_result()


def flood_click_in_preview(event, x, y, flags, param):
    """Add flood seed when clicking in the result preview window."""
    if event == cv2.EVENT_LBUTTONDOWN:
        flood_seeds.append((x, y))
        print(f"💧 Added seed on processed image at ({x},{y})")
        flood_select_from_result()


# ==== MOUSE EVENTS ====
def image_click(event, x, y, flags, param):
    global drawing_region, region_start, region_end, region_mask

    if (flags & cv2.EVENT_FLAG_SHIFTKEY):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_region = True
            region_start = (x, y)
            region_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing_region:
            region_end = (x, y)
            temp = img.copy()
            cv2.rectangle(temp, region_start, region_end, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, temp)
        elif event == cv2.EVENT_LBUTTONUP and drawing_region:
            drawing_region = False
            region_end = (x, y)
            x1, y1 = min(region_start[0], x), min(region_start[1], y)
            x2, y2 = max(region_start[0], x), max(region_start[1], y)
            region_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            region_mask[y1:y2, x1:x2] = 255
            print(f"🟩 Region selected: ({x1},{y1}) → ({x2},{y2})")
            update_preview()
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        included_colors.append(img[y, x])
        print(f"🟢 Include color: {img[y, x].tolist()}")
        update_preview()


def panel_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for i, color in enumerate(included_colors):
        cx = 20 + (i % 6) * (swatch_size + 5)
        cy = 60 + (i // 6) * (swatch_size + 5)
        if cx <= x <= cx + swatch_size and cy <= y <= cy + swatch_size:
            included_colors.pop(i)
            print("🟢 Removed included color")
            update_preview()
            return


def main():
    # ==== SETUP ====
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(PREVIEW_NAME)
    cv2.namedWindow(PANEL_NAME)
    cv2.namedWindow(FLOOD_PREVIEW)
    cv2.namedWindow(FLOOD_PANEL)

    cv2.setMouseCallback(WINDOW_NAME, image_click)
    cv2.setMouseCallback(PANEL_NAME, panel_click)
    cv2.setMouseCallback(PREVIEW_NAME, flood_click_in_preview)
    cv2.setMouseCallback(FLOOD_PANEL, flood_panel_click)

    # Trackbars
    # ==== TRACKBARS (instant update with lambdas) ====
    cv2.createTrackbar(
        "Tol Inc", PREVIEW_NAME, tol_include, 100,
        lambda v: (globals().update(tol_include=v), update_preview())
    )

    cv2.createTrackbar(
        "Open Ker", PREVIEW_NAME, ker_open, 20,
        lambda v: (globals().update(ker_open=v), update_preview())
    )

    cv2.createTrackbar(
        "Close Ker", PREVIEW_NAME, ker_close, 20,
        lambda v: (globals().update(ker_close=v), update_preview())
    )

    cv2.createTrackbar(
        "Flood Tol", FLOOD_PREVIEW, flood_tolerance, 100,
        lambda v: (globals().update(flood_tolerance=v), flood_select_from_result())
    )

    cv2.createTrackbar(
        "Flood Open", FLOOD_PREVIEW, flood_ker_open, 20,
        lambda v: (globals().update(flood_ker_open=v), flood_select_from_result())
    )

    cv2.createTrackbar(
        "Flood Close", FLOOD_PREVIEW, flood_ker_close, 20,
        lambda v: (globals().update(flood_ker_close=v), flood_select_from_result())
    )
    
    cv2.imshow(WINDOW_NAME, img)
    update_preview()
    update_flood_preview()

    print("✅ Controls:")
    print("  - Left click: include color")
    print("  - Shift+drag: select region")
    print("  - o: toggle open/close order")
    print("  - r: toggle region restriction")
    print("  - s: save contoured image")
    print("  - q: quit")
    print("  - f: toggle flood mode (fixed/floating)")
    print("  - Click in the Track Preview window to add flood seeds")

    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Combine color result with flood mask
            masked = cv2.bitwise_and(result, result, mask=flood_result_mask)

            # Convert to grayscale for Otsu thresholding
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

            # Apply Otsu threshold to get crisp black–white result
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply a final morphological closing (kernel size 1 -> effectively 3x3)
            kernel_final = np.ones((3, 3), np.uint8)
            binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_final)

            # Save the final image
            cv2.imwrite("track_binary.png", binary_closed)
            print("💾 Saved track_binary.png (Otsu black–white with final closing)")
        
        elif key == ord('o'):
            morph_order = "close_open" if morph_order == "open_close" else "open_close"
            print(f"🔁 Morph order: {morph_order}")
            update_preview()

        elif key == ord('r'):
            use_region_restrict = not use_region_restrict
            print(f"🟩 Region restriction: {'ON' if use_region_restrict else 'OFF'}")
            update_preview()

        elif key == ord('f'):
            flood_mode_fixed = not flood_mode_fixed
            print(f"💧 Flood mode: {'FIXED' if flood_mode_fixed else 'FLOAT'}")
            flood_select_from_result()

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
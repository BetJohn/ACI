import cv2
import numpy as np

# ==== CONFIG ====
WINDOW_NAME = "Imagine (stânga=include, dreapta=exclude, Shift+drag=regiune)"
PREVIEW_NAME = "Preview pista"
PANEL_NAME = "Culori selectate"
FILENAME = "./images/raw_image_formula_1.png"

# ==== INIT ====
img = cv2.imread(FILENAME)
if img is None:
    raise ValueError("Nu am putut încărca imaginea. Verifică numele fișierului!")

included_colors = []
excluded_colors = []
mask = np.zeros(img.shape[:2], dtype=np.uint8)

tol_include, tol_exclude = 20, 20
ker_include, ker_exclude = 5, 5
morph_order = "open_close"
region_mask = None
use_region_restrict = False

# For region selection
drawing_region = False
region_start = None
region_end = None

# Panel setup
panel_height = 200
swatch_size = 40
font = cv2.FONT_HERSHEY_SIMPLEX


# ==== CORE FUNCTIONS ====
def build_mask():
    """Combine included and excluded colors with morphology and region restriction."""
    global mask
    include_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    exclude_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Include colors
    for color in included_colors:
        b, g, r = color
        lower = np.array([max(b - tol_include, 0), max(g - tol_include, 0), max(r - tol_include, 0)], np.uint8)
        upper = np.array([min(b + tol_include, 255), min(g + tol_include, 255), min(r + tol_include, 255)], np.uint8)
        include_mask |= cv2.inRange(img, lower, upper)

    # Exclude colors
    for color in excluded_colors:
        b, g, r = color
        lower = np.array([max(b - tol_exclude, 0), max(g - tol_exclude, 0), max(r - tol_exclude, 0)], np.uint8)
        upper = np.array([min(b + tol_exclude, 255), min(g + tol_exclude, 255), min(r + tol_exclude, 255)], np.uint8)
        exclude_mask |= cv2.inRange(img, lower, upper)

    mask = cv2.bitwise_and(include_mask, cv2.bitwise_not(exclude_mask))

    # Apply region restriction if active
    if use_region_restrict and region_mask is not None:
        mask = cv2.bitwise_and(mask, region_mask)

    # Morphological cleanup
    k_in = max(1, ker_include)
    k_ex = max(1, ker_exclude)
    if k_in % 2 == 0: k_in += 1
    if k_ex % 2 == 0: k_ex += 1

    kin = np.ones((k_in, k_in), np.uint8)
    kex = np.ones((k_ex, k_ex), np.uint8)

    if morph_order == "open_close":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kin)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kex)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kex)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kin)


def update_preview():
    build_mask()
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(PREVIEW_NAME, result)
    update_panel()


def update_panel():
    panel = np.ones((panel_height, 550, 3), dtype=np.uint8) * 255
    cv2.putText(panel, "INCLUDE", (20, 30), font, 0.7, (0, 180, 0), 2)
    cv2.putText(panel, "EXCLUDE", (280, 30), font, 0.7, (0, 0, 200), 2)

    for i, color in enumerate(included_colors):
        x = 20 + (i % 5) * (swatch_size + 5)
        y = 50 + (i // 5) * (swatch_size + 5)
        cv2.rectangle(panel, (x, y), (x + swatch_size, y + swatch_size), color.tolist(), -1)
        cv2.rectangle(panel, (x, y), (x + swatch_size, y + swatch_size), (0, 0, 0), 1)

    for i, color in enumerate(excluded_colors):
        x = 280 + (i % 5) * (swatch_size + 5)
        y = 50 + (i // 5) * (swatch_size + 5)
        cv2.rectangle(panel, (x, y), (x + swatch_size, y + swatch_size), color.tolist(), -1)
        cv2.rectangle(panel, (x, y), (x + swatch_size, y + swatch_size), (0, 0, 0), 1)

    cv2.putText(panel, f"Morph: {morph_order}", (20, panel_height - 15), font, 0.5, (0, 0, 0), 1)
    cv2.putText(panel, f"Region: {'ON' if use_region_restrict else 'OFF'}", (280, panel_height - 15), font, 0.5, (0, 0, 0), 1)
    cv2.imshow(PANEL_NAME, panel)


def on_trackbar_change(val=None):
    global tol_include, tol_exclude, ker_include, ker_exclude
    tol_include = cv2.getTrackbarPos("Tol Inc", PREVIEW_NAME)
    tol_exclude = cv2.getTrackbarPos("Tol Exc", PREVIEW_NAME)
    ker_include = cv2.getTrackbarPos("Ker Inc", PREVIEW_NAME)
    ker_exclude = cv2.getTrackbarPos("Ker Exc", PREVIEW_NAME)
    update_preview()


# ==== MOUSE EVENTS ====
def image_click(event, x, y, flags, param):
    global drawing_region, region_start, region_end, region_mask

    # Shift+Drag = region select
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

    # Normal color clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        included_colors.append(img[y, x])
        print(f"🟢 Include color: {img[y, x].tolist()}")
        update_preview()
    elif event == cv2.EVENT_RBUTTONDOWN:
        excluded_colors.append(img[y, x])
        print(f"🔴 Exclude color: {img[y, x].tolist()}")
        update_preview()


def panel_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for i, color in enumerate(included_colors):
        cx = 20 + (i % 5) * (swatch_size + 5)
        cy = 50 + (i // 5) * (swatch_size + 5)
        if cx <= x <= cx + swatch_size and cy <= y <= cy + swatch_size:
            included_colors.pop(i)
            print("🟢 Removed included color")
            update_preview()
            return
    for i, color in enumerate(excluded_colors):
        cx = 280 + (i % 5) * (swatch_size + 5)
        cy = 50 + (i // 5) * (swatch_size + 5)
        if cx <= x <= cx + swatch_size and cy <= y <= cy + swatch_size:
            excluded_colors.pop(i)
            print("🔴 Removed excluded color")
            update_preview()
            return


# ==== SETUP ====
cv2.namedWindow(WINDOW_NAME)
cv2.namedWindow(PREVIEW_NAME)
cv2.namedWindow(PANEL_NAME)

cv2.setMouseCallback(WINDOW_NAME, image_click)
cv2.setMouseCallback(PANEL_NAME, panel_click)

cv2.createTrackbar("Tol Inc", PREVIEW_NAME, tol_include, 100, on_trackbar_change)
cv2.createTrackbar("Tol Exc", PREVIEW_NAME, tol_exclude, 100, on_trackbar_change)
cv2.createTrackbar("Ker Inc", PREVIEW_NAME, ker_include, 20, on_trackbar_change)
cv2.createTrackbar("Ker Exc", PREVIEW_NAME, ker_exclude, 20, on_trackbar_change)

cv2.imshow(WINDOW_NAME, img)
update_preview()

print("✅ Controls:")
print("  - Stânga click: include color")
print("  - Dreapta click: exclude color")
print("  - Shift + drag: select region")
print("  - o: toggle open/close order")
print("  - r: toggle region restriction")
print("  - s: save contoured image")
print("  - q: quit")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoured = img.copy()
        cv2.drawContours(contoured, contours, -1, (0, 0, 255), 2)
        cv2.imwrite("track_contoured.jpg", contoured)
        print("💾 Saved track_contoured.jpg")

    elif key == ord('o'):
        morph_order = "close_open" if morph_order == "open_close" else "open_close"
        print(f"🔁 Morph order: {morph_order}")
        update_preview()

    elif key == ord('r'):
        use_region_restrict = not use_region_restrict
        print(f"🟩 Region restriction: {'ON' if use_region_restrict else 'OFF'}")
        update_preview()

    elif key == ord('q'):
        break

cv2.destroyAllWindows()

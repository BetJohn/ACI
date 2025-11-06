import cv2
import numpy as np

WINDOW_NAME = "Selectează culoarea pistei (click)"
PREVIEW_NAME = "Preview pista"
FILENAME = "./images/raw_image_2_formula_1.png"

# Load image
img = cv2.imread(FILENAME)
if img is None:
    raise ValueError("Nu am putut încărca imaginea. Verifică numele fișierului!")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
selected_color = None
kernel_size = 5

# Default tolerances
tol = 10
def update_preview():
    global mask, tol, selected_color
    b, g, r = map(int, selected_color)

    lower = np.array([
        max(b - tol, 0),
        max(g - tol, 0),
        max(r - tol, 0)
    ], dtype=np.uint8)
    upper = np.array([
        min(b + tol, 255),
        min(g + tol, 255),
        min(r + tol, 255)
    ], dtype=np.uint8)

    mask = cv2.inRange(img, lower, upper)


    k = max(1, kernel_size)  # avoid 0
    if k % 2 == 0:
        k += 1  # ensure odd size for symmetry
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow(PREVIEW_NAME, result)


def on_trackbar_change(val=None):
    global tol, kernel_size
    try:
        tol = cv2.getTrackbarPos("Color Tol", PREVIEW_NAME)
        kernel_size = cv2.getTrackbarPos("Erode Size", PREVIEW_NAME)
    except cv2.error:
        return
    if selected_color is not None:
        update_preview()


def mouse_callback(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color = img[y, x]
        print(f"Culoare selectată (BGR): {selected_color.tolist()}")
        update_preview()

cv2.namedWindow(WINDOW_NAME)
cv2.namedWindow(PREVIEW_NAME)


cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# Create sliders for tolerances
cv2.createTrackbar("Color Tol", PREVIEW_NAME, tol, 200, on_trackbar_change)
cv2.createTrackbar("Erode Size", PREVIEW_NAME, kernel_size, 20, on_trackbar_change)

cv2.imshow(WINDOW_NAME, img)
cv2.imshow(PREVIEW_NAME, img)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.resizeWindow(PREVIEW_NAME, 640, 360)
print("➡️ Click pe imagine pentru a selecta culoarea pistei.")
print("🎚 Ajustează slider-ele pentru a modifica toleranțele HSV.")
print("💾 Apasă 's' pentru a salva imaginea cu pista conturată.")
print("❌ Apasă 'q' pentru a ieși.")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoured = img.copy()
        cv2.drawContours(contoured, contours, -1, (0, 0, 255), 2)
        cv2.imwrite("track_contoured.jpg", contoured)
        print("✅ Imagine salvată ca track_contoured.jpg")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()

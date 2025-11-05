import cv2
import numpy as np

WINDOW_NAME = "Selectează culoarea pistei (click)"
PREVIEW_NAME = "Preview pista"
FILENAME = "./images/raw_image_2_formula_1.png"

img = cv2.imread(FILENAME)
if img is None:
    raise ValueError("Nu am putut încărca imaginea. Verifică numele fișierului!")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
selected_color = None

def update_preview(color_bgr):
    global mask
    # Asigurăm că e shape (1,1,3)
    color_bgr = np.uint8([[color_bgr]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = color_hsv

    lower = np.array([max(h - 10, 0), 40, 40])
    upper = np.array([min(h + 10, 179), 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    result = cv2.bitwise_and(img, img, mask=mask)

    combined = np.hstack((img, result))
    cv2.imshow(PREVIEW_NAME, combined)

def mouse_callback(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color = img[y, x]
        print(f"Culoare selectată (BGR): {selected_color.tolist()}")
        update_preview(selected_color)

cv2.namedWindow(WINDOW_NAME)
cv2.namedWindow(PREVIEW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

cv2.imshow(WINDOW_NAME, img)
cv2.imshow(PREVIEW_NAME, np.hstack((img, img)))

print("➡️ Click pe imagine pentru a selecta culoarea pistei.")
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

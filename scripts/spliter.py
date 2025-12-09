import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance
import matplotlib.pyplot as plt

def sort_skeleton_points(points):
    """
    Această funcție reordonează punctele astfel încât să fie consecutive pe traseu.
    Funcționează pe principiul "Nearest Neighbor" (Vecinul cel mai apropiat).
    """
    if len(points) == 0:
        return []

    # Începem cu primul punct găsit
    sorted_points = [points[0]]
    points = list(points)
    points.pop(0) # Îl scoatem din lista de vizitat

    while points:
        # Ultimul punct adăugat în traseu
        current_point = sorted_points[-1]
        
        # Găsim cel mai apropiat punct rămas nevizitat
        # (Calculăm distanța de la curent la toate celelalte)
        dists = distance.cdist([current_point], points)
        nearest_idx = np.argmin(dists)
        
        # Dacă cel mai apropiat punct e prea departe (ruptură în schelet), ne oprim sau sărim
        # Dar aici presupunem că scheletul e continuu.
        
        sorted_points.append(points[nearest_idx])
        points.pop(nearest_idx)
        
    return np.array(sorted_points)

def get_perpendicular_segments(image_path, step=10): # Am redus step la 10 pentru mai multe puncte
    # 1. Încărcare
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_bool = binary > 0

    # 2. Scheletizare
    skeleton = skeletonize(binary_bool)
    
    # 3. Extragere puncte brute (NEORDONATE - Aici era problema)
    y_coords, x_coords = np.where(skeleton)
    raw_points = list(zip(x_coords, y_coords))
    
    # --- FIX: ORDONĂM PUNCTELE PE TRASEU ---
    print(f"Ordonăm {len(raw_points)} puncte de schelet...")
    skeleton_points = sort_skeleton_points(raw_points)
    print("Ordonare completă.")
    
    output_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    segments_data = []

    # Window size pentru PCA
    window_size = 5

    # 4. Iterăm prin punctele ORDONATE
    for i in range(0, len(skeleton_points), step):
        cx, cy = skeleton_points[i]

        # PCA pentru Tangentă
        # Luăm vecini direct din lista ordonată (mult mai precis acum)
        # Ne uităm la câțiva indici în spate și în față în listă
        idx_start = max(0, i - window_size)
        idx_end = min(len(skeleton_points), i + window_size)
        neighbors = skeleton_points[idx_start:idx_end]
        
        if len(neighbors) < 3: continue

        # Calcul PCA
        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = np.dot(centered.T, centered)
        vals, vecs = np.linalg.eig(cov)
        tangent = vecs[:, np.argmax(vals)]
        tangent = tangent / np.linalg.norm(tangent)

        # Normala
        normal = np.array([-tangent[1], tangent[0]])

        # Ray Casting (rămâne la fel)
        def cast_ray(start_x, start_y, dx, dy):
            curr_x, curr_y = start_x, start_y
            steps = 0
            while steps < 500: # limită de siguranță
                next_x = int(curr_x + dx)
                next_y = int(curr_y + dy)
                if not (0 <= next_x < binary.shape[1] and 0 <= next_y < binary.shape[0]): break
                if binary[next_y, next_x] == 0: break
                curr_x, curr_y = next_x, next_y
                steps += 1
            return (int(curr_x), int(curr_y))

        p1 = cast_ray(cx, cy, normal[0], normal[1])
        p2 = cast_ray(cx, cy, -normal[0], -normal[1])

        # Desenare
        cv2.line(output_img, p1, p2, (0, 255, 0), 1)
        cv2.circle(output_img, (cx, cy), 2, (0, 0, 255), -1)

        width = np.linalg.norm(np.array(p1) - np.array(p2))
        segments_data.append([cx, cy, width])

    return output_img, segments_data

# Rulare
img_result, data = get_perpendicular_segments('track_binary1.png', step=1) # Step mai mic = mai multe puncte

if img_result is not None:
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title(f"Pista Segmentata Corect - {len(data)} segmente")
    plt.show()
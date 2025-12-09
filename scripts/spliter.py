import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

File_name = "../track_binary1.png"

def get_perpendicular_segments(image_path, step=15, window_size=3):
    # 1. Încărcare și pre-procesare
    # Citim imaginea ca Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Eroare: Nu am găsit imaginea.")
        return

    # Binarizare (asigurăm că avem doar 0 și 1/255)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Convertim la format boolean pentru skeletonize (True/False)
    binary_bool = binary > 0

    # 2. Scheletizare (Găsirea liniei mediane)
    # Aceasta funcție subțiază pista până la 1 pixel lățime
    skeleton = skeletonize(binary_bool)

    # Extragem coordonatele (y, x) ale tuturor punctelor de pe schelet
    y_coords, x_coords = np.where(skeleton)
    skeleton_points = list(zip(x_coords, y_coords))

    # Imagine color pentru vizualizare finală
    output_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    segments_data = []  # Aici vom stoca datele matematice

    # 3. Iterăm prin punctele scheletului
    # Nu luăm fiecare pixel, ci sărim peste 'step' pixeli
    for i in range(0, len(skeleton_points), step):
        cx, cy = skeleton_points[i]

        # --- A. Calculul Tangentei folosind PCA (Analiza Componentelor Principale) ---
        # Colectăm o vecinătate de pixeli din jurul punctului curent
        neighbors = []
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                ny, nx = cy + dy, cx + dx
                # Verificăm limitele imaginii
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx]:  # Dacă pixelul face parte din schelet
                        neighbors.append([nx, ny])

        neighbors = np.array(neighbors)

        if len(neighbors) < 3:
            continue  # Nu putem calcula direcția dacă suntem izolați

        # Calculăm vectorii proprii (Eigenvectors) ai covarianței
        # Aceasta ne dă direcția principală în care "curge" linia în acel punct
        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = np.dot(centered.T, centered)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Vectorul propriu corespunzător celei mai mari valori proprii este Tangenta
        tangent_index = np.argmax(eigenvalues)
        tangent = eigenvectors[:, tangent_index]

        # Normalizarea vectorului (lungime 1)
        tangent = tangent / np.linalg.norm(tangent)

        # --- B. Calculul Normalei (Perpendiculara) ---
        # Rotim tangenta cu 90 grade: (x, y) -> (-y, x)
        normal = np.array([-tangent[1], tangent[0]])

        # --- C. Ray Casting (Tragem liniile stânga-dreapta) ---
        # Funcție internă să "meargă" pe linie până dă de negru
        def cast_ray(start_x, start_y, direction_x, direction_y):
            curr_x, curr_y = start_x, start_y
            while True:
                next_x = int(curr_x + direction_x)
                next_y = int(curr_y + direction_y)

                # Verificăm dacă am ieșit din imagine
                if not (0 <= next_x < binary.shape[1] and 0 <= next_y < binary.shape[0]):
                    break
                # Verificăm dacă am lovit fundalul negru (marginea pistei)
                if binary[next_y, next_x] == 0:
                    break

                curr_x, curr_y = next_x, next_y
                return (int(curr_x), int(curr_y))

        # Căutăm marginea 1 (sensul normalei)
        p1 = cast_ray(cx, cy, normal[0], normal[1])
        # Căutăm marginea 2 (sensul opus normalei)
        p2 = cast_ray(cx, cy, -normal[0], -normal[1])

        # --- D. Desenare și Salvare ---
        # Desenăm linia verde (segmentul transversal)
        cv2.line(output_img, p1, p2, (0, 255, 0), 1)
        # Desenăm punctul central roșu
        cv2.circle(output_img, (cx, cy), 1, (0, 0, 255), -1)

        # Calculăm lățimea pistei în acest punct (Distanța Euclidiană)
        width = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        segments_data.append({
            "center": (cx, cy),
            "p1": p1,
            "p2": p2,
            "width": width,
            "normal_vector": normal.tolist()
        })

    return output_img, segments_data


# --- Rularea codului ---
# Înlocuiește 'track_binary1.png' cu numele exact al fișierului tău dacă e diferit
img_result, data = get_perpendicular_segments(File_name, step=20)

if img_result is not None:
    # Afișare cu Matplotlib (mai ușor în notebooks/IDE-uri)
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title(f"Pista Segmentata - {len(data)} segmente")
    plt.axis('off')
    plt.show()

    # Opțional: Salvare imagine
    cv2.imwrite("track_processed_python.png", img_result)
    print("Gata! Imaginea a fost procesata.")
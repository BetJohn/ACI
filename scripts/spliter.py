import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import networkx as nx
import csv  # <--- Added for CSV export

# --- GRAPH-BASED SORT FUNCTION (Unchanged) ---
def sort_skeleton_points_final(skeleton_img):
    y_coords, x_coords = np.where(skeleton_img)
    points = list(zip(x_coords, y_coords))
    
    if len(points) == 0: return np.array([])

    print(f"Building Graph from {len(points)} skeleton points...")
    G = nx.Graph()
    for p in points: G.add_node(p)
    
    points_set = set(points)
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    
    for x, y in points:
        for dx, dy in neighbors_offsets:
            neighbor = (x + dx, y + dy)
            if neighbor in points_set:
                G.add_edge((x, y), neighbor)

    print("Graph built. Finding cycles...")
    try:
        cycles = nx.cycle_basis(G)
    except nx.NetworkXNoCycle:
        return np.array(points)

    if not cycles:
        largest_cc = max(nx.connected_components(G), key=len)
        return np.array(list(largest_cc))

    longest_cycle_nodes = max(cycles, key=len)
    print(f"Longest loop found with {len(longest_cycle_nodes)} points.")

    H = G.subgraph(longest_cycle_nodes).copy()
    ordered_points = []
    if len(longest_cycle_nodes) > 0:
        start_node = longest_cycle_nodes[0]
        ordered_points = list(nx.dfs_preorder_nodes(H, source=start_node))
    
    return np.array(ordered_points)

# ------------------------------------------------------------

def get_perpendicular_segments(image_path):
    step = 13
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return None, []
        
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_bool = binary > 0

    skeleton = skeletonize(binary_bool)
    skeleton_points = sort_skeleton_points_final(skeleton)
    
    output_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    if len(skeleton_points) < 10:
        print("Not enough points to generate segments.")
        return output_img, []

    num_points = len(skeleton_points)
    window_size = 7

    # 1. STORAGE
    raw_segments = []

    for i in range(0, num_points, step):
        cx, cy = skeleton_points[i]

        # --- CIRCULAR INDEXING ---
        indices = [j % num_points for j in range(i - window_size, i + window_size + 1)]
        neighbors = skeleton_points[indices]
        
        if len(neighbors) < 3: continue

        # --- PCA ---
        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = np.dot(centered.T, centered)
        vals, vecs = np.linalg.eig(cov)
        tangent = vecs[:, np.argmax(vals)]
        tangent = tangent / np.linalg.norm(tangent)

        # --- DIRECTION CORRECTION ---
        flow_vector = neighbors[-1] - neighbors[0]
        dist = np.linalg.norm(flow_vector)
        if dist > 0:
            flow_vector = flow_vector / dist
            if np.dot(tangent, flow_vector) < 0:
                tangent = -tangent

        # --- NORMAL ---
        normal = np.array([-tangent[1], tangent[0]])

        # Ray Casting
        def cast_ray(start_x, start_y, dx, dy):
            curr_x, curr_y = start_x, start_y
            steps = 0
            while steps < 2000: 
                next_x = (curr_x + dx)
                next_y = (curr_y + dy)
                if not (0 <= next_x < binary.shape[1] and 0 <= next_y < binary.shape[0]): break
                if binary[int(next_y), int(next_x)] == 0: break
                curr_x, curr_y = next_x, next_y
                steps += 1
            return (int(curr_x), int(curr_y))

        p1 = cast_ray(cx, cy, normal[0], normal[1])
        p2 = cast_ray(cx, cy, -normal[0], -normal[1])

        raw_segments.append({
            'id': i,  # Keep track of original index
            'center': (cx, cy),
            'tangent': tangent,
            'p1': list(p1), 
            'p2': list(p2) 
        })

    # --- 2. ITERATIVE POST-PROCESSING CHECK ---
    iteration = 0
    max_iterations = 100
    count = len(raw_segments)

    while iteration < max_iterations:
        changes_made = False
        
        for k in range(count):
            curr_idx = k
            next_idx = (k + 1) % count
            
            curr_seg = raw_segments[curr_idx]
            next_seg = raw_segments[next_idx]
            
            ref_tangent = curr_seg['tangent']

            # Check Side 1 (p1)
            v1 = np.array(next_seg['p1']) - np.array(curr_seg['p1'])
            if np.dot(v1, ref_tangent) < -1e-5: 
                next_seg['p1'] = list(curr_seg['p1']) 
                changes_made = True

            # Check Side 2 (p2)
            v2 = np.array(next_seg['p2']) - np.array(curr_seg['p2'])
            if np.dot(v2, ref_tangent) < -1e-5:
                next_seg['p2'] = list(curr_seg['p2'])
                changes_made = True
        
        if not changes_made:
            print(f"Converged after {iteration} iterations.")
            break
            
        iteration += 1

    # --- 3. FINAL DRAWING & DATA PREP ---
    csv_data = [] # List to store all point data
    
    for idx, seg in enumerate(raw_segments):
        cx, cy = seg['center']
        tangent = seg['tangent']
        p1 = tuple(seg['p1'])
        p2 = tuple(seg['p2'])
        
        scale = 20 
        end_x = int(cx + tangent[0] * scale)
        end_y = int(cy + tangent[1] * scale)

        cv2.arrowedLine(output_img, (cx, cy), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3)
        cv2.line(output_img, p1, p2, (255, 0, 0), 1)

        width = np.linalg.norm(np.array(p1) - np.array(p2))
        
        # Add to CSV list: Index, CenterX, CenterY, P1_X, P1_Y, P2_X, P2_Y, Width
        csv_data.append([idx, cx, cy, p1[0], p1[1], p2[0], p2[1], width])

    return output_img, csv_data

# Run
img_result, track_data = get_perpendicular_segments('track_binary1.png')

# --- SAVE TO CSV ---
if track_data:
    csv_filename = "track_data.csv"
    print(f"Saving {len(track_data)} points to {csv_filename}...")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(["Index", "Center_X", "Center_Y", "P1_X", "P1_Y", "P2_X", "P2_Y", "Width_Px"])
        # Data
        writer.writerows(track_data)
        
    print("CSV save complete.")

if img_result is not None:
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title(f"Track Analysis - Saved to CSV")
    plt.axis('off')
    plt.show()
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import networkx as nx  # REQUIRES: pip install networkx

# --- NEW GRAPH-BASED SORT FUNCTION ---
def sort_skeleton_points_final(skeleton_img):
    """
    Parses the skeleton as a Graph, finds the longest Cycle (Loop),
    and returns the points ordered along that loop.
    """
    # 1. Get all skeleton points
    y_coords, x_coords = np.where(skeleton_img)
    points = list(zip(x_coords, y_coords))
    
    if len(points) == 0:
        return np.array([])

    print(f"Building Graph from {len(points)} skeleton points...")

    # 2. Build the Graph (Nodes = pixels, Edges = 8-connectivity)
    G = nx.Graph()
    
    # Add nodes
    for p in points:
        G.add_node(p)
    
    # Create a set for O(1) lookups
    points_set = set(points)
    
    # Add edges between 8-connected neighbors
    # We only check half the neighbors to avoid duplicate edges (undirected graph)
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    
    for x, y in points:
        for dx, dy in neighbors_offsets:
            neighbor = (x + dx, y + dy)
            if neighbor in points_set:
                G.add_edge((x, y), neighbor)

    print("Graph built. Finding cycles...")

    # 3. Find the Cycle Basis
    # cycle_basis finds the elementary cycles in the graph.
    try:
        cycles = nx.cycle_basis(G)
    except nx.NetworkXNoCycle:
        print("Error: No loops found in the skeleton.")
        return np.array(points) # Fallback to unsorted

    if not cycles:
        print("Warning: No closed loops found. Is the track broken?")
        # Fallback: Return the largest connected component if no cycle
        largest_cc = max(nx.connected_components(G), key=len)
        return np.array(list(largest_cc))

    # 4. Identify the Longest Loop
    # We assume the race track is the largest cycle in the image
    longest_cycle_nodes = max(cycles, key=len)
    
    print(f"Longest loop found with {len(longest_cycle_nodes)} points.")

    # 5. Order the points for smooth traversal
    # cycle_basis returns nodes, but we want to ensure they are ordered sequentially.
    # We create a subgraph of JUST the cycle and traverse it.
    H = G.subgraph(longest_cycle_nodes).copy()
    
    # Simple DFS/Traversal to order them
    ordered_points = []
    if len(longest_cycle_nodes) > 0:
        # Start from arbitrary node in the cycle
        start_node = longest_cycle_nodes[0]
        # dfs_preorder_nodes ensures we follow the path structure
        ordered_points = list(nx.dfs_preorder_nodes(H, source=start_node))
    
    return np.array(ordered_points)

# ------------------------------------------------------------

def get_perpendicular_segments(image_path):
    step = 1
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return None, []
        
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_bool = binary > 0

    skeleton = skeletonize(binary_bool)
    
    # --- USING THE NEW GRAPH FUNCTION ---
    skeleton_points = sort_skeleton_points_final(skeleton)
    
    output_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    segments_data = []
    
    if len(skeleton_points) < 10:
        print("Not enough points to generate segments.")
        return output_img, []

    window_size = 7

    for i in range(0, len(skeleton_points), step):
        cx, cy = skeleton_points[i]

        # PCA Tangent Calculation
        idx_start = max(0, i - window_size)
        idx_end = min(len(skeleton_points), i + window_size)
        neighbors = skeleton_points[idx_start:idx_end]
        
        # Determine wrapping for a closed loop (connect end to start)
        # If we are at the end, grab points from the beginning to smooth the transition
        if i + window_size >= len(skeleton_points):
            remaining = (i + window_size) - len(skeleton_points)
            neighbors = np.vstack((neighbors, skeleton_points[:remaining]))
        
        if len(neighbors) < 3: continue

        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = np.dot(centered.T, centered)
        vals, vecs = np.linalg.eig(cov)
        tangent = vecs[:, np.argmax(vals)]
        tangent = tangent / np.linalg.norm(tangent)

        # Normal vector
        normal = np.array([-tangent[1], tangent[0]])

        # Ray Casting
        def cast_ray(start_x, start_y, dx, dy):
            curr_x, curr_y = start_x, start_y
            steps = 0
            while steps < 2000: 
                next_x = int(curr_x + dx)
                next_y = int(curr_y + dy)
                if not (0 <= next_x < binary.shape[1] and 0 <= next_y < binary.shape[0]): break
                if binary[next_y, next_x] == 0: break
                curr_x, curr_y = next_x, next_y
                steps += 1
            return (int(curr_x), int(curr_y))

        p1 = cast_ray(cx, cy, normal[0], normal[1])
        p2 = cast_ray(cx, cy, -normal[0], -normal[1])

        cv2.line(output_img, p1, p2, (0, 255, 0), 1)
        cv2.circle(output_img, (cx, cy), 1, (0, 0, 255), -1)

        width = np.linalg.norm(np.array(p1) - np.array(p2))
        segments_data.append([int(cx), int(cy), float(width)])

    return output_img, segments_data

# Run
# Make sure to update 'track_binary1.png' to your actual file
img_result, data = get_perpendicular_segments('track_binary1.png')

if img_result is not None:
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title(f"Track Analysis - Longest Graph Loop ({len(data)} segments)")
    plt.axis('off')
    plt.show()
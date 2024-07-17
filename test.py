import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calculate_transition_matrices(points):
    n = len(points)
    dx_matrix = np.zeros((n, n))
    dy_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dx_matrix[i, j] = points[j, 0] - points[i, 0]
            dy_matrix[i, j] = points[j, 1] - points[i, 1]
    
    return dx_matrix, dy_matrix

def find_longest_path(points, dx_matrix, dy_matrix, start, dx_tol, dy_tol, max_depth):
    def backtrack(path, depth):
        if depth > max_depth:
            return []
        current_point = path[-1]
        current_idx = np.where((points == current_point).all(axis=1))[0][0]
        longest_path = path

        for next_idx, next_point in enumerate(points):
            if not any((next_point == p).all() for p in path):
                dx = dx_matrix[current_idx, next_idx]
                dy = dy_matrix[current_idx, next_idx]

                if abs(dx) <= dx_tol and abs(dy) <= dy_tol:
                    new_path = np.vstack([path, next_point])
                    candidate_path = backtrack(new_path, depth + 1)
                    if len(candidate_path) > len(longest_path):
                        longest_path = candidate_path

        return longest_path

    initial_path = np.array([points[start]])
    return backtrack(initial_path, 1)

def animate_path(points, dx_tol, dy_tol, start_index, max_depth):
    dx_matrix, dy_matrix = calculate_transition_matrices(points)
    longest_path = find_longest_path(points, dx_matrix, dy_matrix, start_index, dx_tol, dy_tol, max_depth)

    fig, ax = plt.subplots()
    x_range = max(points[:, 0]) - min(points[:, 0])
    y_range = max(points[:, 1]) - min(points[:, 1])
    margin_x = 0.05 * x_range
    margin_y = 0.05 * y_range

    ax.set_xlim(min(points[:, 0]) - margin_x, max(points[:, 0]) + margin_x)
    ax.set_ylim(min(points[:, 1]) - margin_y, max(points[:, 1]) + margin_y)
    
    scatter = ax.scatter(points[:, 0], points[:, 1], color='black')
    lines = {}

    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                line, = ax.plot([], [], 'k-', alpha=0.3)
                lines[(i, j)] = line

    def init():
        for line in lines.values():
            line.set_data([], [])
        return lines.values()

    def update(frame):
        path = frame[0]
        accepted = frame[1]
        
        for line in lines.values():
            line.set_data([], [])
        
        if len(path) > 1:
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i + 1]
                idx1 = np.where((points == p1).all(axis=1))[0][0]
                idx2 = np.where((points == p2).all(axis=1))[0][0]
                if (idx1, idx2) in lines:
                    line = lines[(idx1, idx2)]
                    line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    line.set_color('green' if accepted else 'black')
                    line.set_alpha(1 if accepted else 0.3)
        
        return lines.values()

    frames = []
    def collect_frames(path, depth):
        if depth > max_depth:
            return
        current_point = path[-1]
        current_idx = np.where((points == current_point).all(axis=1))[0][0]

        for next_idx, next_point in enumerate(points):
            if not any((next_point == p).all() for p in path):
                dx = dx_matrix[current_idx, next_idx]
                dy = dy_matrix[current_idx, next_idx]

                if abs(dx) <= dx_tol and abs(dy) <= dy_tol:
                    new_path = np.vstack([path, next_point])
                    frames.append((new_path, True))
                    collect_frames(new_path, depth + 1)
                    frames.append((path, False))

    collect_frames(np.array([points[start_index]]), 1)
    
    # Add the final longest path as the last frame
    frames.append((longest_path, True))

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False)

    plt.show()

# Example usage
points = np.random.rand(25, 2)
dx_tol = 0.2
dy_tol = 0.2
start_index = 0
max_depth = len(points)

animate_path(points, dx_tol, dy_tol, start_index, max_depth)

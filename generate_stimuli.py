import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import os
from scipy.spatial import distance_matrix



def normalize_coords(coordinates):


    coordinates[:, 0] -= np.min(coordinates[:, 0])
    coordinates[:, 1] -= np.min(coordinates[:, 1])
    if  np.max(coordinates[:,0]) > 0:
        coordinates[:, 0] /= np.max(coordinates[:, 0])
    else:
        coordinates[:, 0] += 0.5
    if   np.max(coordinates[:,1]) > 0:
        coordinates[:, 1] /= np.max(coordinates[:, 1])
    else:
        coordinates[:, 1] += 0.5

    return coordinates

def generate_coordinates(n, density, regularity, clustering, arrangement='grid'):

    # if regularity == 0:
    #     coordinates = np.random.rand(n, 2)
    #else:   
    if arrangement == 'grid':
        grid_size = int(np.ceil(np.sqrt(n)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, grid_size),
            np.linspace(0, 1, grid_size)
        )
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        chosen_indices = np.random.choice(grid_points.shape[0], n, replace=False)
        chosen_points = grid_points[chosen_indices]
    elif arrangement == 'line':
        chosen_points = np.column_stack((np.linspace(0, 1, n), np.zeros(n)+0.5))
    elif arrangement == 'circle':
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        chosen_points = np.column_stack((0.5 + 0.5 * np.cos(angles), 0.5 + 0.5 * np.sin(angles)))
    else:
        raise ValueError("Invalid arrangement. Choose from 'grid', 'line', or 'circle'.")

    #random_offsets = np.random.uniform(-0.5, 0.5, chosen_points.shape) / (grid_size if arrangement == 'grid' else int(np.ceil(np.sqrt(n))))
    random_offsets = []
    for i in range(len(chosen_points)):
        if arrangement == "line":
            random_offsets.append([np.random.uniform(-chosen_points[i][0], 1-chosen_points[i][0]), 0])

        else:
            random_offsets.append([np.random.uniform(-chosen_points[i][0], 1-chosen_points[i][0]), np.random.uniform(-chosen_points[i][1], 1-chosen_points[i][1])])
    random_offsets = np.array(random_offsets)

    coordinates = regularity * chosen_points + (1 - regularity) * (chosen_points + random_offsets)

    #coordinates = regularity * chosen_points + (1-regularity) * np.random.randn(n, 2)


    coordinates = normalize_coords(coordinates)

    if clustering > 0:
        dist_matrix = distance_matrix(coordinates, coordinates)
        new_coordinates = []
        for i in range(n):
            close_points = dist_matrix[i] <= (np.mean(dist_matrix) * clustering)
            nearby_points = coordinates[close_points]
            if len(nearby_points) > 1:
                centroid = np.mean(nearby_points, axis=0)
                new_coordinates.append(coordinates[i] + clustering * (centroid - coordinates[i]))
            else:
                new_coordinates.append(coordinates[i])
        coordinates = np.array(new_coordinates)

    coordinates = normalize_coords(coordinates)

    coordinates[:, 0] = coordinates[:, 0] * (1 - density) + density / 2
    coordinates[:, 1] = coordinates[:, 1] * (1 - density) + density / 2

    return coordinates






def generate_and_store_coordinates(n_values, density_values, regularity_values, clustering_values, arrangements, n_generate, output_file):

    coordinates_dict = {}
    
    idx = 0

    for _ in range(n_generate):
        for n in n_values:
            for density in density_values:
                for regularity in regularity_values:
                    for clustering in clustering_values:
                        for arrangement in arrangements:
                            print(idx, arrangement)
                            coordinates = generate_coordinates(n, density, regularity, clustering, arrangement)
                            coordinates_dict[f'set_{idx}'] = {
                                'params': {'n': n, 'density': density, 'regularity': regularity, 'clustering':clustering, "arrangement":arrangement},
                                'coordinates': coordinates
                            }
                            idx += 1
        
        with open(output_file, 'wb') as file:
            pickle.dump(coordinates_dict, file)



def plot_coordinates(coordinates_dict, output_folder='stimuli'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for key, value in coordinates_dict.items():
        coordinates = value['coordinates']
        params = value['params']
        
        plt.figure()
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue')
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.title(f'Coordinates ({params["arrangement"]}, n={params["n"]}, density={params["density"]}, regularity={params["regularity"]}, clustering={params["clustering"]})')

        plt.grid(False)
        
        plot_file = os.path.join(output_folder, f'{key}.png')
        plt.savefig(plot_file)
        plt.close()


n_values = [16,24,32]
density_values = [0, 0.25,0.5]
regularity_values = [0.5, 0.75,1]
clustering_values = [0.0,0.5,1]
arrangements = ["line", "grid", "circle"]

n_generate=1


output_file = 'output/coordinates.pkl'

generate_and_store_coordinates(n_values, density_values, regularity_values, clustering_values, arrangements, n_generate, output_file)

with open(output_file, 'rb') as file:
    coordinates_dict = pickle.load(file)

plot_coordinates(coordinates_dict)

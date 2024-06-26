import numpy as np
from sklearn.svm import SVC
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import pandas as pd


kernel = "rbf"


def load_coordinates(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def generate_points(num_points):
    return np.random.rand(num_points, 2)


def update_svm(points, true_labels, cost_param ):
    if len(np.unique(true_labels)) < 2:
        return None

    svm = SVC(kernel=kernel, C=cost_param, probability=True)
    svm.fit(points, true_labels)

    return svm



def get_corner_point_index(points):
    num_points = len(points)
    labels = np.zeros(num_points)
    
    #corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    corners = np.array([[0,1]])
    distances = np.min(np.linalg.norm(points[:, np.newaxis] - corners, axis=2), axis=1)

    
    closest_point_index = np.argmin(distances)
    return closest_point_index

def pick_next_point(svm, points, predicted_labels, last_touched_indices):

    if len(last_touched_indices) == 0:
        return get_corner_point_index(points)
    
    untouched_indices = np.where(predicted_labels == 0)[0]

    untouched_points = points[untouched_indices]
    
    last_touched_point = points[last_touched_indices[-1]]
    distance_prev = np.linalg.norm(untouched_points - last_touched_point, axis=1)
    distance_prev_n = np.linalg.norm(untouched_points - np.mean(points[last_touched_indices], axis=0), axis=1)
    df = -svm.decision_function(untouched_points)

    scores = -(distance_prev + distance_prev_n +  df)  #dumb way of picking next points
    
    sorted_indices = np.argsort(np.abs(scores))
    
    for idx in sorted_indices:
        candidate_index = untouched_indices[idx]
        if candidate_index not in last_touched_indices:
            return candidate_index
    return untouched_indices[sorted_indices[0]]



def run_model(points, cost_param, memory_size):
    num_points = len(points)
    true_labels = np.zeros(num_points)
    predicted_labels = copy.deepcopy(true_labels)
    last_touched_indices = []
    history = []
    touch_order = []  

    xx, yy = np.meshgrid(np.linspace(0, 1, 100),
                         np.linspace(0, 1, 100))
    Z = np.zeros(xx.shape)

    svm=None

    while not np.all(predicted_labels == 1) and (len(touch_order) < num_points*2):
        
        next_point_index = pick_next_point(svm, points, predicted_labels, last_touched_indices)
        if next_point_index is None:
            break

        
        true_labels[next_point_index] = 1

        last_touched_indices.append(next_point_index)
        if len(last_touched_indices) > memory_size:
            last_touched_indices.pop(0)

        for i in range(len(last_touched_indices)):
            predicted_labels[last_touched_indices[i]] = 1



        if not np.all(predicted_labels == 1):

            svm = update_svm(points, predicted_labels, cost_param)
            predicted_labels[:] = svm.predict(points)

            Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

        else:
            Z = np.ones(xx.shape)


        
        history.append((points.copy(), true_labels.copy(), xx, yy, Z))
        touch_order.append(next_point_index)
    
    return true_labels, touch_order, history

def calculate_error_rate(history):

    final_labels = history[-1][0]
    errors = np.sum(final_labels != 1)
    error_rate = errors / len(history)
    return error_rate


def store_results(coordinates_dict, cost_param, memory_size=5):
    results = []

    for key, value in coordinates_dict.items():
        points = value['coordinates']
        params = value['params']

        true_labels, touch_order, history = run_model(points, cost_param, memory_size)
        print(params)
        print(len(points), len(history))
        print("")
        for i, idx in enumerate(touch_order):
            results.append({
                'set_id': key,
                'x': points[idx][0],
                'y': points[idx][1],
                'touched': 1,
                'touch_time': i,
                'n': params['n'],
                'density': params['density'],
                'regularity': params['regularity'],
                'clustering':params['clustering'],
                'arrangement':params["arrangement"]
            })
        
        # Record untouched points
        untouched_indices = np.where(true_labels == 0)[0]
        for idx in untouched_indices:
            results.append({
                'set_id': key,
                'x': points[idx][0],
                'y': points[idx][1],
                'touched': 0,
                'touch_time': np.nan,
                'n': params['n'],
                'density': params['density'],
                'regularity': params['regularity'],
                'clustering':params['clustering'],
                'arrangement':params["arrangement"]


            })
        
        error_rate = calculate_error_rate(history)
        for result in results:
            if result['set_id'] == key:
                result['error_rate'] = error_rate
    
    df = pd.DataFrame(results)
    df.to_csv(f"output/results_{kernel}.csv", index=False)

coordinates_dict = load_coordinates('output/coordinates.pkl')

memory_size = 4
cost_param = 5
store_results(coordinates_dict, cost_param, memory_size)

all_histories = []
all_touch_orders = []
metadata = []

for key, value in list(coordinates_dict.items())[::17]:
    points = value["coordinates"]
    params = value['params']
    true_labels, touch_order, history = run_model(points, cost_param, memory_size)

    all_histories.extend(history)


    for i in range(len(history)):
        all_touch_orders.extend([touch_order[:i+1]])
    metadata.extend([{'key': key, 'params': params}] * len(history))

total_frames = len(all_histories)



fig, ax = plt.subplots()

def init():
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    return ax,

def update(frame):
    ax.clear()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)



    points, true_labels, xx, yy, Z = all_histories[frame]
    current_metadata = metadata[frame]
    touch_order = all_touch_orders[frame]

    touched_points = points[true_labels == 1]
    ax.scatter(touched_points[:, 0], touched_points[:, 1], c='blue', label='Counted')
    
    untouched_points = points[true_labels == 0]
    if len(untouched_points) > 0:
        ax.scatter(untouched_points[:, 0], untouched_points[:, 1], c='red', label='Uncounted', alpha=0.5)
    


    if Z.max() == 0:
        ax.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(Z.min()-0.001, Z.max(), 3), colors=["red", "red"])
    else:

        ax.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(Z.min()-0.001, Z.max(), 3), colors=["red", "blue"])
    
    # Plot fixation path using touch_order
    if frame > 0:
        for i in range(1, len(touch_order)):
            start_idx = touch_order[i-1]
            end_idx = touch_order[i]
            start_point = points[start_idx]
            end_point = points[end_idx]
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-', alpha=i/len(touch_order))


    print(current_metadata["params"])

    ax.set_title(f'(n={current_metadata["params"]["n"]}, '
                 f'density={current_metadata["params"]["density"]}, '
                 f'regularity={current_metadata["params"]["regularity"]}, '
                 f'clustering={current_metadata["params"]["clustering"]})')

    ax.legend()
    return ax,

ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init,
                              blit=False, interval=400, repeat=False)
# ani.save(f'output/fixation_animation_{kernel}.gif', writer='pillow', fps=4)

# plt.show()





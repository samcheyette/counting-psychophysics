import os
import matplotlib
from copy import deepcopy
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def normalize(v):
    return [k/np.sum(k) for k in v]

def generate_stimuli(n_points, true_clusters, overlap_factor):
    points = []
    means = np.random.rand(true_clusters, 2) * (1 - overlap_factor) + overlap_factor / 2
    sds = [0.1 for _ in range(true_clusters)]

    initial_cluster_assignment = []

    for i in range(n_points):
        idx = i % true_clusters
        mean, sd = means[idx], sds[idx]
        point = np.random.normal(mean, sd)
        point[0] = min(max(point[0], 0), 1)
        point[1] = min(max(point[1], 0), 1)
        points.append(point)
        initial_cluster_assignment.append(idx)

    points = np.array(points)
    return points, initial_cluster_assignment, means, sds

def find_clusters_DPMM(points, N):
    DPMM = BayesianGaussianMixture(
        n_components=min(N, len(points)),
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1
    )

    DPMM.fit(points)
    labels = DPMM.predict(points)

    clusters = []
    unique_labels = np.unique(labels)
    for l in unique_labels:
        clust = np.where(labels == l)[0].tolist()
        clusters.append(clust)

    means = DPMM.means_[unique_labels]
    covs = DPMM.covariances_[unique_labels]

    return clusters, means, covs, DPMM

def reassign_point(pt, new_cluster_num, clusters, points):
    clusters = deepcopy(clusters)
    for c in clusters:
        if pt in c:
            c.remove(pt)
            break

    new_cluster = clusters[new_cluster_num].copy()
    new_cluster.append(pt)
    clusters[new_cluster_num] = sorted(new_cluster)

    cluster_means = []
    cluster_covs = []
    for c in clusters:
        cluster_points = points[c]
        if len(cluster_points) > 0:
            cluster_means.append(np.mean(cluster_points, axis=0))
            cov = np.eye(2) * 1e-6 + np.cov(cluster_points, rowvar=False)
            cluster_covs.append(cov)

    return clusters, cluster_means, cluster_covs

def resample_point_cluster(pt_num, cluster_means, cluster_covs, points):
    pt = points[pt_num]
    likelihoods = []
    for mean, cov in zip(cluster_means, cluster_covs):
        lik = multivariate_normal(mean, cov).pdf(pt)
        likelihoods.append(lik)

    likelihoods = np.array(likelihoods) / np.sum(likelihoods)

    return np.argmax(np.random.multinomial(1, likelihoods))

def pick_next_cluster(clusters, visited_clusters):
    for idx, cluster in enumerate(clusters):
        if idx not in visited_clusters:
            return idx
    return -1

def run_model_on_stimuli(points, initial_cluster_assignment, subitizing_range, within_cluster_forgetting,
                         max_clusters, overlap_factor, output_file, count=0):
    records = []
    records_at_time = []

    cluster_members, cluster_means, cluster_covs, DPMM = find_clusters_DPMM(points, max_clusters)
    init_cluster_members = deepcopy(cluster_members)

    visited_points = []
    visited_clusters = set()
    current_cluster = pick_next_cluster(cluster_members, visited_clusters)
    current_cluster_points = deepcopy(cluster_members[current_cluster]) if current_cluster != -1 else []
    visited_in_cluster = []

    fixation_number = 0

    while current_cluster != -1 and (fixation_number < 2 * len(points)):
        if len(current_cluster_points) > 0:
            current_fixation_idx = np.random.choice(current_cluster_points)





            visited_points.append(current_fixation_idx)
            current_cluster_points.remove(current_fixation_idx)
            visited_in_cluster.append(current_fixation_idx)

            if len(cluster_members[current_cluster]) > subitizing_range:
                for pt in cluster_members[current_cluster]:
                    if np.random.random() < within_cluster_forgetting:
                        if pt in current_cluster_points:
                            current_cluster_points.remove(pt)
                            visited_in_cluster.append(pt)
                        elif (pt in visited_in_cluster):
                            if (pt != visited_in_cluster[-1]):
                                current_cluster_points.append(pt)
                                visited_in_cluster.remove(pt)
                        else:
                            print("this shouldn't happen...")
                            assert False

            record_at_time = []
            for point_idx in range(len(points)):
                assigned_cluster = next((i for i, cluster in enumerate(cluster_members) if point_idx in cluster), -1)
                initial_cluster = next((i for i, cluster in enumerate(init_cluster_members) if point_idx in cluster), -1)

                d = {
                    'id': count,
                    'subit_range': subitizing_range,
                    'within_cluster_forgetting': within_cluster_forgetting,
                    'fixation_number': fixation_number,
                    'point_idx': point_idx,
                    'x': points[point_idx, 0],
                    'y': points[point_idx, 1],
                    'current_fixation_idx': current_fixation_idx,
                    'current_x': points[current_fixation_idx, 0],
                    'current_y': points[current_fixation_idx, 1],
                    'initial_cluster': initial_cluster,
                    'current_cluster': assigned_cluster,
                    'true_cluster': initial_cluster_assignment[point_idx],
                    'total_points': len(points),
                    'n_true_clusters': len(set(initial_cluster_assignment)),
                    'overlap_factor': overlap_factor
                }

                records.append(d)
                record_at_time.append(d)


            records_at_time.append(record_at_time)
            fixation_number += 1
        else:
            visited_clusters.add(current_cluster)
            current_cluster = pick_next_cluster(cluster_members, visited_clusters)
            if current_cluster != -1:
                for point_idx in range(len(points)):
                    new_assn = resample_point_cluster(point_idx, cluster_means, cluster_covs, points)
                    cluster_members, cluster_means, cluster_covs = reassign_point(point_idx, new_assn, cluster_members, points)

                current_cluster_points = deepcopy(cluster_members[current_cluster])
                visited_in_cluster = []

    df = pd.DataFrame(records)
    file_exists = os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)


    return records_at_time


def animate(records_at_time, points, anim_count):
    fig, ax = plt.subplots()
    colors = matplotlib.colormaps.get_cmap('tab10').resampled(max(record['current_cluster'] for records in records_at_time for record in records) + 1)

    def init():
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        return ax,

    def update(frame):
        ax.clear()
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        frame_records = records_at_time[frame]
        current_fixation_idx = frame_records[0]['current_fixation_idx']


        # Draw the clusters
        for cluster_idx in range(max(record['current_cluster'] for records in records_at_time for record in records) + 1):
            cluster_points = points[[record['point_idx'] for record in frame_records if record['current_cluster'] == cluster_idx]]
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(cluster_idx), label=f'Cluster {cluster_idx + 1}')
                centroid = np.mean(cluster_points, axis=0)
                radius = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
                circle = plt.Circle(centroid, radius, color=colors(cluster_idx), fill=False, linestyle='--')
                ax.add_artist(circle)

        # Draw the fixation path
        if frame > 0:
            for i in range(1, frame + 1):
                prev_fixation_idx = records_at_time[i - 1][0]['current_fixation_idx']
                current_fixation_idx = records_at_time[i][0]['current_fixation_idx']
                start_point = points[prev_fixation_idx]
                end_point = points[current_fixation_idx]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-')


        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(records_at_time), init_func=init, blit=False, interval=500, repeat=False)
    animation_output_path = f"animations/fixation_animation_{anim_count}.gif"
    ani.save(animation_output_path, writer='imagemagick', fps=2)
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    n_points_list = [8, 12, 16, 24]
    true_clusters_list = [1, 2, 4]
    overlap_factors = [0, 0.5]

    subitizing_range, within_cluster_forgetting = 4, 0.05

    output_file = 'output/fixation_model_output.csv'

    if os.path.exists(output_file):
        os.remove(output_file)

    anim_count = 0

    for _ in range(20):
        for n_points in n_points_list:
            for true_clusters in true_clusters_list:
                for overlap_factor in overlap_factors:

                    points, initial_cluster_assignment, means, sds = generate_stimuli(n_points, true_clusters, overlap_factor)
                    records = run_model_on_stimuli(points, initial_cluster_assignment, subitizing_range, within_cluster_forgetting,
                                                   max_clusters=10, overlap_factor=overlap_factor, output_file=output_file,
                                                   count=anim_count)
                    print(f"N: {n_points}, true_clusters: {true_clusters}, counted: {records[-1][-1]['fixation_number']}")
                    print("="*50 + "\n")

                    #animate(records, points, anim_count)
                    anim_count += 1
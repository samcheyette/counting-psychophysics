import os
import matplotlib
from copy import deepcopy
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_stimuli(n_points, true_clusters, overlap_factor):
    points = []
    means = np.random.rand(true_clusters, 2)
    sds = overlap_factor * (0.1 + 0.25 * np.random.rand(true_clusters))

    initial_cluster_assignment = []

    for i in range(n_points):
        idx = np.random.randint(0, len(means))
        mean, sd = means[idx], sds[idx]
        point = np.random.normal(mean, sd)
        point[0] = min(max(point[0], 0), 1)
        point[1] = min(max(point[1], 0), 1)
        points.append(point)
        initial_cluster_assignment.append(idx)

    points = np.array(points)
    return points, initial_cluster_assignment, means, sds

def find_clusters_DPMM(points, N, epsilon):
    DPMM = BayesianGaussianMixture(
        n_components=N,
        weight_concentration_prior_type='dirichlet_distribution',
        weight_concentration_prior=1000
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

def run_model_on_stimuli(points, true_cluster_assignments, max_clusters, overlap_factor, output_file, animate=False):
    records = []


    cluster_members, cluster_means, cluster_covs, DPMM = find_clusters_DPMM(points, max_clusters, epsilon=0.5)
    init_cluster_members = deepcopy(cluster_members)

    visited_points = []
    visited_clusters = set()
    current_cluster = pick_next_cluster(cluster_members, visited_clusters)
    current_cluster_points = deepcopy(cluster_members[current_cluster]) if current_cluster != -1 else []
    current_point_idx = 0
    fixation_idx = 0

    def animate_fixation():
        fig, ax = plt.subplots()
        colors = matplotlib.colormaps.get_cmap('tab10').resampled(len(cluster_members))  # fix mpl deprecation

        def init():
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            return ax,

        def update(frame):
            nonlocal cluster_members, cluster_means, cluster_covs, visited_points, visited_clusters
            nonlocal current_cluster, current_cluster_points, current_point_idx

            ax.clear()
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)

            if current_point_idx < len(current_cluster_points):
                pt = current_cluster_points[current_point_idx]
                visited_points.append(pt)

                new_assn = resample_point_cluster(pt, cluster_means, cluster_covs, points)
                cluster_members, cluster_means, cluster_covs = reassign_point(pt, new_assn, cluster_members, points)

                current_point_idx += 1
            else:
                visited_clusters.add(current_cluster)
                current_cluster = pick_next_cluster(cluster_members, visited_clusters)
                if current_cluster != -1:
                    current_cluster_points = deepcopy(cluster_members[current_cluster])
                    current_point_idx = 0

            # Draw the clusters
            for idx, cluster in enumerate(cluster_members):
                cluster_points = points[cluster]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(idx), label=f'Cluster {idx + 1}')
                centroid = np.mean(cluster_points, axis=0)
                radius = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
                circle = plt.Circle(centroid, radius, color=colors(idx), fill=False, linestyle='--')
                ax.add_artist(circle)

            # Draw the fixation path
            for i in range(1, len(visited_points)):
                start_point = points[visited_points[i-1]]
                end_point = points[visited_points[i]]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-')

            return ax,

        ani = animation.FuncAnimation(fig, update, frames=len(points) * 3, init_func=init, blit=False, interval=500, repeat=False)
        plt.show()

    while current_cluster != -1:
        if current_point_idx < len(current_cluster_points):
            pt = current_cluster_points[current_point_idx]
            visited_points.append(pt)

            new_assn = resample_point_cluster(pt, cluster_means, cluster_covs, points)
            cluster_members, cluster_means, cluster_covs = reassign_point(pt, new_assn, cluster_members, points)

            current_point_idx += 1
            for point_idx in range(len(points)):
                assigned_cluster = next((i for i, cluster in enumerate(cluster_members) if point_idx in cluster), -1)
                initial_cluster = next((i for i, cluster in enumerate(init_cluster_members) if point_idx in cluster), -1)

                records.append({
                    'fixation': fixation_idx,
                    'point_idx': point_idx,
                    'x': points[point_idx, 0],
                    'y': points[point_idx, 1],
                    'initial_cluster': initial_cluster,
                    'current_cluster': assigned_cluster,
                    'true_cluster': true_cluster_assignments[point_idx],
                    'total_points': len(points),

                    'n_true_clusters': len(set(initial_cluster_assignment)),
                    'overlap_factor': overlap_factor
                })

            fixation_idx += 1

        else:
            visited_clusters.add(current_cluster)
            current_cluster = pick_next_cluster(cluster_members, visited_clusters)
            if current_cluster != -1:
                current_cluster_points = deepcopy(cluster_members[current_cluster])
                current_point_idx = 0



    df = pd.DataFrame(records)

    file_exists = os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)
    print(f"Data saved to {output_file}")

    if animate:
        animate_fixation()

if __name__ == "__main__":
    n_points_list = [20, 40, 60] 
    true_clusters_list = [2, 4, 6]  
    overlap_factors = [0.1, 0.5, 1.0] 

    output_file = 'fixation_model_output.csv'

    if os.path.exists(output_file):
        os.remove(output_file)

    for n_points in n_points_list:
        for true_clusters in true_clusters_list:
            for overlap_factor in overlap_factors:
                points, initial_cluster_assignment, means, sds = generate_stimuli(n_points, true_clusters, overlap_factor)
                run_model_on_stimuli(points, initial_cluster_assignment, max_clusters=10, overlap_factor=overlap_factor, output_file=output_file, animate=True)

import matplotlib
from copy import deepcopy
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier


def get_corner_point_index(points):
    num_points = len(points)
    labels = np.zeros(num_points)
    
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    distances = np.min(np.linalg.norm(points[:, np.newaxis] - corners, axis=2), axis=1)

    
    closest_point_index = np.argmin(distances)
    return closest_point_index

class MemoryModel:
    def __init__(self, n_neighbors=1, memory_size=3):
        self.n_neighbors = n_neighbors
        self.memory_size = memory_size
        self.clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.centroids = None
        self.labels = None
        self.beliefs = None
        self.recent_memory = []

    def initialize(self, centroids):
        self.centroids = centroids
        self.labels = np.zeros(len(centroids))
        self.beliefs = np.zeros(len(centroids))

    def remember(self, cluster_id):
        self.labels[cluster_id] = 1
        self.recent_memory.append(cluster_id)

        if len(self.recent_memory) > self.memory_size:
            forgotten_item = self.recent_memory.pop(0)
            print("Forgot " + str(forgotten_item))


        if len(np.unique(self.labels)) > 1:
            self.clf.fit(self.centroids, self.labels)
            self.beliefs = self.clf.predict(self.centroids)

        for l in self.recent_memory:
            self.beliefs[l] = 1

        # print("---\n")
        # print(self.labels)
        # print(self.beliefs)
        # print("---\n")

    def has_visited(self, k):
        return self.beliefs[k] == 1

def calculate_distances(points):
    num_points = points.shape[0]
    distance_matrix = np.zeros((num_points, num_points))
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix

# def find_clusters(points, N, epsilon):
#     distance_matrix = calculate_distances(points)
#     D_average = np.mean(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])

#     threshold_distance = epsilon * D_average

#     clusters = []
#     visited = set()

#     for i in range(len(points)):
#         if i in visited:
#             continue

#         cluster = [i]
#         visited.add(i)

#         for j in range(len(points)):
#             if j not in visited and len(cluster) < N:
#                 if all(distance_matrix[j][k] < threshold_distance for k in cluster):
#                     cluster.append(j)
#                     visited.add(j)

#         clusters.append(cluster)

#     return clusters

def find_clusters_DPMM(points, N, epsilon):

    DPMM = BayesianGaussianMixture(
        n_components=N,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=10e4
    )

    DPMM.fit(points)
    labels = DPMM.predict(points)

    clusters = []
    unique_labels = np.unique(labels)
    for l in unique_labels:
        clust = np.where(labels==l)[0].tolist()
        clusters.append(clust)

    means = DPMM.means_[unique_labels]
    covs = DPMM.covariances_[unique_labels]

    return clusters, means, covs, DPMM

# def find_clusters(points, N, epsilon):
#     # from pgsm import
#     from pgsm.distributions.mvn import MultivariateNormalDistribution
#     from pgsm.partition_priors import DirichletProcessPartitionPrior
#     from pgsm.mcmc.collapsed_gibbs import CollapsedGibbsSampler
#     from pgsm.mcmc.particle_gibbs_split_merge import ParticleGibbsSplitMergeSampler
#     from pgsm.mcmc.split_merge_setup import UniformSplitMergeSetupKernel
#     from pgsm.mcmc.concentration import GammaPriorConcentrationSampler
#     dim = 2
#     init_concentration = 1.0

#     partition_prior = DirichletProcessPartitionPrior(init_concentration)
#     dist = MultivariateNormalDistribution(dim)
#     DPMM = CollapsedGibbsSampler(dist, partition_prior)


#     setup_kernel = UniformSplitMergeSetupKernel(points, dist, partition_prior)
#     pgsm_sampler = ParticleGibbsSplitMergeSampler.create_from_dist(dist, partition_prior, setup_kernel, num_anchors=2)
#     conc_sampler = GammaPriorConcentrationSampler(1, 1)
#     n_points = points.shape[0]
#     clustering = np.zeros(n_points)

#     for it in range(10000):
#         if it % 10 == 0:
#             print(it)
#             print(clustering)
#         # clustering = DPMM.sample(clustering, points)
#         num_clusters = len(np.unique(clustering))
#         pgsm_sampler.sample(clustering, points)
#         partition_prior.alpha = conc_sampler.sample(partition_prior.alpha, num_clusters, n_points)
#     import pdb; pdb.set_trace()


def pick_next_cluster(memory_model, centroids, current_cluster):
    distances = np.linalg.norm(centroids - centroids[current_cluster], axis=1)
    visited_status = [memory_model.has_visited(i) for i in range(len(centroids))]
    scores = np.array([distances[i] if not visited_status[i] else np.inf for i in range(len(centroids))])

    next_cluster = np.argmin(scores)
    
    if scores[next_cluster] == np.inf:
        return -1
    
    return next_cluster

def find_path_through_clusters(points, clusters, memory_model):
    centroids = np.array([np.mean(points[cluster], axis=0) for cluster in clusters])

    memory_model.initialize(centroids)
    
    current_cluster = get_corner_point_index(centroids)
    cluster_path = [current_cluster]
    visited_clusters = set(cluster_path)
    memory_model.remember(current_cluster)
    
    path_memory = []
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    
    if len(np.unique(memory_model.labels)) > 1:
        Z = memory_model.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    else:
        Z = np.zeros_like(xx)
    

    for _ in clusters[current_cluster]:
        path_memory.append((xx, yy, Z))
    
    while len(cluster_path) < len(clusters):
        next_cluster = pick_next_cluster(memory_model, centroids, current_cluster)
        if next_cluster == -1:
            print("No unvisited clusters found. Terminating path search.")
            break
        if next_cluster == current_cluster:
            print("Next cluster is the same as current cluster. Breaking loop to avoid infinite loop.")
            break
        cluster_path.append(next_cluster)
        visited_clusters.add(next_cluster)
        memory_model.remember(next_cluster)
        current_cluster = next_cluster
        
        if len(np.unique(memory_model.beliefs)) > 1:
            Z = memory_model.clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
        else:
            if np.max(memory_model.beliefs) == 0:   
                Z = np.zeros_like(xx)
            else:
                Z = np.ones_like(xx)

        for _ in clusters[next_cluster]: 
            path_memory.append((xx, yy, Z))

    full_path = []
    for cluster_index in cluster_path:
        full_path.extend(clusters[cluster_index])
    
    return cluster_path, full_path, path_memory

def animate_clusters(points, clusters, cluster_path, full_path, path_memory):
    """Create an animation of the clustering process."""
    fig, ax = plt.subplots()
    colors = matplotlib.colormaps.get_cmap('tab10').resampled(len(clusters))  # fix mpl deprecation
    full_path_points = points[full_path]

    def init():
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        return ax,

    def update(frame):
        ax.clear()
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        for idx, cluster in enumerate(clusters):
            cluster_points = points[cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(idx), label=f'Cluster {idx + 1}')
            centroid = np.mean(cluster_points, axis=0)
            radius = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
            circle = plt.Circle(centroid, radius, color=colors(idx), fill=False, linestyle='--')
            ax.add_artist(circle)

        xx, yy, Z = path_memory[frame]
        if Z.min() != Z.max():
            levels = np.linspace(Z.min(), Z.max(), 3)
            ax.contourf(xx, yy, Z, alpha=0.3, levels=levels, colors=["red", "blue"])

        else:
            levels = [Z.min()-0.001, Z.max()+0.001] 
            if Z.max() == 0:
                ax.contourf(xx, yy, Z, alpha=0.3, levels=levels, colors=["blue", "blue"])
            else:
                ax.contourf(xx, yy, Z, alpha=0.3, levels=levels, colors=["red", "red"])

        if frame > 0:
            ax.plot(full_path_points[:frame+1, 0], full_path_points[:frame+1, 1], 'k--')
        
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(path_memory), init_func=init, blit=False, interval=500, repeat=True)
    plt.show()

def animate_reassignment(points, clusters, cluster_means, cluster_covs):
    fig, ax = plt.subplots()
    colors = matplotlib.colormaps.get_cmap('tab10').resampled(len(clusters))  # fix mpl deprecation

    def init():
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        return ax,

    def update(frame):
        ax.clear()
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        pt = np.random.randint(points.shape[0])

        new_assn = resample_point_cluster(pt, cluster_means, cluster_covs)
        # DOES NOT YET RECALCULATE MEAN AND COV

        recluster, _, _ = reassign_point(pt, new_assn, clusters)
        print(recluster)

        for idx, cluster in enumerate(recluster):
            cluster_points = points[cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(idx), label=f'Cluster {idx + 1}')
            centroid = np.mean(cluster_points, axis=0)
            radius = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
            circle = plt.Circle(centroid, radius, color=colors(idx), fill=False, linestyle='--')
            ax.add_artist(circle)

        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(path_memory), init_func=init, blit=False, interval=500, repeat=True)
    plt.show()


def reassign_point(pt, new_cluster_num, clusters):
    # import pdb; pdb.set_trace()

    print(clusters)
    clusters = deepcopy(clusters)
    for c in clusters:
        if pt in c:
            break
    else:
        raise Exception('point in no cluster')
    c.remove(pt)
    new_cluster = clusters[new_cluster_num].copy()
    new_cluster.append(pt)
    clusters[new_cluster_num] = sorted(new_cluster)

    cluster_means = []
    cluster_covs = []
    for c in clusters:
        cluster_means.append(np.array(np.mean(points[c, 0]), np.mean(points[c, 1])))
        cluster_covs.append(np.cov(points[c]))

    return clusters, cluster_means, cluster_covs

def resample_point_cluster(pt_num, cluster_means, cluster_covs):
    pt = points[pt_num]
    likelihoods = []
    for mean, cov in zip(cluster_means, cluster_covs):
        lik = multivariate_normal(mean, cov).pdf(pt)
        likelihoods.append(lik)

    likelihoods = np.array(likelihoods) / np.sum(likelihoods)

    # print(likelihoods)
    return np.argmax(np.random.multinomial(1, likelihoods))


points = np.random.rand(20, 2)
N = 10                          # sklearn's implementation uses N as the max # of clusters
epsilon = 0.5

cluster_members, cluster_means, cluster_covs, DPMM = find_clusters_DPMM(points, N, epsilon)
# print("Clusters:", clusters)

memory_model = MemoryModel(n_neighbors=2, memory_size=4)
cluster_path, full_path, path_memory = find_path_through_clusters(points, cluster_members, memory_model)

# print(sorted(full_path))
# animate_clusters(points, clusters, cluster_path, full_path, path_memory)

animate_reassignment(points, cluster_members, cluster_means, cluster_covs)

import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


def flatten_weights(weights):

    flat = []

    for layer in weights:

        flat.extend(layer.flatten())

    return np.array(flat)


def compute_similarity_matrix(client_weights):

    client_ids = list(client_weights.keys())

    flattened = []

    for cid in client_ids:

        flattened.append(
            flatten_weights(client_weights[cid])
        )

    flattened = np.array(flattened)

    similarity_matrix = cosine_similarity(flattened)

    return client_ids, similarity_matrix


def cluster_clients(client_weights, num_clusters=3):

    client_ids, similarity_matrix = compute_similarity_matrix(
        client_weights
    )

    distance_matrix = 1 - similarity_matrix

    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric="precomputed",
        linkage="average"
    )

    labels = clustering.fit_predict(distance_matrix)

    clusters = {}

    for cid, label in zip(client_ids, labels):

        if label not in clusters:

            clusters[label] = []

        clusters[label].append(cid)

    return clusters
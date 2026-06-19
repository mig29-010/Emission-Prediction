import flwr as fl
import numpy as np
from clustering import cluster_clients

def average_weights(weight_list):
    # (Move your average_weights function from main.py to here, 
    # since the strategy needs to use it for the clusters)
    avg_weights = []
    for layer_idx in range(len(weight_list[0])):
        layer_avg = np.mean(
            [w[layer_idx] for w in weight_list],
            axis=0
        )
        avg_weights.append(layer_avg)
    return avg_weights

class ClusteredFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_clusters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters
        
        # State trackers for our clusters
        self.cluster_models = {}    # Maps: cluster_label -> aggregated weights
        self.client_to_cluster = {} # Maps: client_id -> cluster_label

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # 1. Extract local weights from all participating clients
        client_weights = {}
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            # Convert Flower Parameters back to NumPy arrays
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            client_weights[cid] = weights

        # 2. Run your clustering algorithm
        # (Using the cluster_clients function you already wrote)
        clusters = cluster_clients(client_weights, num_clusters=self.num_clusters)
        print(f"\n--- Round {server_round} Clusters Discovered ---")
        print(clusters)

        # 3. Aggregate weights PER CLUSTER instead of globally
        for label, cids in clusters.items():
            # Get the weights for clients in this specific cluster
            cluster_weight_list = [client_weights[cid] for cid in cids]
            
            # Average them and store in our state
            self.cluster_models[label] = average_weights(cluster_weight_list)
            
            # Update our routing table so we know who gets what next round
            for cid in cids:
                self.client_to_cluster[cid] = label

        # 4. Fallback for Flower's internal mechanics
        # Flower strictly requires returning a set of parameters here. 
        # We return the standard global FedAvg just to satisfy the API, 
        # but we will bypass it entirely in configure_fit.
        return super().aggregate_fit(server_round, results, failures)


    def configure_fit(self, server_round, parameters, client_manager):
        # 1. Ask the client manager to sample clients for this round
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Calculate the exact number of clients to sample based on the fraction
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )

        # 2. Route the correct cluster model to the correct client
        fit_instructions = []
        for client in clients:
            cid = client.cid
            
            # If the client has an assigned cluster from a previous round, send them the cluster model
            if cid in self.client_to_cluster and self.client_to_cluster[cid] in self.cluster_models:
                cluster_weights = self.cluster_models[self.client_to_cluster[cid]]
                # Convert NumPy arrays back to Flower Parameters
                cluster_parameters = fl.common.ndarrays_to_parameters(cluster_weights)
                fit_ins = fl.common.FitIns(cluster_parameters, config)
            else:
                # First round fallback: send the initial global model since clusters don't exist yet
                fit_ins = fl.common.FitIns(parameters, config)
                
            fit_instructions.append((client, fit_ins))

        return fit_instructions
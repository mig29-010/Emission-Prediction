import os
import glob

import flwr as fl

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from model import SOCLSTM
from utils import load_dataset

from clustering import cluster_clients


def average_weights(weight_list):

    avg_weights = []

    for layer_idx in range(len(weight_list[0])):

        layer_avg = np.mean(
            [w[layer_idx] for w in weight_list],
            axis=0
        )

        avg_weights.append(layer_avg)

    return avg_weights


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ROUNDS = 200

BATCH_SIZE = 32

LEARNING_RATE = 0.001

NUM_CLUSTERS = 3

EPOCH_EXPERIMENTS = [1, 5, 10, 20, 50]


CLIENT_FILES = sorted(
    glob.glob("clients/*.csv")
)


os.makedirs("results", exist_ok=True)


all_rmse_histories = {}

all_mae_histories = {}


for LOCAL_EPOCHS in EPOCH_EXPERIMENTS:

    print("\n============================")
    print(f"RUNNING {LOCAL_EPOCHS} LOCAL EPOCHS")
    print("============================\n")

    round_rmse = []

    round_mae = []

    global_client_weights = {}


    class SOCClient(fl.client.NumPyClient):

        def __init__(self, cid):

            csv_path = CLIENT_FILES[int(cid)]

            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                self.y_mean,
                self.y_std
            ) = load_dataset(csv_path)

            self.model = SOCLSTM(
                input_size=self.X_train.shape[2]
            ).to(DEVICE)

            self.criterion = nn.MSELoss()

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=LEARNING_RATE
            )

            self.cid = cid

        def get_parameters(self, config):

            return [
                val.cpu().numpy()
                for _, val in self.model.state_dict().items()
            ]

        def set_parameters(self, parameters):

            params_dict = zip(
                self.model.state_dict().keys(),
                parameters
            )

            state_dict = {
                k: torch.tensor(v)
                for k, v in params_dict
            }

            self.model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):

            self.set_parameters(parameters)

            X_tensor = torch.tensor(
                self.X_train,
                dtype=torch.float32
            )

            y_tensor = torch.tensor(
                self.y_train,
                dtype=torch.float32
            ).view(-1, 1)

            dataset = TensorDataset(
                X_tensor,
                y_tensor
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            self.model.train()

            for epoch in range(LOCAL_EPOCHS):

                running_loss = 0.0

                for X_batch, y_batch in loader:

                    X_batch = X_batch.to(DEVICE)

                    y_batch = y_batch.to(DEVICE)

                    self.optimizer.zero_grad()

                    predictions = self.model(X_batch)

                    loss = self.criterion(
                        predictions,
                        y_batch
                    )

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )

                    self.optimizer.step()

                    running_loss += loss.item()

            updated_weights = self.get_parameters(config={})

            global_client_weights[self.cid] = updated_weights

            return (
                updated_weights,
                len(self.X_train),
                {}
            )

        def evaluate(self, parameters, config):

            self.set_parameters(parameters)

            self.model.eval()

            X_tensor = torch.tensor(
                self.X_test,
                dtype=torch.float32
            ).to(DEVICE)

            with torch.no_grad():

                predictions = self.model(X_tensor)

                predictions = predictions.cpu().numpy().flatten()

                y_true = self.y_test

                mse_loss = np.mean(
                    (predictions - y_true) ** 2
                )

                rmse_normalized = np.sqrt(mse_loss)

                mae_normalized = np.mean(
                np.abs(predictions - y_true)
                )

                rmse_soc = rmse_normalized * self.y_std

                mae_soc = mae_normalized * self.y_std

            return (
                float(mse_loss),
                len(self.X_test),
                {
                    "rmse": float(rmse_soc),
                    "mae": float(mae_soc)
                }
            )


    def client_fn(context):

        cid = context.node_config["partition-id"]

        return SOCClient(str(cid)).to_client()



    def weighted_average(metrics):

        total_examples = sum(
            num_examples
            for num_examples, _ in metrics
        )

        rmse = sum(
            num_examples * m["rmse"]
            for num_examples, m in metrics
        ) /total_examples

        mae = sum(
            num_examples * m["mae"]
            for num_examples, m in metrics
        ) / total_examples

        round_rmse.append(rmse)

        round_mae.append(mae)

        print(
            f"Epochs={LOCAL_EPOCHS} | "
            f"Round={len(round_rmse)} | "
            f"RMSE={rmse:.4f}%SOC | "
            f"MAE={mae:.4f}%SOC"
        )

        if len(global_client_weights) == len(CLIENT_FILES):

            clusters = cluster_clients(
                global_client_weights,
                num_clusters=NUM_CLUSTERS
            )

            print("Clusters:", clusters)

        return {
            "rmse": rmse,
            "mae": mae
        }

    strategy = fl.server.strategy.FedAvg(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=len(CLIENT_FILES),

        min_evaluate_clients=len(CLIENT_FILES),

        min_available_clients=len(CLIENT_FILES),

        evaluate_metrics_aggregation_fn=weighted_average
    )


    fl.simulation.start_simulation(

        client_fn=client_fn,

        num_clients=len(CLIENT_FILES),

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy,

        client_resources={
            "num_cpus": 1,
            "num_gpus": 0.0
        }
    )


    all_rmse_histories[LOCAL_EPOCHS] = round_rmse

    all_mae_histories[LOCAL_EPOCHS] = round_mae


# =========================================
# RMSE COMPARISON PLOT
# =========================================

plt.figure(figsize=(14, 7))

for epochs, history in all_rmse_histories.items():

    plt.plot(
        history,
        label=f"{epochs} Local Epochs"
    )

plt.title("Clustered FL RMSE (%SOC) Comparison")

plt.xlabel("Communication Round")

plt.ylabel("RMSE (%SOC)")

plt.legend()

plt.grid(True)

plt.savefig(
    "results/clustered_fl_rmse_comparison.png"
)


# =========================================
# MAE COMPARISON PLOT
# =========================================

plt.figure(figsize=(14, 7))

for epochs, history in all_mae_histories.items():

    plt.plot(
        history,
        label=f"{epochs} Local Epochs"
    )

plt.title(
    "Clustered FL MAE (%SOC) Comparison"
)

plt.xlabel(
    "Communication Round"
)

plt.ylabel(
    "MAE (%SOC)"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/clustered_fl_mae_comparison.png"
)


print("\n===================================")
print("ALL EXPERIMENTS COMPLETE")
print("===================================")
"""
Chapter Four - Step 2: Load features, classes and edges; build a PyTorch Geometric
graph; apply the temporal train/validation/test split exactly as documented in
Chapter Three, Table 3.2 (train: steps 1-34, validation: steps 35-39, test: steps 40-49).
"""
import os
os.makedirs("outputs", exist_ok=True)
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

t0 = time.time()

# ---------------------------------------------------------------------------
# 1. Load the three source files
# ---------------------------------------------------------------------------
print("Loading merged features...")
feature_columns = [f"feature_{i}" for i in range(1, 166)]
all_columns = ["txId", "time_step"] + feature_columns
dtype_map = {"txId": np.int64, "time_step": np.int64}
dtype_map.update({c: np.float32 for c in feature_columns})
features_df = pd.read_csv("outputs/ch4_features_merged.csv", header=None,
                           names=all_columns, dtype=dtype_map)
print("  features_df shape:", features_df.shape)

print("Loading classes...")
classes_df = pd.read_csv("data/elliptic_txs_classes.csv")
print("  classes_df shape:", classes_df.shape)
print("  class value counts:\n", classes_df["class"].value_counts())

print("Loading edge list...")
edges_df = pd.read_csv("data/elliptic_txs_edgelist.csv")
print("  edges_df shape:", edges_df.shape)

# ---------------------------------------------------------------------------
# 2. Merge features with class labels on transaction ID
# ---------------------------------------------------------------------------
merged_df = features_df.merge(classes_df, on="txId", how="left")
print("Merged shape:", merged_df.shape)

label_lookup = {"1": 1, "2": 0, "unknown": -1}  # 1 = illicit, 0 = licit, -1 = unlabelled
merged_df["label"] = merged_df["class"].map(label_lookup)

# ---------------------------------------------------------------------------
# 3. Build contiguous node index and the edge_index tensor
# ---------------------------------------------------------------------------
transaction_ids = merged_df["txId"].values
id_to_node_index = {tx_id: idx for idx, tx_id in enumerate(transaction_ids)}

source_nodes = edges_df["txId1"].map(id_to_node_index)
target_nodes = edges_df["txId2"].map(id_to_node_index)
valid_edges = source_nodes.notna() & target_nodes.notna()
source_nodes = source_nodes[valid_edges].astype(np.int64).values
target_nodes = target_nodes[valid_edges].astype(np.int64).values
print(f"Edges retained: {len(source_nodes)} / {len(edges_df)}")

edge_index = torch.tensor(np.vstack([source_nodes, target_nodes]), dtype=torch.long)

feature_matrix = merged_df[feature_columns].values  # already float32
node_labels = merged_df["label"].values.astype(np.int64)
node_time_steps = merged_df["time_step"].values.astype(np.int64)
tx_ids_int = transaction_ids.astype(np.int64)

graph_data = Data(
    x=torch.tensor(feature_matrix, dtype=torch.float32),
    edge_index=edge_index,
    y=torch.tensor(node_labels, dtype=torch.long),
)
graph_data.time_step = torch.tensor(node_time_steps, dtype=torch.long)
graph_data.tx_id = torch.tensor(tx_ids_int, dtype=torch.long)

# free the large intermediate DataFrames now that the graph tensors are built
import gc
del features_df, merged_df, feature_matrix, node_labels, node_time_steps
gc.collect()

# ---------------------------------------------------------------------------
# 4. Temporal split, matching Chapter Three Table 3.2 exactly
#    Training:   time steps 1-34
#    Validation: time steps 35-39
#    Test:       time steps 40-49
# ---------------------------------------------------------------------------
is_labelled = graph_data.y >= 0
train_mask = is_labelled & (graph_data.time_step >= 1) & (graph_data.time_step <= 34)
val_mask = is_labelled & (graph_data.time_step >= 35) & (graph_data.time_step <= 39)
test_mask = is_labelled & (graph_data.time_step >= 40) & (graph_data.time_step <= 49)

graph_data.train_mask = train_mask
graph_data.val_mask = val_mask
graph_data.test_mask = test_mask

print("\n--- Table 3.2 split, as applied ---")
for split_name, mask in [("Training (steps 1-34)", train_mask),
                          ("Validation (steps 35-39)", val_mask),
                          ("Test (steps 40-49)", test_mask)]:
    n_nodes = mask.sum().item()
    n_illicit = (graph_data.y[mask] == 1).sum().item()
    pct_illicit = 100 * n_illicit / n_nodes if n_nodes else 0
    print(f"{split_name}: {n_nodes} nodes, {n_illicit} illicit ({pct_illicit:.2f}%)")

torch.save(graph_data, "outputs/ch4_graph_data.pt")
print(f"\nSaved graph object to outputs/ch4_graph_data.pt")
print(f"Total nodes: {graph_data.num_nodes}, total edges: {graph_data.num_edges}, feature dim: {graph_data.num_node_features}")
print(f"Done in {time.time() - t0:.1f}s")

"""
Chapter Four - Step 4: Generate SHAP explanations for a representative sample of
real, held-out test-set transactions, following the explainability approach
described in Chapter Three Section 3.6.

Because a Graph Attention Network's prediction for a node depends on its graph
neighbourhood as well as its own features, this script explains each transaction
by perturbing only that transaction's own 165 features while holding the rest of
its local neighbourhood fixed, and restricts each computation to the transaction's
2-hop subgraph so that KernelSHAP's repeated model evaluations remain tractable.
This script was written from scratch for this project.
"""
import os
os.makedirs("outputs", exist_ok=True)
import json
import time

import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class GraphAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_heads=4, dropout=0.2):
        super().__init__()
        self.first_layer = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.second_layer = GATConv(hidden_dim * num_heads, hidden_dim, heads=1,
                                     concat=False, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        h = F.elu(self.first_layer(x, edge_index))
        h = self.second_layer(h, edge_index)
        return self.output_layer(h)


def select_cases_for_explanation(graph_data, probabilities, n_true_positive=2,
                                   n_false_positive=2, n_true_negative=2):
    """Pick a representative sample from the test set that honestly reflects the
    model's real behaviour: some correct high-confidence illicit predictions
    (true positives), some incorrect high-confidence illicit predictions (false
    positives, since Chapter Four's confusion matrix shows these are common),
    and some correct low-confidence licit predictions (true negatives). This
    mirrors a realistic analyst queue rather than cherry-picking only successes."""
    test_node_indices = torch.where(graph_data.test_mask)[0].numpy()
    test_probabilities = probabilities[test_node_indices]
    test_true_labels = graph_data.y[test_node_indices].numpy()

    is_predicted_illicit = test_probabilities >= 0.5
    is_true_illicit = test_true_labels == 1

    true_positive_pool = np.where(is_predicted_illicit & is_true_illicit)[0]
    false_positive_pool = np.where(is_predicted_illicit & ~is_true_illicit)[0]
    true_negative_pool = np.where(~is_predicted_illicit & ~is_true_illicit)[0]

    def top_n_by_confidence(pool, n, descending=True):
        pool_probs = test_probabilities[pool]
        order = np.argsort(-pool_probs if descending else pool_probs)
        chosen = pool[order[:n]]
        return [(test_node_indices[i], test_probabilities[i], test_true_labels[i]) for i in chosen]

    selected = (
        top_n_by_confidence(true_positive_pool, n_true_positive, descending=True) +
        top_n_by_confidence(false_positive_pool, n_false_positive, descending=True) +
        top_n_by_confidence(true_negative_pool, n_true_negative, descending=False)
    )
    return selected


def build_local_predict_function(model, graph_data, node_index, num_hops=2):
    """Return a prediction function restricted to the node's local subgraph,
    so that KernelSHAP's many perturbed forward passes stay fast."""
    subgraph_nodes, subgraph_edge_index, node_mapping, _ = k_hop_subgraph(
        node_index, num_hops=num_hops, edge_index=graph_data.edge_index, relabel_nodes=True
    )
    base_subgraph_features = graph_data.x[subgraph_nodes].clone()
    local_node_position = int(node_mapping[0].item())

    def predict(feature_rows):
        batch = torch.tensor(feature_rows, dtype=torch.float32)
        outputs = np.zeros(batch.shape[0])
        for row_idx in range(batch.shape[0]):
            perturbed_features = base_subgraph_features.clone()
            perturbed_features[local_node_position] = batch[row_idx]
            with torch.no_grad():
                logits = model(perturbed_features, subgraph_edge_index)
                outputs[row_idx] = F.softmax(logits, dim=1)[local_node_position, 1].item()
        return outputs

    return predict, subgraph_nodes.numel()


def main():
    print("Loading trained model and graph...")
    graph_data = torch.load("outputs/ch4_graph_data_standardized.pt", weights_only=False)
    checkpoint = torch.load("outputs/ch4_trained_model.pt", weights_only=False)

    model = GraphAttentionClassifier(input_dim=graph_data.num_node_features)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index)
        all_probabilities = F.softmax(logits, dim=1)[:, 1].numpy()

    selected_cases = select_cases_for_explanation(graph_data, all_probabilities)
    print(f"Selected {len(selected_cases)} test-set transactions for SHAP explanation:")
    for node_index, prob, true_label in selected_cases:
        print(f"  node {node_index}: predicted P(illicit)={prob:.3f}, true label={true_label}")

    # Background reference set for KernelSHAP, sampled from the training set only
    training_node_indices = np.where(graph_data.train_mask.numpy())[0]
    background_indices = np.random.RandomState(SEED).choice(
        training_node_indices, size=25, replace=False
    )
    background_features = graph_data.x[background_indices].numpy()

    explanation_results = []
    t0 = time.time()
    for node_index, predicted_prob, true_label in selected_cases:
        node_index = int(node_index)
        predict_fn, subgraph_size = build_local_predict_function(model, graph_data, node_index)

        explainer = shap.KernelExplainer(predict_fn, background_features, silent=True)
        target_features = graph_data.x[node_index].numpy().reshape(1, -1)
        shap_values = np.array(
            explainer.shap_values(target_features, nsamples=120, silent=True)
        ).flatten()

        # neighbour / in-degree context from the raw (unstandardised) edge structure
        incoming_mask = graph_data.edge_index[1] == node_index
        outgoing_mask = graph_data.edge_index[0] == node_index

        explanation_results.append({
            "node_index": node_index,
            "tx_id": int(graph_data.tx_id[node_index].item()),
            "time_step": int(graph_data.time_step[node_index].item()),
            "predicted_probability": float(predicted_prob),
            "true_label": int(true_label),
            "in_degree": int(incoming_mask.sum().item()),
            "out_degree": int(outgoing_mask.sum().item()),
            "subgraph_size_2hop": int(subgraph_size),
            "shap_base_value": float(
                explainer.expected_value if np.isscalar(explainer.expected_value)
                else explainer.expected_value[0]
            ),
            "shap_values": shap_values.tolist(),
        })
        print(f"  node {node_index}: SHAP explanation done (elapsed {time.time() - t0:.1f}s)")

    with open("outputs/ch4_shap_results.json", "w") as f:
        json.dump(explanation_results, f, indent=2)
    print(f"\nSaved SHAP results for {len(explanation_results)} transactions "
          f"to outputs/ch4_shap_results.json")


if __name__ == "__main__":
    main()

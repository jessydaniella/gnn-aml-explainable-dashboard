"""
Chapter Four - Step 3: Define and train a two-layer Graph Attention Network (GAT)
on the Elliptic transaction graph, following the architecture described in
Chapter Three Section 3.5, and evaluate it using the six metrics specified in
Chapter Three Table 3.4 (Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix).

This script was written from scratch for this project. No external model
implementation was copied; only the standard, publicly documented PyTorch Geometric
GATConv layer (Velickovic et al., 2018; Fey and Lenssen, 2019) is used as a building
block, in the same way any published study would use a standard library layer.
"""
import os
os.makedirs("outputs", exist_ok=True)
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class GraphAttentionClassifier(nn.Module):
    """
    Two-layer GAT for binary illicit / licit transaction classification.

    Layer 1: multi-head attention (4 heads), followed by ELU activation and dropout.
    Layer 2: single-head attention that averages across the first layer's heads.
    Output:  linear layer producing two logits (licit, illicit).
    """

    def __init__(self, input_dim, hidden_dim=16, num_heads=4, dropout=0.2):
        super().__init__()
        self.first_layer = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.second_layer = GATConv(hidden_dim * num_heads, hidden_dim, heads=1,
                                     concat=False, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, return_attention=False):
        h, (edge_index_1, attention_1) = self.first_layer(
            x, edge_index, return_attention_weights=True
        )
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h, (edge_index_2, attention_2) = self.second_layer(
            h, edge_index, return_attention_weights=True
        )
        logits = self.output_layer(h)
        if return_attention:
            return logits, (edge_index_1, attention_1), (edge_index_2, attention_2)
        return logits


def standardize_features(graph_data):
    """Standardize node features using training-set mean and std only, to avoid
    leaking information from validation/test transactions into the scaling."""
    train_features = graph_data.x[graph_data.train_mask]
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True) + 1e-6
    graph_data.x = (graph_data.x - mean) / std
    return graph_data, mean, std


def compute_metrics(y_true, y_pred, y_prob):
    """Compute the six evaluation metrics specified in Chapter Three, Table 3.4."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate(model, graph_data, mask, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index)
        probabilities = F.softmax(logits, dim=1)[:, 1]
        predictions = (probabilities >= threshold).long()
    y_true = graph_data.y[mask].cpu().numpy()
    y_pred = predictions[mask].cpu().numpy()
    y_prob = probabilities[mask].cpu().numpy()
    return compute_metrics(y_true, y_pred, y_prob)


def main():
    print("Loading graph data...")
    graph_data = torch.load("outputs/ch4_graph_data.pt", weights_only=False)
    graph_data, feature_mean, feature_std = standardize_features(graph_data)

    model = GraphAttentionClassifier(input_dim=graph_data.num_node_features)

    # Class-weighted loss: penalise misclassifying the rare illicit class more
    # heavily than the licit class, following the imbalance-handling approach
    # described in Chapter Three Section 3.3 (Chawla et al., 2002 as the
    # motivating literature for treating class imbalance seriously).
    n_illicit_train = (graph_data.y[graph_data.train_mask] == 1).sum().item()
    n_licit_train = (graph_data.y[graph_data.train_mask] == 0).sum().item()
    class_weights = torch.tensor([1.0, n_licit_train / n_illicit_train], dtype=torch.float32)
    print(f"Class weights [licit, illicit]: {class_weights.tolist()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    print("\nTraining GAT model...")
    t0 = time.time()
    best_val_f1 = -1.0
    best_model_state = None
    training_log = []

    n_epochs = 80
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph_data.x, graph_data.edge_index)
        loss = F.cross_entropy(
            logits[graph_data.train_mask], graph_data.y[graph_data.train_mask],
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            val_metrics = evaluate(model, graph_data, graph_data.val_mask)
            print(f"Epoch {epoch:3d} | train loss {loss.item():.4f} | "
                  f"val Acc {val_metrics['accuracy']:.3f} "
                  f"P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f} "
                  f"F1 {val_metrics['f1']:.3f} AUC {val_metrics['roc_auc']:.3f}")
            training_log.append({"epoch": epoch, "loss": loss.item(), **val_metrics})
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nTraining complete in {time.time() - t0:.1f}s. Best validation F1: {best_val_f1:.3f}")
    model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, graph_data, graph_data.test_mask)
    print("\n=== FINAL TEST SET RESULTS (held-out, time steps 40-49) ===")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    torch.save({
        "model_state": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "class_weights": class_weights,
        "test_metrics": test_metrics,
        "training_log": training_log,
        "best_val_f1": best_val_f1,
    }, "outputs/ch4_trained_model.pt")

    torch.save(graph_data, "outputs/ch4_graph_data_standardized.pt")
    print("\nSaved trained model to outputs/ch4_trained_model.pt")


if __name__ == "__main__":
    main()

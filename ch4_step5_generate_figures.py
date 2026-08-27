"""
Chapter Four - Step 5: Generate figures directly from the real training log and
test-set results (training curves, ROC curve, confusion matrix heatmap).
Written from scratch for this project using matplotlib.
"""
import os
os.makedirs("outputs/figures", exist_ok=True)
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"

checkpoint = torch.load("outputs/ch4_trained_model.pt", weights_only=False)
training_log = checkpoint["training_log"]
test_metrics = checkpoint["test_metrics"]


# ---------------------------------------------------------------------------
# Figure: training progress (loss and validation F1 / ROC-AUC over epochs)
# ---------------------------------------------------------------------------
epochs = [row["epoch"] for row in training_log]
losses = [row["loss"] for row in training_log]
val_f1 = [row["f1"] for row in training_log]
val_auc = [row["roc_auc"] for row in training_log]

fig, ax1 = plt.subplots(figsize=(7, 4.2), dpi=200)
color1 = "#B3261E"
ax1.set_xlabel("Training epoch")
ax1.set_ylabel("Training loss", color=color1)
ax1.plot(epochs, losses, color=color1, marker="o", markersize=4, linewidth=1.8, label="Training loss")
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
color2 = "#1F6F5C"
ax2.set_ylabel("Validation F1 / ROC-AUC", color=color2)
ax2.plot(epochs, val_f1, color=color2, marker="s", markersize=4, linewidth=1.8, label="Validation F1")
ax2.plot(epochs, val_auc, color="#2E4374", marker="^", markersize=4, linewidth=1.8, label="Validation ROC-AUC")
ax2.tick_params(axis="y", labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8, frameon=True)
plt.title("GAT Training Progress: Loss and Validation Metrics", fontsize=11)
fig.tight_layout()
fig.savefig("outputs/figures/fig_training_curve.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_training_curve.png")

# ---------------------------------------------------------------------------
# Figure: confusion matrix heatmap
# ---------------------------------------------------------------------------
cm = np.array(test_metrics["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
im = ax.imshow(cm, cmap="Reds")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted\nLicit", "Predicted\nIllicit"])
ax.set_yticklabels(["Actual\nLicit", "Actual\nIllicit"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13, fontweight="bold")
ax.set_title("Confusion Matrix — Test Set (n = 11,184)", fontsize=11)
fig.tight_layout()
fig.savefig("outputs/figures/fig_confusion_matrix.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_confusion_matrix.png")

# ---------------------------------------------------------------------------
# Figure: ROC curve, recomputed directly from the trained model's real test-set
# probabilities (not just plotting the summary AUC number)
# ---------------------------------------------------------------------------
class GraphAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_heads=4, dropout=0.2):
        super().__init__()
        self.first_layer = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.second_layer = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        h = F.elu(self.first_layer(x, edge_index))
        h = self.second_layer(h, edge_index)
        return self.output_layer(h)


graph_data = torch.load("outputs/ch4_graph_data_standardized.pt", weights_only=False)
model = GraphAttentionClassifier(input_dim=graph_data.num_node_features)
model.load_state_dict(checkpoint["model_state"])
model.eval()
with torch.no_grad():
    logits = model(graph_data.x, graph_data.edge_index)
    probs = F.softmax(logits, dim=1)[:, 1].numpy()

y_true_test = graph_data.y[graph_data.test_mask].numpy()
y_prob_test = probs[graph_data.test_mask.numpy()]
fpr, tpr, _ = roc_curve(y_true_test, y_prob_test)

fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=200)
ax.plot(fpr, tpr, color="#2E4374", linewidth=2.2, label=f"GAT (AUC = {test_metrics['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color="#B9B4A8", linestyle="--", linewidth=1.2, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Test Set (steps 40-49)", fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig("outputs/figures/fig_roc_curve.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_roc_curve.png")

print("\nAll figures generated from real training log and real test-set predictions.")

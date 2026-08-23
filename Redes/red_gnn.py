import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

import matplotlib.pyplot as plt

# =========================
# FIX PyTorch 2.6 loader
# =========================
torch.serialization.add_safe_globals([Data])

data_list = torch.load(
    "grafos_temporales.pt",
    weights_only=False
)

print(f"Grafos cargados: {len(data_list)}")

# =========================
# 2. Split temporal
# =========================
train_graphs = data_list[:21]
val_graphs   = data_list[21:24]
test_graphs  = data_list[24:]

# =========================
# 3. Class weights
# =========================
all_labels = torch.cat([g.y for g in train_graphs]).cpu().numpy()

classes = np.unique(all_labels)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=all_labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Class weights:", class_weights)

# =========================
# 3.1 Mejora 4 Focal loss
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            weight=self.alpha
        )

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()
# =========================
# 4. Modelo GraphSAGE
# =========================
class GraphSAGE(nn.Module):
    def __init__(self, input_dim=24):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = SAGEConv(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = SAGEConv(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(32, 64)
        self.bn4 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, 2)

        self.dropout = 0.2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Conv 1
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Conv 2
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Conv 3 + residual
        x3 = self.conv3(x2, edge_index)
        x3 = x3 + x2[:, :32]  # residual skip connection
        x3 = F.relu(self.bn3(x3))
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # Dense layers
        x = F.relu(self.bn4(self.fc1(x3)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.bn5(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.out(x)

        return out

# =========================
# 5. Entrenamiento
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GraphSAGE(input_dim=24).to(device)
class_weights = class_weights.to(device)

#Mejora 4 sustituir criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = FocalLoss(
    alpha=class_weights,
    gamma=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 6. Loop de entrenamiento
# =========================
def run_epoch(graphs, train=True):
    model.train() if train else model.eval()

    losses = []
    preds_all = []
    labels_all = []

    for g in graphs:
        g = g.to(device)

        out = model(g)
        loss = criterion(out, g.y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        preds = out.argmax(dim=1).detach().cpu().numpy()
        labels = g.y.detach().cpu().numpy()

        preds_all.append(preds)
        labels_all.append(labels)

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    return np.mean(losses), preds_all, labels_all

# =========================
# 7. Early stopping
# =========================
#best_val_loss = float("inf")
best_val_f1 = 0
best_epoch = 0
# Mejora 1 antes patience era 20
patience = 30
counter = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": []
}
# Mejora 1
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer,
#    mode="min",
#    factor=0.5,
#    patience=5,
#    min_lr=1e-5
#)

#Mejora 2 scheduler con f1
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    min_lr=1e-5
)

for epoch in range(200):

    train_loss, train_preds, train_labels = run_epoch(train_graphs, train=True)

    val_loss, val_preds, val_labels = run_epoch(val_graphs, train=False)

    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)
    #Guardar métricas
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_labels,
        val_preds,
        average="weighted",
        zero_division=0
    )

    history["val_precision"].append(precision)
    history["val_recall"].append(recall)
    history["val_f1"].append(f1)
    # Mejora 1
    #scheduler.step(val_loss)
    #Mejora 2
    scheduler.step(f1)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    #Imprimir métricas
    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Recall: {recall:.4f} | "
        f"Val F1: {f1:.4f}"
    )
    #Mejora 2 cambiamos if val_loss < best_val_loss:
    if f1 > best_val_f1:
        best_val_f1 = f1
        counter = 0
        torch.save(model.state_dict(), "best_gnn.pt")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activado")
            break

# =========================
# 8. Evaluación final
# =========================
model.load_state_dict(torch.load("best_gnn.pt"))
model.eval()

test_loss, test_preds, test_labels = run_epoch(test_graphs, train=False)

acc = accuracy_score(test_labels, test_preds)
prec, rec, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="weighted")

mae = mean_absolute_error(test_labels, test_preds)
rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
cm = confusion_matrix(test_labels, test_preds)

print("\n===== RESULTADOS TEST =====")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)
print("MAE:", mae)
print("RMSE:", rmse)
print("Confusion Matrix:\n", cm)

# =========================
# 9. Gráficas
# =========================
plt.figure()
plt.plot(history["train_loss"], label="Pérdida del entrenamiento")
plt.plot(history["val_loss"], label="Pérdida de la validación")
plt.xlabel("Épocas")
plt.ylabel("Pérdida (Loss)")
plt.legend()
plt.title("Pérdida GNN")
plt.savefig("loss_gnn.png")

plt.figure()

plt.plot(history["train_acc"], label="Exactitud del entrenamiento")
plt.plot(history["val_acc"], label="Exactitud de la validación")
plt.xlabel("Épocas")
plt.ylabel("Exactitud (Accuracy)")
plt.legend()
plt.title("Exactitud GNN")
plt.savefig("accuracy_gnn.png")

plt.figure()

plt.plot(history["val_precision"], label="Validación de la presición")
plt.plot(history["val_recall"], label="Validación de Recall (exhaustividad)")
plt.plot(history["val_f1"], label="Validación F1-score")

plt.xlabel("Épocas")
plt.ylabel("Puntuación (Score)")
plt.title("Métricas de validación")
plt.legend()

plt.savefig("validation_metrics_gnn.png")
plt.close()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import Callback

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

def focal_loss(alpha=0.25, gamma=2.0):

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        epsilon = K.epsilon()

        y_pred = K.clip(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        cross_entropy = -(
            y_true * K.log(y_pred)
            +
            (1 - y_true) * K.log(1 - y_pred)
        )

        p_t = (
            y_true * y_pred
            +
            (1 - y_true) * (1 - y_pred)
        )

        alpha_factor = (
            y_true * alpha
            +
            (1 - y_true) * (1 - alpha)
        )

        focal_weight = alpha_factor * K.pow(
            1 - p_t,
            gamma
        )

        return K.mean(
            focal_weight * cross_entropy
        )

    return loss

# =====================================
# 1. CARGAR DATASET
# =====================================

print("Cargando dataset...")

dataset = pd.read_csv(
    "dataset_mlp_temporal.csv"
)

dataset = dataset[
    dataset["periodo_prediccion"] <= "2024-07"
].copy()

print("Total muestras:", len(dataset))


# =====================================
# 2. PARTICIÓN TEMPORAL
# =====================================

train = dataset[
    dataset["periodo_prediccion"] <= "2023-12"
]

val = dataset[
    (dataset["periodo_prediccion"] >= "2024-01") &
    (dataset["periodo_prediccion"] <= "2024-03")
]

test = dataset[
    dataset["periodo_prediccion"] >= "2024-04"
]


# =====================================
# 3. VARIABLES
# =====================================

columnas_excluir = [
    "grid_id",
    "periodo_prediccion",
    "zona_prioritaria"
]

features = [
    col for col in dataset.columns
    if col not in columnas_excluir
]

X_train = train[features]
X_val = val[features]
X_test = test[features]

y_train = train["zona_prioritaria"]
y_val = val["zona_prioritaria"]
y_test = test["zona_prioritaria"]


# =====================================
# 4. NORMALIZACIÓN
# =====================================

print("\nNormalizando...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)


# =====================================
# 5. CLASS WEIGHTS
# =====================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("\nClass weights:")

print(class_weights)


# =====================================
# 6. CONSTRUIR MLP
# =====================================

print("\nConstruyendo modelo...")

model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    BatchNormalization(),

    Dropout(0.3),


    Dense(
        64,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.3),


    Dense(
        32,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.2),


    Dense(
        1,
        activation="sigmoid"
    )

])


model.compile(

    optimizer=Adam(),

    loss=focal_loss(
        alpha=0.75,
        gamma=2
    ),

    metrics=["accuracy"]

)

model.summary()


# =====================================
# 7. EARLY STOPPING
# =====================================
class ValidationMetrics(Callback):

    def __init__(self, validation_data):
        super().__init__()

        self.X_val = validation_data[0]
        self.y_val = validation_data[1]

        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        y_prob = self.model.predict(
            self.X_val,
            verbose=0
        )

        y_pred = (
            y_prob >= 0.5
        ).astype(int)

        precision = precision_score(
            self.y_val,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            self.y_val,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            self.y_val,
            y_pred,
            zero_division=0
        )

        self.val_precision.append(
            precision
        )

        self.val_recall.append(
            recall
        )

        self.val_f1.append(
            f1
        )

        logs["val_f1"] = f1

        lr = float(
            tf.keras.backend.get_value(
                self.model.optimizer.learning_rate
            )
        )

        self.learning_rates.append(lr)

        print(
            f" - val_precision: {precision:.4f}"
            f" - val_recall: {recall:.4f}"
            f" - val_f1: {f1:.4f}"
        )

early_stopping = EarlyStopping(

    monitor="val_f1",

    mode="max",

    patience=30,

    restore_best_weights=True,

    verbose=1

)

scheduler = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_f1",

    mode="max",

    factor=0.5,

    patience=5,

    min_lr=1e-5,

    verbose=1

)

metrics_callback = ValidationMetrics(

    validation_data=(
        X_val,
        y_val
    )

)
# =====================================
# 8. ENTRENAMIENTO
# =====================================

print("\nEntrenando...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=200,

    batch_size=64,

    #class_weight=class_weights,

    callbacks=[

        early_stopping,

        scheduler,

        metrics_callback

    ],

    verbose=1

)


# =====================================
# 9. GUARDAR MODELO
# =====================================

model.save(
    "MLP_model.keras"
)

print("\nModelo guardado.")


# =====================================
# 10. EVALUACIÓN
# =====================================

print("\nEvaluando...")

y_prob = model.predict(
    X_test
)

y_pred = (
    y_prob >= 0.5
).astype(int)


# =====================================
# 11. MÉTRICAS
# =====================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_prob
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_prob
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nAccuracy:")

print(accuracy)

print("\nMAE:")

print(mae)

print("\nRMSE:")

print(rmse)

print("\nMatriz de confusión:")

print(cm)

print("\nReporte:")

print(

    classification_report(

        y_test,

        y_pred,

        digits=4

    )

)


# =====================================
# 12. CURVA LOSS
# =====================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Pérdida del entrenamiento"
)

plt.plot(
    history.history["val_loss"],
    label="Pérdida de la validación"
)

plt.xlabel("Épocas")

plt.ylabel("Pérdida (Loss)")

plt.legend()

plt.title("Pérdida (MLP)")

plt.savefig(
    "loss_mlp.png"
)

plt.show()


# =====================================
# 13. CURVA ACCURACY
# =====================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Exactitud del entrenamiento"
)

plt.plot(
    history.history["val_accuracy"],
    label="Exactitud de la validación"
)

plt.xlabel("Épocas")

plt.ylabel("Exactitud (Accuracy)")

plt.legend()

plt.title("Exactitud (MLP)")

plt.savefig(
    "accuracy_mlp.png"
)

plt.show()

# =====================================
# 14. CURVA METRICAS
# =====================================

plt.figure()

plt.plot(
    metrics_callback.val_precision,
    label="Validación de la presición"
)

plt.plot(
    metrics_callback.val_recall,
    label="Validación de Recall (exhaustividad)"
)

plt.plot(
    metrics_callback.val_f1,
    label="Validación F1-score"
)

plt.xlabel("Épocas")

plt.ylabel("Puntuación (Score)")

plt.legend()

plt.title("Métricas de validación")

plt.savefig(
    "metrics_mlp.png"
)

plt.show()

plt.figure()

plt.plot(
    metrics_callback.learning_rates
)

plt.xlabel("Épocas")

plt.ylabel("Learning Rate")

plt.title("Learning Rate (MLP)")

plt.savefig(
    "learning_rate_mlp.png"
)

plt.show()

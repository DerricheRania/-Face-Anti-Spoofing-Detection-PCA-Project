import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



#  ÉTAPE 2 — PRÉTRAITEMENT



def normalize(X):
    """
    Normalise les valeurs de pixels de [0, 255] vers [0, 1].

    Pourquoi normaliser ?
    ─────────────────────
    Les pixels bruts vont de 0 (noir) à 255 (blanc).
    Sans normalisation, l'ACP et les classifieurs donnent plus
    d'importance aux pixels à haute valeur, ce qui biaise les résultats.
    En divisant par 255, tous les pixels ont le même poids.
    """
    return X / 255.0


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divise le dataset en train et test de manière stratifiée.

    Le paramètre stratify=y garantit que les deux classes
    (Réel et Fake) sont représentées proportionnellement
    dans chaque sous-ensemble.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def plot_distribution(y_train, y_test, save_path="output_2_distribution.png"):
    """
    Affiche la distribution des classes dans train et test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Distribution des classes après division Train/Test",
                 fontsize=12, fontweight="bold")

    for ax, y_split, title in zip(
        axes,
        [y_train, y_test],
        ["Entraînement (80%)", "Test (20%)"]
    ):
        counts = [np.sum(y_split == 1), np.sum(y_split == 0)]
        bars = ax.bar(["Réel", "Fake"], counts,
                      color=["#3A86FF", "#FF006E"], edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Nombre d'images")
        for b, v in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                    ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Figure sauvegardée → {save_path}")

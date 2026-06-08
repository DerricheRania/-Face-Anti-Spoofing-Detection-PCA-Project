import os
import warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# ── Imports des modules du projet 
from one_load_and_explore  import load_balanced_dataset, explore_dataset
from two_preprocessing     import normalize, split_data, plot_distribution
from three_pca             import (apply_pca,
                                   # Visualisations PCA de base (existantes)
                                   plot_variance, plot_eigenfaces,
                                   plot_pca_2d, plot_pca_3d,
                                   plot_reconstruction, plot_pc_distributions,
                                   # ✨ NOUVELLES fonctions PCA pédagogiques
                                   plot_correlation_matrix,
                                   plot_eigenvalues,
                                   plot_eigenvectors,
                                   print_eigentable,
                                   plot_pca_axes_pairs_with_variance)
from four_classification   import train_all_classifiers
from five_evaluation       import evaluate_all



#  ⚙️  CONFIGURATION : Modifiez uniquement cette section


DATA_DIR      = "LCC_FASD"
REAL_DIR      = os.path.join(DATA_DIR, "real")
FAKE_DIR      = os.path.join(DATA_DIR, "spoof")

IMG_SIZE      = (64, 64)   # taille cible des images (H, W)
N_COMPONENTS  = 50         # composantes ACP à garder
TEST_SIZE     = 0.2        # 20% pour le test
RANDOM_STATE  = 42



#  POINT D'ENTRÉE


if __name__ == "__main__":
    print("\n" + "█" * 62)
    print("  PROJET — Detection Anti-Spoofing   |   ACP + SVM + RF + LR + GB + MLP")
    print("  Dataset : LCC_FASD")
    print("█" * 62)

    # ──────────────────────────────────────────────────────────
    #  ÉTAPE 1 — Chargement et Exploration
    # ──────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  ÉTAPE 1 — Chargement et Exploration du Dataset")
    print("═" * 62)

    X, y, X_real, X_fake = load_balanced_dataset(
        real_dir=REAL_DIR,
        fake_dir=FAKE_DIR,
        img_size=IMG_SIZE
    )

    explore_dataset(X_real, X_fake,
                    img_size=IMG_SIZE,
                    save_path="output_1_exemples.png")

    # ──────────────────────────────────────────────────────────
    #  ÉTAPE 2 — Prétraitement
    # ──────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  ÉTAPE 2 — Prétraitement")
    print("═" * 62)

    X_norm = normalize(X)
    print(f"\n   Normalisation : pixels [0,255] → [0,1]")
    print(f"   Min: {X_norm.min():.2f}  |  Max: {X_norm.max():.2f}")

    X_train, X_test, y_train, y_test = split_data(
        X_norm, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n   Entraînement : {X_train.shape[0]} images")
    print(f"   Test         : {X_test.shape[0]} images")

    plot_distribution(y_train, y_test,
                      save_path="output_2_distribution.png")

    # ──────────────────────────────────────────────────────────
    #  ÉTAPE 3 — ACP (PCA)   ← section étendue
    # ──────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  ÉTAPE 3 — Réduction de dimension par ACP")
    print("═" * 62)

    X_train_pca, X_test_pca, pca, scaler = apply_pca(
        X_train, X_test,
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )
    print(f"\n   Dimension originale : {X_train.shape[1]} pixels "
          f"({IMG_SIZE[0]}×{IMG_SIZE[1]})")
    print(f"   Après ACP           : {N_COMPONENTS} composantes")
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    print(f"   Variance expliquée  : {cumvar[-1]:.1f}%")

    # ── Standardisation train (pour les visualisations ACP) ───
    from sklearn.preprocessing import StandardScaler as _SC
    _sc_tmp = _SC()
    X_train_sc = _sc_tmp.fit_transform(X_train)

    # ─────────────────────────────────────────────────────────
    #  3A — MATRICE DE CORRÉLATION (entrée de l'ACP)
    # ─────────────────────────────────────────────────────────
    print("\n   [3A] Matrice de corrélation ...")
    plot_correlation_matrix(
        X_train_sc,
        n_pixels=40,
        save_path="output_3a_correlation_matrix.png"
    )

    # ─────────────────────────────────────────────────────────
    #  3B — VALEURS PROPRES (scree plot + variance %)
    # ─────────────────────────────────────────────────────────
    print("\n   [3B] Valeurs propres (eigenvalues) ...")
    plot_eigenvalues(pca, n_show=N_COMPONENTS,
                     save_path="output_3b_eigenvalues.png")

    # ─────────────────────────────────────────────────────────
    #  3C — TABLEAU VALEURS / VECTEURS PROPRES (console)
    # ─────────────────────────────────────────────────────────
    print_eigentable(pca, n_show=15)

    # ─────────────────────────────────────────────────────────
    #  3D — VECTEURS PROPRES (eigenfaces détaillées)
    # ─────────────────────────────────────────────────────────
    print("\n   [3D] Vecteurs propres (eigenvectors) ...")
    plot_eigenvectors(pca, img_size=IMG_SIZE, n_show=6,
                      save_path="output_3d_eigenvectors.png")

    # ─────────────────────────────────────────────────────────
    #  3E — VARIANCE EXPLIQUÉE (courbe du coude)
    # ─────────────────────────────────────────────────────────
    print("\n   [3E] Variance expliquée + courbe du coude ...")
    plot_variance(pca, save_path="output_3_variance_pca.png")

    # ─────────────────────────────────────────────────────────
    #  3F — EIGENFACES (toutes les 15 premières)
    # ─────────────────────────────────────────────────────────
    print("\n   [3F] Eigenfaces ...")
    plot_eigenfaces(pca, img_size=IMG_SIZE, n_show=15,
                    save_path="output_4_eigenfaces.png")

    # ─────────────────────────────────────────────────────────
    #  3G — PROJECTION DANS LES AXES CHOISIS (paires PC)
    # ─────────────────────────────────────────────────────────
    print("\n   [3G] Représentation dans les axes choisis (PC1/PC2, PC1/PC3, PC2/PC3) ...")
    plot_pca_axes_pairs_with_variance(
        X_train_pca, y_train, pca,
        save_path="output_3g_pca_axes_pairs.png"
    )

    # ─────────────────────────────────────────────────────────
    #  3H — PROJECTION 2D (PC1 vs PC2)
    # ─────────────────────────────────────────────────────────
    print("\n   [3H] Projection 2D ...")
    plot_pca_2d(X_train_pca, y_train, title_suffix="Train",
                save_path="output_5_pca_2d.png")

    # ─────────────────────────────────────────────────────────
    #  3I — PROJECTION 3D (PC1, PC2, PC3)
    # ─────────────────────────────────────────────────────────
    print("\n   [3I] Projection 3D ...")
    plot_pca_3d(X_train_pca, y_train, title_suffix="Train",
                save_path="output_5b_pca_3d.png")

    # ─────────────────────────────────────────────────────────
    #  3J — DISTRIBUTION DES SCORES PC1 & PC2
    # ─────────────────────────────────────────────────────────
    print("\n   [3J] Distribution des scores PC1 & PC2 ...")
    plot_pc_distributions(X_train_pca, y_train,
                          save_path="output_5d_pc_distributions.png")

    # ─────────────────────────────────────────────────────────
    #  3K — RECONSTRUCTION D'IMAGES
    # ─────────────────────────────────────────────────────────
    print("\n   [3K] Reconstruction des images ...")
    plot_reconstruction(X_train_sc, pca, _sc_tmp,
                        img_size=IMG_SIZE, n_samples=4,
                        save_path="output_5c_reconstruction.png")

    # Sauvegarde des objets ACP pour réutilisation
    joblib.dump(pca,    "pca_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\n   ✅ Modèles ACP sauvegardés → pca_model.pkl  scaler.pkl")

    # ──────────────────────────────────────────────────────────
    #  ÉTAPE 4 — Classification
    # ──────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  ÉTAPE 4 — Entraînement des classifieurs")
    print("═" * 62)

    classifiers = train_all_classifiers(X_train_pca, y_train,
                                        random_state=RANDOM_STATE)

    # Sauvegarde de tous les modèles
    for name, clf in classifiers.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_model.pkl"
        joblib.dump(clf, fname)
    print(f"   ✅ {len(classifiers)} modèles sauvegardés (.pkl)")

    # ──────────────────────────────────────────────────────────
    #  ÉTAPE 5 — Évaluation
    # ──────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  ÉTAPE 5 — Évaluation sur le jeu de test")
    print("═" * 62)
    print(f"\n   Réel dans test : {np.sum(y_test==1)}  |  "
          f"Fake dans test : {np.sum(y_test==0)}")

    results = evaluate_all(
        classifiers, X_test_pca, y_test,
        save_detail="output_6_evaluation.png",
        save_compare="output_7_comparaison.png"
    )

    # ──────────────────────────────────────────────────────────
    #  RÉSUMÉ FINAL
    # ──────────────────────────────────────────────────────────
    print("\n" + "█" * 62)
    print("  ✅ Projet terminé !  Figures générées :")
    print("     output_1_exemples.png          — exemples dataset")
    print("     output_2_distribution.png      — répartition train/test")
    print("     output_3a_correlation_matrix.png — matrice de corrélation (ACP)")
    print("     output_3b_eigenvalues.png       — valeurs propres (scree plot)")
    print("     output_3d_eigenvectors.png      — vecteurs propres (6 premiers)")
    print("     output_3_variance_pca.png       — variance ACP (courbe du coude)")
    print("     output_4_eigenfaces.png         — 15 composantes principales")
    print("     output_3g_pca_axes_pairs.png    — représentation dans les axes choisis")
    print("     output_5_pca_2d.png             — projection 2D PC1 vs PC2")
    print("     output_5b_pca_3d.png            — projection 3D PC1/PC2/PC3")
    print("     output_5c_reconstruction.png    — reconstruction images ACP")
    print("     output_5d_pc_distributions.png  — distributions PC1 & PC2")
    print("     output_6_evaluation.png         — confusion + ROC + métriques")
    print("     output_7_comparaison.png        — comparaison des 5 classifieurs")
    print("█" * 62 + "\n")
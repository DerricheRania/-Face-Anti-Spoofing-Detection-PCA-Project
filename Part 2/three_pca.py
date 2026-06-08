import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



#  ÉTAPE 3 — RÉDUCTION DE DIMENSION PAR ACP (PCA)



def apply_pca(X_train, X_test, n_components=50, random_state=42):
    """
    Applique la standardisation puis l'ACP sur train et test.

    Pourquoi standardiser AVANT l'ACP ?
    ─────────────────────────────────────
    L'ACP maximise la variance. Si une variable (pixel) a une variance
    naturellement plus grande, elle dominera les composantes sans que
    ce soit pertinent. La standardisation met chaque pixel sur un même pied
    d'égalité (moyenne=0, écart-type=1) AVANT de calculer les axes.

    Règle d'or : fit uniquement sur X_train, transform sur X_test.
    Cela évite la fuite de données (data leakage).

    Retourne
    --------
    X_train_pca : (n_train, n_components)
    X_test_pca  : (n_test,  n_components)
    pca         : objet PCA ajusté
    scaler      : objet StandardScaler ajusté
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit + transform sur train
    X_test_sc  = scaler.transform(X_test)         # transform seulement sur test

    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_sc)   # fit + transform sur train
    X_test_pca  = pca.transform(X_test_sc)         # transform seulement sur test

    return X_train_pca, X_test_pca, pca, scaler



#  NOUVEAU — MATRICE DE CORRÉLATION (sur un sous-espace réduit)



def plot_correlation_matrix(X_train_sc, n_pixels=40,
                             save_path="output_pca_correlation.png"):
    """
    Affiche la matrice de corrélation entre les n_pixels premiers pixels
    après standardisation.

    Pourquoi la matrice de corrélation ?
    ──────────────────────────────────────
    L'ACP cherche à diagonaliser cette matrice : elle transforme les données
    pour que les nouvelles variables (les PC) soient DÉCORRÉLÉES.
    Les blocs colorés montrent les groupes de pixels qui varient ensemble —
    ce sont exactement les structures que l'ACP va capturer.

    Note : on ne prend que les n_pixels premiers pixels pour que la figure
    reste lisible (64×64 = 4096 pixels serait illisible).
    """
    # Sous-ensemble de pixels pour la lisibilité
    X_sub = X_train_sc[:, :n_pixels]
    corr  = np.corrcoef(X_sub.T)          # matrice (n_pixels × n_pixels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Coefficient de corrélation (r)")

    ax.set_title(
        f"Matrice de Corrélation entre les {n_pixels} premiers pixels\n"
        "(données standardisées — entrée de l'ACP)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Index pixel")
    ax.set_ylabel("Index pixel")

    # Ajouter les valeurs numériques si la matrice est petite
    if n_pixels <= 15:
        for i in range(n_pixels):
            for j in range(n_pixels):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Matrice de corrélation sauvegardée → {save_path}")

    # Résumé console
    mask = np.abs(corr) > 0.5
    np.fill_diagonal(mask, False)
    print(f"   → {mask.sum()//2} paires de pixels fortement corrélées (|r| > 0.5)")
    print(f"   → L'ACP va compresser ces redondances en composantes indépendantes.")



#  NOUVEAU — VALEURS PROPRES ET VECTEURS PROPRES



def plot_eigenvalues(pca, n_show=20, save_path="output_pca_eigenvalues.png"):
    """
    Affiche les valeurs propres (eigenvalues) de la matrice de covariance.

    Qu'est-ce qu'une valeur propre ?
    ──────────────────────────────────
    Chaque axe principal (PC) de l'ACP est associé à une valeur propre λ.
    λ mesure la VARIANCE capturée par cet axe dans l'espace original.
    Plus λ est grand, plus la composante est importante.

    Règle de Kaiser : on garde les composantes avec λ > 1
    (elles expliquent plus d'une variable standardisée).

    pca.explained_variance_  → ce sont directement les valeurs propres λ_i
    pca.explained_variance_ratio_ → λ_i / Σλ  (proportion de variance)
    """
    eigenvalues = pca.explained_variance_        # λ_1, λ_2, ..., λ_k
    n = min(n_show, len(eigenvalues))
    idx = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Valeurs propres (Eigenvalues) de la matrice de covariance",
                 fontsize=14, fontweight="bold")

    # ── Graphique 1 : Valeurs propres brutes (scree plot) ─────────
    axes[0].bar(idx, eigenvalues[:n], color="#3A86FF", edgecolor="none", alpha=0.85)
    axes[0].axhline(1, color="red", linestyle="--", linewidth=1.5,
                    label="Règle de Kaiser (λ = 1)")
    axes[0].set_xlabel("Numéro de la composante principale")
    axes[0].set_ylabel("Valeur propre λ")
    axes[0].set_title("Scree Plot — Valeurs propres")
    axes[0].legend()
    axes[0].set_xticks(idx)

    # Annoter les 5 premières valeurs
    for i in range(min(5, n)):
        axes[0].text(i + 1, eigenvalues[i] + eigenvalues[0]*0.01,
                     f"λ={eigenvalues[i]:.1f}",
                     ha="center", fontsize=8, color="navy")

    # ── Graphique 2 : Variance expliquée % par composante ──────────
    var_pct = pca.explained_variance_ratio_[:n] * 100
    cumvar  = np.cumsum(var_pct)

    axes[1].bar(idx, var_pct, color="#FF006E", edgecolor="none", alpha=0.7,
                label="Variance par PC (%)")
    ax2 = axes[1].twinx()
    ax2.plot(idx, cumvar, color="#3A86FF", linewidth=2.5,
             marker="o", markersize=4, label="Variance cumulée (%)")
    ax2.axhline(90, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("Variance cumulée (%)", color="#3A86FF")
    ax2.set_ylim(0, 105)

    axes[1].set_xlabel("Numéro de la composante principale")
    axes[1].set_ylabel("Variance expliquée par PC (%)", color="#FF006E")
    axes[1].set_title("Variance expliquée par composante")
    axes[1].set_xticks(idx)

    # Légendes combinées
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Valeurs propres sauvegardées → {save_path}")

    # Résumé console
    kaiser_count = np.sum(eigenvalues > 1)
    print(f"\n   Valeurs propres (λ) — 10 premières :")
    for i in range(min(10, len(eigenvalues))):
        print(f"     PC{i+1:2d} : λ = {eigenvalues[i]:8.3f} "
              f"  →  {pca.explained_variance_ratio_[i]*100:.2f}% de variance")
    print(f"\n   → Règle de Kaiser : {kaiser_count} composantes avec λ > 1")



#  NOUVEAU — VECTEURS PROPRES (premiers axes de l'ACP)



def plot_eigenvectors(pca, img_size=(64, 64), n_show=6,
                      save_path="output_pca_eigenvectors.png"):
    """
    Visualise les n_show premiers vecteurs propres (eigenvectors).

    Qu'est-ce qu'un vecteur propre ?
    ──────────────────────────────────
    Chaque composante principale est définie par un VECTEUR PROPRE v_i
    dans l'espace des pixels. C'est la DIRECTION dans laquelle l'ACP
    projette les données pour capturer le plus de variance.

    pca.components_[i] est le i-ème vecteur propre, de dimension
    égale au nombre de pixels (ici 64×64 = 4096).

    On peut le réafficher comme une image 64×64 → c'est ce qu'on appelle
    une "eigenface" dans le contexte de la reconnaissance faciale.

    Rouge = direction positive  |  Bleu = direction négative
    → Les zones rouges/bleues intenses sont les pixels les PLUS discriminants.
    """
    fig = plt.figure(figsize=(15, 5 * n_show // 3 + 3))
    fig.suptitle(
        f"Vecteurs propres (Eigenvectors) — {n_show} premiers axes de l'ACP\n"
        "Chaque image représente UNE direction dans l'espace pixel\n"
        "Rouge = activation positive | Bleu = activation négative",
        fontsize=12, fontweight="bold"
    )

    cols = 3
    rows = int(np.ceil(n_show / cols))
    gs   = gridspec.GridSpec(rows, cols, figure=fig,
                             hspace=0.5, wspace=0.3)

    for i in range(n_show):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ev = pca.components_[i].reshape(img_size)   # vecteur propre → image
        vmax = np.abs(ev).max()
        im = ax.imshow(ev, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        var_pct = pca.explained_variance_ratio_[i] * 100
        ax.set_title(
            f"PC {i+1} — Vecteur propre v_{i+1}\n"
            f"λ_{i+1} = {pca.explained_variance_[i]:.2f}  "
            f"({var_pct:.2f}% var.)",
            fontsize=9
        )
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        ax.axis("off")

    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Vecteurs propres sauvegardés → {save_path}")

    # Résumé console
    print(f"\n   Forme de pca.components_ : {pca.components_.shape}")
    print(f"   = ({pca.n_components_} composantes × {pca.components_.shape[1]} pixels)")
    print(f"   Chaque ligne est un vecteur propre (direction dans l'espace pixel).")



#  NOUVEAU — TABLEAU RÉCAPITULATIF VALEURS/VECTEURS PROPRES



def print_eigentable(pca, n_show=10):
    """
    Affiche un tableau résumant valeurs propres, vecteurs propres (norme,
    top pixels) et variance capturée — utile pour un rapport.
    """
    print("\n" + "═"*70)
    print("  TABLEAU DES VALEURS ET VECTEURS PROPRES")
    print("═"*70)
    print(f"  {'PC':>4}  {'λ (valeur propre)':>18}  "
          f"{'Var. (%)':>9}  {'Var. cum. (%)':>14}  {'||v|| (norme)':>13}")
    print("  " + "─"*66)

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    for i in range(min(n_show, pca.n_components_)):
        lam   = pca.explained_variance_[i]
        vpct  = pca.explained_variance_ratio_[i] * 100
        vnorm = np.linalg.norm(pca.components_[i])   # doit être ≈ 1 (unitaire)
        print(f"  {i+1:>4}  {lam:>18.4f}  {vpct:>9.3f}%  "
              f"{cumvar[i]:>14.2f}%  {vnorm:>13.6f}")

    print("  " + "─"*66)
    print("  Note : ||v|| ≈ 1.0 confirme que les vecteurs propres sont unitaires.")
    print("═"*70 + "\n")



#  NOUVEAU — PROJECTION SUR PAIRES D'AXES (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3)



def plot_pca_axes_pairs(X_pca, y, save_path="output_pca_axes_pairs.png"):
    """
    Représente les données dans plusieurs paires de composantes principales.

    Pourquoi plusieurs paires ?
    ────────────────────────────
    PC1 vs PC2 capture le plus de variance, mais d'autres paires peuvent
    révéler des structures cachées. En regardant PC1/PC2, PC1/PC3, PC2/PC3,
    on explore les trois premiers axes de l'espace réduit.

    Si les nuages Réel/Fake se séparent bien dans un plan → ce plan
    est pertinent pour la classification.
    """
    pairs  = [(0, 1), (0, 2), (1, 2)]
    labels = ["PC1", "PC2", "PC3"]
    colors = {1: "#3A86FF", 0: "#FF006E"}
    names  = {1: "Réel", 0: "Fake/Spoof"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Représentation dans les axes principaux choisis\n"
        "Réel (bleu) vs Fake (rouge) — Plan PC1/PC2, PC1/PC3, PC2/PC3",
        fontsize=13, fontweight="bold"
    )

    for ax, (i, j) in zip(axes, pairs):
        for cls in [1, 0]:
            mask = y == cls
            ax.scatter(
                X_pca[mask, i], X_pca[mask, j],
                c=colors[cls],
                label=f"{names[cls]} (n={mask.sum()})",
                alpha=0.35, s=15, edgecolors="none"
            )
        var_i = 0  # will compute from pca if passed; here shown as axis label only
        ax.set_xlabel(f"{labels[i]} →", fontsize=11)
        ax.set_ylabel(f"{labels[j]} →", fontsize=11)
        ax.set_title(f"Plan {labels[i]} × {labels[j]}", fontsize=11)
        ax.legend(markerscale=2, fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Paires d'axes sauvegardées → {save_path}")


def plot_pca_axes_pairs_with_variance(X_pca, y, pca,
                                      save_path="output_pca_axes_pairs.png"):
    """
    Comme plot_pca_axes_pairs mais ajoute le % de variance sur chaque axe.
    Version recommandée quand on a l'objet pca disponible.
    """
    pairs  = [(0, 1), (0, 2), (1, 2)]
    pc_labels = [
        f"PC{k+1} ({pca.explained_variance_ratio_[k]*100:.1f}%)"
        for k in range(min(3, pca.n_components_))
    ]
    colors = {1: "#3A86FF", 0: "#FF006E"}
    names  = {1: "Réel", 0: "Fake/Spoof"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Représentation des données dans les axes principaux (ACP)\n"
        "Réel (bleu) vs Fake (rouge)",
        fontsize=13, fontweight="bold"
    )

    for ax, (i, j) in zip(axes, pairs):
        for cls in [1, 0]:
            mask = y == cls
            ax.scatter(
                X_pca[mask, i], X_pca[mask, j],
                c=colors[cls],
                label=f"{names[cls]} (n={mask.sum()})",
                alpha=0.35, s=15, edgecolors="none"
            )
        ax.set_xlabel(pc_labels[i], fontsize=10)
        ax.set_ylabel(pc_labels[j], fontsize=10)
        ax.set_title(f"Plan {pc_labels[i].split(' ')[0]} × "
                     f"{pc_labels[j].split(' ')[0]}", fontsize=11)
        ax.legend(markerscale=2, fontsize=9)
        ax.grid(True, alpha=0.25)
        # Centroïdes
        for cls in [1, 0]:
            mask = y == cls
            cx = X_pca[mask, i].mean()
            cy = X_pca[mask, j].mean()
            ax.scatter(cx, cy, c=colors[cls], s=120, marker="*",
                       edgecolors="black", linewidths=0.8, zorder=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Paires d'axes (avec variance) sauvegardées → {save_path}")



#  VISUALISATION 1 : Variance expliquée



def plot_variance(pca, save_path="output_3_variance_pca.png"):
    """
    Deux graphiques :
      - Variance expliquée (%) par chaque composante (barplot)
      - Variance cumulée avec seuils 90 % et 95 % (courbe du coude)
    """
    n = len(pca.explained_variance_ratio_)
    cumvar  = np.cumsum(pca.explained_variance_ratio_) * 100
    var_each = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ACP — Analyse de la variance expliquée",
                 fontsize=14, fontweight="bold")

    # Barplot individuel
    axes[0].bar(range(1, n + 1), var_each,
                color="#3A86FF", edgecolor="none", alpha=0.85)
    axes[0].set_xlabel("Numéro de la composante principale")
    axes[0].set_ylabel("Variance expliquée (%)")
    axes[0].set_title("Variance par composante")
    axes[0].axhline(var_each.mean(), color="orange", linestyle="--",
                    label=f"Moyenne : {var_each.mean():.2f}%")
    axes[0].legend(fontsize=9)

    # Courbe cumulée
    axes[1].plot(range(1, n + 1), cumvar,
                 color="#FF006E", linewidth=2.5,
                 marker="o", markersize=3)
    axes[1].axhline(90, color="gray",   linestyle="--", label="Seuil 90%")
    axes[1].axhline(95, color="orange", linestyle="--", label="Seuil 95%")
    axes[1].axhline(99, color="red",    linestyle=":",  label="Seuil 99%")
    axes[1].set_xlabel("Nombre de composantes")
    axes[1].set_ylabel("Variance cumulée (%)")
    axes[1].set_title("Variance cumulée (courbe du coude)")
    axes[1].legend()
    axes[1].set_ylim(0, 105)

    # Annotations automatiques pour chaque seuil
    for threshold, color in [(90, "gray"), (95, "orange"), (99, "red")]:
        idx = np.argmax(cumvar >= threshold)
        if idx < n:
            axes[1].annotate(
                f"{idx + 1} PC\npour {threshold}%",
                xy=(idx + 1, cumvar[idx]),
                xytext=(idx + 1 + max(n // 10, 2), cumvar[idx] - 8),
                arrowprops=dict(arrowstyle="->", color=color),
                fontsize=8, color=color
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Figure sauvegardée → {save_path}")

    # Résumé console
    for t in [80, 90, 95, 99]:
        idx = np.argmax(cumvar >= t)
        print(f"   → {idx + 1:3d} composantes pour capturer {t}% de la variance")
    print(f"   → {n} composantes capturent {cumvar[-1]:.1f}% de la variance totale")



#  VISUALISATION 2 : Eigenfaces (les axes de l'ACP)



def plot_eigenfaces(pca, img_size=(64, 64), n_show=15,
                    save_path="output_4_eigenfaces.png"):
    """
    Affiche les n_show premières composantes principales (eigenfaces).

    Chaque eigenface est un vecteur de l'espace pixel qui représente
    une direction de variance maximale dans les données d'images.
    Les premières eigenfaces capturent le plus de variance.

    Couleurs RdBu : rouge = contribution positive, bleu = négative.
    """
    rows = (n_show + 4) // 5  # 5 colonnes
    fig, axes = plt.subplots(rows, 5, figsize=(14, rows * 2.8))
    fig.suptitle(f"Eigenfaces — {n_show} premières composantes principales de l'ACP\n"
                 "(Rouge = forte activation positive | Bleu = forte activation négative)",
                 fontsize=11, fontweight="bold")

    axes_flat = axes.flat if rows > 1 else [axes] if n_show == 1 else list(axes.flat)
    for i, ax in enumerate(axes_flat):
        if i < n_show:
            ef = pca.components_[i].reshape(img_size)
            im = ax.imshow(ef, cmap="RdBu_r")
            var_pct = pca.explained_variance_ratio_[i] * 100
            ax.set_title(f"PC {i + 1}\n({var_pct:.2f}%)", fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Figure sauvegardée → {save_path}")



#  VISUALISATION 3 : Projection 2D (PC1 vs PC2)



def plot_pca_2d(X_pca, y, title_suffix="Train",
                save_path="output_5_pca_2d.png"):
    """
    Projette les données sur PC1 et PC2.
    Si les nuages Réel/Fake sont bien séparés → classification facile.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {1: "#3A86FF", 0: "#FF006E"}
    names  = {1: "Réel", 0: "Fake/Spoof"}

    for cls in [1, 0]:
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[cls], label=f"{names[cls]} (n={mask.sum()})",
                   alpha=0.40, s=18, edgecolors="none")

    ax.set_xlabel("Composante principale 1 (PC1)", fontsize=12)
    ax.set_ylabel("Composante principale 2 (PC2)", fontsize=12)
    ax.set_title(f"Projection ACP 2D — Séparabilité Réel vs Fake\n({title_suffix})",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=2, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Figure sauvegardée → {save_path}")



#  VISUALISATION 4 : Projection 3D (PC1, PC2, PC3)



def plot_pca_3d(X_pca, y, title_suffix="Train",
                save_path="output_5b_pca_3d.png"):
    """
    Projection sur les 3 premières composantes — vue 3D interactive.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    colors = {1: "#3A86FF", 0: "#FF006E"}
    names  = {1: "Réel", 0: "Fake/Spoof"}

    for cls in [1, 0]:
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   c=colors[cls], label=names[cls],
                   alpha=0.35, s=12, edgecolors="none")

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"Projection ACP 3D ({title_suffix})",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Figure sauvegardée → {save_path}")



#  VISUALISATION 5 : Reconstruction d'images via ACP



def plot_reconstruction(X_train_sc, pca, scaler, img_size=(64, 64),
                        n_samples=4, save_path="output_5c_reconstruction.png"):
    """
    Montre la reconstruction de quelques images après ACP :
    comparer l'original et le reconstruit pour différents nombres de PC.

    Cela illustre la perte d'information selon n_components.
    """
    n_components_list = [5, 10, 20, 50, min(100, pca.n_components_)]
    n_components_list = sorted(set(n_components_list))

    fig, axes = plt.subplots(
        n_samples, len(n_components_list) + 1,
        figsize=(3 * (len(n_components_list) + 1), 3 * n_samples)
    )
    fig.suptitle("Reconstruction des images par ACP\n"
                 "(plus de composantes = meilleure fidélité)",
                 fontsize=12, fontweight="bold")

    for row in range(n_samples):
        x = X_train_sc[row]

        # Image originale (après standardisation, ramenée à [0,1] pour affichage)
        x_orig = scaler.inverse_transform(x.reshape(1, -1))[0] / 255.0
        x_orig = np.clip(x_orig, 0, 1)
        axes[row, 0].imshow(x_orig.reshape(img_size), cmap="gray")
        axes[row, 0].set_title("Original" if row == 0 else "", fontsize=9)
        axes[row, 0].axis("off")

        for col, k in enumerate(n_components_list):
            # Reconstruit avec k composantes seulement
            x_proj   = pca.components_[:k] @ x           # projection
            x_rec_sc = pca.components_[:k].T @ x_proj     # retour dans espace pixel
            x_rec = scaler.inverse_transform(x_rec_sc.reshape(1, -1))[0] / 255.0
            x_rec = np.clip(x_rec, 0, 1)
            axes[row, col + 1].imshow(x_rec.reshape(img_size), cmap="gray")
            axes[row, col + 1].set_title(f"{k} PC" if row == 0 else "", fontsize=9)
            axes[row, col + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Figure sauvegardée → {save_path}")



#  VISUALISATION 6 : Distribution des scores sur PC1 et PC2



def plot_pc_distributions(X_pca, y, save_path="output_5d_pc_distributions.png"):
    """
    Histogrammes des projections sur PC1 et PC2 par classe.
    Si les distributions se séparent bien → bonne discriminabilité.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribution des projections sur PC1 et PC2 par classe",
                 fontsize=13, fontweight="bold")

    labels_map = {1: ("Réel", "#3A86FF"), 0: ("Fake", "#FF006E")}

    for ax, pc_idx, pc_name in zip(axes, [0, 1], ["PC1", "PC2"]):
        for cls, (name, color) in labels_map.items():
            mask = y == cls
            ax.hist(X_pca[mask, pc_idx], bins=60, alpha=0.55,
                    color=color, label=name, density=True)
        ax.set_xlabel(f"Score sur {pc_name}")
        ax.set_ylabel("Densité")
        ax.set_title(f"Distribution sur {pc_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Figure sauvegardée → {save_path}")
    
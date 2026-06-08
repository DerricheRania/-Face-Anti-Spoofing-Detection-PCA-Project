import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)



#  ÉTAPE 5 — ÉVALUATION DES CLASSIFIEURS



def evaluate_one(name, clf, X_test, y_test, ax_cm, ax_roc, ax_bar):
    """
    Évalue un classifieur et remplit 3 sous-graphiques :
      ax_cm  : matrice de confusion
      ax_roc : courbe ROC
      ax_bar : précision / rappel / F1 par classe
    """
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)

    print(f"\n  {'─' * 52}")
    print(f"  {name}")
    print(f"  {'─' * 52}")
    print(f"  Accuracy sur le TEST : {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred,
                                target_names=["Fake", "Reel"]))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Reel"],
                yticklabels=["Fake", "Reel"],
                ax=ax_cm, cbar=False)
    ax_cm.set_title(f"{name}\nMatrice de confusion")
    ax_cm.set_ylabel("Vrai label")
    ax_cm.set_xlabel("Label predit")

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color="#3A86FF", lw=2,
                label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], color="lightgray", linestyle="--")
    ax_roc.fill_between(fpr, tpr, alpha=0.08, color="#3A86FF")
    ax_roc.set_xlabel("Taux de faux positifs (FPR)")
    ax_roc.set_ylabel("Taux de vrais positifs (TPR)")
    ax_roc.set_title(f"{name}\nCourbe ROC")
    ax_roc.legend(loc="lower right")

    # Metriques par classe
    report = classification_report(y_test, y_pred,
                                   target_names=["Fake", "Reel"],
                                   output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(metrics))
    w = 0.32
    ax_bar.bar(x - w / 2, [report["Fake"][m] for m in metrics],
               w, label="Fake", color="#FF006E", alpha=0.85)
    ax_bar.bar(x + w / 2, [report["Reel"][m] for m in metrics],
               w, label="Reel", color="#3A86FF", alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Precision", "Rappel", "F1-score"])
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_title(f"{name}\nMetriques par classe")
    ax_bar.set_ylabel("Score")
    ax_bar.legend()

    return acc, roc_auc


def evaluate_all(classifiers, X_test, y_test,
                 save_detail="output_6_evaluation.png",
                 save_compare="output_7_comparaison.png"):
    """
    Evalue tous les classifieurs et produit :
      - Grille detaillee (confusion + ROC + metriques) par modele
      - Graphique comparatif final Accuracy & AUC (toutes les courbes ROC)

    Retourne
    --------
    dict { nom : {"accuracy": float, "auc": float} }
    """
    n   = len(classifiers)
    fig = plt.figure(figsize=(17, 5 * n))
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.5, wspace=0.38)

    results = {}
    for row, (name, clf) in enumerate(classifiers.items()):
        ax_cm  = fig.add_subplot(gs[row, 0])
        ax_roc = fig.add_subplot(gs[row, 1])
        ax_bar = fig.add_subplot(gs[row, 2])
        acc, roc_auc = evaluate_one(name, clf, X_test, y_test,
                                    ax_cm, ax_roc, ax_bar)
        results[name] = {"accuracy": acc, "auc": roc_auc}

    plt.suptitle("Evaluation detaillee des classifieurs",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(save_detail, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"   Figure sauvegardee -> {save_detail}")

    # ── Comparaison finale ──────────────────────────────────────────
    _plot_comparison(results, classifiers, X_test, y_test, save_compare)

    return results


def _plot_comparison(results, classifiers, X_test, y_test, save_path):
    """
    Figure comparative avec :
      - Accuracy par modele (barplot)
      - AUC-ROC par modele (barplot)
      - Toutes les courbes ROC superposees
    """
    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]
    aucs  = [results[n]["auc"]            for n in names]

    # Palette de couleurs distinctes pour N modeles
    palette = ["#3A86FF", "#FF006E", "#06D6A0", "#FFB703", "#8338EC",
               "#FB5607", "#023047", "#E63946"]
    colors = palette[:len(names)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Comparaison finale des classifieurs",
                 fontsize=14, fontweight="bold")

    # -- Accuracy --
    bars = axes[0].bar(names, accs, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy sur le test")
    axes[0].set_ylim(0, max(accs) * 1.18)
    axes[0].tick_params(axis="x", rotation=20)
    for b, v in zip(bars, accs):
        axes[0].text(b.get_x() + b.get_width() / 2,
                     b.get_height() + 0.3,
                     f"{v:.1f}%", ha="center", va="bottom",
                     fontweight="bold", fontsize=9)

    # -- AUC --
    bars2 = axes[1].bar(names, aucs, color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("AUC-ROC sur le test")
    axes[1].set_ylim(0, min(max(aucs) * 1.18, 1.0))
    axes[1].tick_params(axis="x", rotation=20)
    for b, v in zip(bars2, aucs):
        axes[1].text(b.get_x() + b.get_width() / 2,
                     b.get_height() + 0.002,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontweight="bold", fontsize=9)

    # -- Toutes les courbes ROC superposees --
    axes[2].plot([0, 1], [0, 1], color="lightgray", linestyle="--", lw=1)
    for (name, clf), color in zip(classifiers.items(), colors):
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_test_arr = y_test
        fpr, tpr, _ = roc_curve(y_test_arr, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[2].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name} (AUC={roc_auc:.3f})")
    axes[2].set_xlabel("FPR")
    axes[2].set_ylabel("TPR")
    axes[2].set_title("Courbes ROC superposees")
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].grid(True, alpha=0.3)

    # Meilleur modele
    best = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\n   Meilleur modele : {best}")
    print(f"      Accuracy : {results[best]['accuracy'] * 100:.2f}%")
    print(f"      AUC-ROC  : {results[best]['auc']:.3f}")

    # Classement
    print("\n   Classement par Accuracy sur le TEST :")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
    for rank, (name, metrics) in enumerate(ranked, 1):
        print(f"      {rank}. {name:<25} "
              f"Acc={metrics['accuracy']*100:.2f}%  "
              f"AUC={metrics['auc']:.3f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"   Figure sauvegardee -> {save_path}")


def predict_image(image_path, clf, pca, scaler, img_size=(64, 64)):
    """
    Predit si une image est Reelle ou Fake.

    Exemple d'appel dans main.py :
        predict_image("photo.jpg", classifiers["SVM (RBF)"], pca, scaler)
    """
    img   = Image.open(image_path).convert("L").resize(img_size)
    x     = np.array(img, dtype=np.float32).flatten() / 255.0
    x_sc  = scaler.transform([x])
    x_pca = pca.transform(x_sc)

    label_id = clf.predict(x_pca)[0]
    proba    = clf.predict_proba(x_pca)[0]
    label    = "Reel" if label_id == 1 else "Fake"
    conf     = proba[label_id] * 100

    plt.figure(figsize=(4, 4))
    plt.imshow(np.array(img), cmap="gray")
    plt.title(f"Prediction : {label}\nConfiance : {conf:.1f}%",
              color="green" if label == "Reel" else "red",
              fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"\n   Image     : {image_path}")
    print(f"   Resultat  : {label}")
    print(f"   Confiance : {conf:.1f}%")
    return label, conf

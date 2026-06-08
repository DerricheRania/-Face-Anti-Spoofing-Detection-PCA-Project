import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



#  ÉTAPE 4 — ENTRAÎNEMENT DES CLASSIFIEURS



def train_svm(X_train, y_train, C=10, random_state=42):
    """
    SVM à noyau RBF (Radial Basis Function).

    Cherche l'hyperplan séparateur à marge maximale dans l'espace
    réduit par l'ACP. Le noyau RBF capture des frontières non-linéaires
    entre Réel et Fake.

    Paramètres
    ----------
    C            : pénalité sur les erreurs (grand C = moins de tolérance)
    kernel='rbf' : noyau gaussien
    gamma='scale': γ = 1 / (n_features × Var(X))
    probability  : active predict_proba() pour la courbe ROC
    """
    clf = SVC(kernel="rbf", C=C, gamma="scale",
              probability=True, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """
    Random Forest — ensemble de 200 arbres de décision (bagging).

    Chaque arbre est entraîné sur un sous-ensemble aléatoire des données
    et des features. La décision finale est le vote majoritaire.
    Robuste au surapprentissage grâce à la diversité des arbres.

    Paramètres
    ----------
    n_estimators : nombre d'arbres dans la forêt
    n_jobs=-1    : utilise tous les cœurs CPU disponibles
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Régression Logistique — modèle linéaire probabiliste (baseline).

    Simple mais souvent efficace après ACP car les composantes sont
    déjà décorrélées. Utilise la sigmoïde pour produire des probabilités.

    Paramètres
    ----------
    C=1.0        : inverse de la régularisation L2
    max_iter=500 : augmenté pour assurer la convergence sur les données ACP
    solver='lbfgs': optimiseur adapté aux petits datasets (après ACP)
    """
    clf = LogisticRegression(
        C=1.0,
        max_iter=500,
        solver="lbfgs",
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf


def train_gradient_boosting(X_train, y_train, random_state=42):
    """
    Gradient Boosting — arbres construits séquentiellement.

    Contrairement au Random Forest (arbres parallèles indépendants),
    chaque arbre corrige les erreurs du précédent. Très performant
    sur des features structurées comme les composantes ACP.

    Paramètres
    ----------
    n_estimators=200  : nombre d'itérations de boosting
    learning_rate=0.1 : taux d'apprentissage (shrinkage)
    max_depth=4       : profondeur max de chaque arbre de base
    subsample=0.8     : fraction des données (stochastic GB)
    """
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf


def train_mlp(X_train, y_train, random_state=42):
    """
    MLP (Multi-Layer Perceptron) — réseau de neurones dense.

    Architecture : couches cachées (256 → 128) avec activation ReLU.
    Régularisation L2 via alpha. Early stopping pour éviter le
    surapprentissage sur le set de validation interne.

    Paramètres
    ----------
    hidden_layer_sizes : architecture (256, 128)
    activation='relu'  : fonction d'activation
    alpha=1e-4         : régularisation L2
    early_stopping=True: arrêt si la val. ne s'améliore plus
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf


def train_all_classifiers(X_train, y_train, random_state=42):
    """
    Entraîne tous les classifieurs et affiche les accuracies sur le train.

    Modèles entraînés
    -----------------
    - SVM (RBF)           : classifieur à vaste marge, noyau non-linéaire
    - Random Forest       : ensemble d'arbres par bagging
    - Logistic Regression : baseline linéaire probabiliste
    - Gradient Boosting   : boosting séquentiel d'arbres
    - MLP Neural Network  : réseau de neurones dense (256 -> 128)

    Retourne
    --------
    dict { nom_affichage : objet classifieur sklearn }
    """
    pipeline = [
        ("SVM (RBF)",           train_svm,                 {"random_state": random_state}),
        ("Random Forest",       train_random_forest,       {"random_state": random_state}),
        ("Logistic Regression", train_logistic_regression, {"random_state": random_state}),
        ("Gradient Boosting",   train_gradient_boosting,   {"random_state": random_state}),
        ("MLP Neural Network",  train_mlp,                 {"random_state": random_state}),
    ]

    trained = {}
    for name, fn, kwargs in pipeline:
        print(f"\n   Entrainement {name} ...")
        clf = fn(X_train, y_train, **kwargs)
        acc = accuracy_score(y_train, clf.predict(X_train))
        print(f"      -> Accuracy sur train : {acc * 100:.1f}%")
        trained[name] = clf

    print("\n   Note : l'accuracy sur train seule ne suffit pas.")
    print("      L'evaluation sur le TEST mesure la vraie generalisation.")

    return trained

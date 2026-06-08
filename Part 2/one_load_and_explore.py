import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Affiche une barre de progression si tqdm est installé, sinon boucle simple
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    print("   ⚠️  tqdm non installé — installez-le via : pip install tqdm")


#  ÉTAPE 1 : CHARGEMENT ET EXPLORATION DU DATASET


def _progress(iterable, desc="", total=None):
    """Wrapper autour de tqdm ou d'une boucle simple."""
    if _HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total,
                    ncols=80, unit="img", colour="cyan")
    else:
        print(f"   {desc} ({total} images) ...")
        return iterable


def load_images_from_folder(folder, label, max_images=None):
    """
    Charge les images d'un dossier et retourne des vecteurs 1D.

    Paramètres
    ----------
    folder     : chemin du dossier (ex: LCC_FASD/real)
    label      : 1 pour Réel, 0 pour Fake/Spoof
    max_images : nombre max d'images à charger (None = toutes)

    Retourne
    --------
    images : tableau numpy (N, IMG_H * IMG_W)
    labels : tableau numpy (N,)
    """
    images, labels = [], []
    extensions = (".jpg", ".jpeg", ".png", ".bmp")

    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(extensions)])
    if max_images is not None:
        files = files[:max_images]

    for filename in _progress(files, desc=f"Chargement {folder}", total=len(files)):
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert("L")       # niveaux de gris
            img = img.resize((64, 64))                 # redimensionnement
            arr = np.array(img, dtype=np.float32)      # tableau numpy
            images.append(arr.flatten())               # aplatissement 1D
            labels.append(label)
        except KeyboardInterrupt:
            # L'utilisateur a appuyé Ctrl+C : on retourne ce qu'on a déjà
            print("\n   ⚠️  Interruption clavier : retour partiel des données chargées.")
            break
        except Exception as e:
            print(f"   ⚠️  Erreur sur {filename} : {e}")

    return np.array(images), np.array(labels)


def load_balanced_dataset(real_dir, fake_dir, img_size=(64, 64)):
    """
    Charge le dataset de façon équilibrée :
    - Charge toutes les images réelles (2 254)
    - Échantillonne aléatoirement le même nombre depuis les fakes (7 046 → 2 254)

    FIX DU BUG : le KeyboardInterrupt était propagé hors de PIL.Image.open()
    pendant le chargement lent des images. On l'attrape maintenant à chaque
    fichier et on affiche une barre de progression (tqdm) pour rendre
    le chargement visible et interruptible proprement.

    Retourne
    --------
    X       : (N_total, H*W)
    y       : (N_total,)
    X_real  : images réelles seules (pour la visu)
    X_fake  : images fake sélectionnées (pour la visu)
    """
    extensions = (".jpg", ".jpeg", ".png", ".bmp")

    # ── Réel : toutes les images ─────────────────────────────────
    real_files = sorted([f for f in os.listdir(real_dir)
                         if f.lower().endswith(extensions)])
    n_real = len(real_files)
    print(f"   Dossier réel  : {n_real} images trouvées")

    # ── Fake : on en prend exactement n_real (sous-échantillonnage équilibré) ──
    fake_files_all = sorted([f for f in os.listdir(fake_dir)
                              if f.lower().endswith(extensions)])
    rng = np.random.default_rng(42)
    chosen_idx = rng.choice(len(fake_files_all), size=n_real, replace=False)
    chosen_idx.sort()
    fake_files = [fake_files_all[i] for i in chosen_idx]
    print(f"   Dossier fake  : {len(fake_files_all)} images trouvées "
          f"→ {n_real} sélectionnées pour l'équilibre")

    def _read(folder, filelist, label):
        images, labs = [], []
        for fname in _progress(filelist,
                               desc=f"  {'Réel' if label==1 else 'Fake'}",
                               total=len(filelist)):
            try:
                img = Image.open(os.path.join(folder, fname)).convert("L")
                img = img.resize(img_size)
                images.append(np.array(img, dtype=np.float32).flatten())
                labs.append(label)
            except KeyboardInterrupt:
                # Interruption propre : on retourne ce qui est déjà chargé
                print("\n   ⚠️  Interruption — données partielles retournées.")
                break
            except Exception as e:
                print(f"   ⚠️  Erreur {fname} : {e}")
        return np.array(images), np.array(labs)

    print("\n   Chargement des images réelles ...")
    X_real, y_real = _read(real_dir, real_files, label=1)

    print("\n   Chargement des images fake ...")
    X_fake, y_fake = _read(fake_dir, fake_files, label=0)

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([y_real, y_fake])

    print(f"\n   Dataset total : {X.shape[0]} images")
    print(f"   Distribution  : {np.sum(y==1)} Réel  |  {np.sum(y==0)} Fake  "
          f"(ratio 1:1 ✅)")

    return X, y, X_real, X_fake


def explore_dataset(X_real, X_fake, img_size=(64, 64),
                    save_path="output_1_exemples.png"):
    """
    Affiche 5 exemples de chaque classe côte à côte.
    """
    fig, axes = plt.subplots(2, 5, figsize=(13, 5))
    fig.suptitle("Exemples du dataset\nLigne 1 : Réel  |  Ligne 2 : Fake",
                 fontsize=13, fontweight="bold")

    for i in range(5):
        axes[0, i].imshow(X_real[i].reshape(img_size), cmap="gray")
        axes[0, i].set_title("Réel", color="green", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(X_fake[i].reshape(img_size), cmap="gray")
        axes[1, i].set_title("Fake", color="red", fontsize=10)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"   ✅ Figure sauvegardée → {save_path}")
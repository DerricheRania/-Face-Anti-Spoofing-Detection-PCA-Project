import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image

IMAGE_PATH = "image_real.png"          
MODEL_NAME = "mlp_neural_network_model.pkl"
IMG_SIZE = (64, 64)


# 1. Chargement et prétraitement (même pipeline que l'entraînement)
img_pil  = Image.open(IMAGE_PATH).convert("L")        # niveaux de gris
img_pil  = img_pil.resize(IMG_SIZE)                   # 64×64
arr      = np.array(img_pil, dtype=np.float32)
X        = arr.flatten() / 255.0                      # normalisation [0,1]
X        = X.reshape(1, -1)                           # shape (1, 4096)

# 2. Chargement des modèles
scaler = joblib.load("scaler.pkl")
pca    = joblib.load("pca_model.pkl")
model  = joblib.load(MODEL_NAME)

# 3. Transformation
X_scaled = scaler.transform(X)
X_pca    = pca.transform(X_scaled)

# 4. Prédiction
pred  = model.predict(X_pca)[0]          # 0 = Spoof, 1 = Réel
label = "RÉEL" if pred == 1 else "SPOOF"
color = "#3A86FF" if pred == 1 else "#FF006E"
icon  = "✅" if pred == 1 else "❌"

proba_txt = ""
if hasattr(model, "predict_proba"):
    p = model.predict_proba(X_pca)[0]
    proba_txt = f"\nConfiance : {max(p)*100:.1f}%"

# 5. Affichage
img_original = Image.open(IMAGE_PATH).convert("RGB")

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(img_original)
ax.axis("off")
ax.set_title(
    f"{icon}  {label}{proba_txt}\n[{MODEL_NAME.replace('_model.pkl','')}]",
    fontsize=14, fontweight="bold", color=color,
    bbox=dict(boxstyle="round,pad=0.4", fc="#0f0f1a", ec=color, lw=2)
)
fig.patch.set_facecolor("#0f0f1a")
plt.tight_layout()
plt.savefig("resultat_test.png", dpi=120, facecolor=fig.get_facecolor())
plt.show()

print(f"\n  {icon}  Prédiction : {label}  —  {MODEL_NAME}{proba_txt}")
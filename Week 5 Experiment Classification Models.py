import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(42)
N = 10000

# True boundary functions: class 1 when f(x1,x2) > 0 (with a little label noise)
TRUE = {
    "linear":    lambda x1, x2: 1.5*x1 - x2 + 0.5,
    "quadratic": lambda x1, x2: 4 - (x1**2 + x2**2),        # circle radius 2
    "cubic":     lambda x1, x2: x2 - (0.5*x1**3 - 1.5*x1),  # cubic curve
}

def make_data(kind):
    X = rng.uniform(-3, 3, size=(N, 2))
    f = TRUE[kind](X[:, 0], X[:, 1])
    p = 1/(1+np.exp(-6*f/np.std(f)))        # sharp but slightly noisy
    y = rng.binomial(1, p)
    return X, y

def models():
    return {
        "Logistic (linear features)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Logistic (quadratic features)": make_pipeline(PolynomialFeatures(2), StandardScaler(), LogisticRegression(max_iter=2000)),
        "Logistic (cubic features)": make_pipeline(PolynomialFeatures(3), StandardScaler(), LogisticRegression(max_iter=5000)),
        "KNN (k=1)": KNeighborsClassifier(1),
        "KNN (k=15)": KNeighborsClassifier(15),
        "KNN (k=200)": KNeighborsClassifier(200),
    }

rows = []
gx, gy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
grid = np.c_[gx.ravel(), gy.ravel()]

for kind in TRUE:
    X, y = make_data(kind)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    ms = models()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, m) in zip(axes.ravel(), ms.items()):
        m.fit(Xtr, ytr)
        acc = m.score(Xte, yte)
        rows.append((kind, name, acc))
        Z = m.predict(grid).reshape(gx.shape)
        ax.contourf(gx, gy, Z, alpha=0.25, cmap="coolwarm")
        ax.contour(gx, gy, TRUE[kind](gx, gy), levels=[0], colors="k", linewidths=2)
        ax.scatter(Xte[:300, 0], Xte[:300, 1], c=yte[:300], cmap="coolwarm", s=8, edgecolor="none")
        ax.set_title(f"{name}\ntest acc = {acc:.4f}")
    fig.suptitle(f"{kind} true boundary (black line) vs. fitted decision regions", fontsize=14)
    plt.tight_layout(); plt.savefig(f"boundary_{kind}.png", dpi=110); plt.close()

df = pd.DataFrame(rows, columns=["Boundary", "Model", "Test Accuracy"])
df.to_csv("results.csv", index=False)
print(df.pivot(index="Model", columns="Boundary", values="Test Accuracy").round(4).to_string())

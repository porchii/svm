
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_digits, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVR


class SMOBinarySVM:

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", tol=1e-3,
                 max_passes=10, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter

    def _kernel_matrix(self, X, Y=None):
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        if self.kernel == "linear":
            return X @ Y.T
        if self.kernel == "rbf":
            gamma = (1.0 / X.shape[1] if self.gamma == "scale"
                     else float(self.gamma))
            sq = np.sum(X * X, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - 2 * X @ Y.T
            return np.exp(-gamma * np.maximum(sq, 0.0))
        raise ValueError("kernel должен быть 'linear' или 'rbf'")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m = X.shape[0]
        self.X_train = X
        self.y_train = y
        K = self._kernel_matrix(X)
        alpha = np.zeros(m)
        b = 0.0
        passes = 0
        iterations = 0

        while passes < self.max_passes and iterations < self.max_iter:
            num_changed = 0
            for i in range(m):
                f_i = float(np.sum(alpha * y * K[:, i]) + b)
                E_i = f_i - y[i]
                if (y[i] * E_i < -self.tol and alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and alpha[i] > 0):
                    non_bound = np.where((alpha > 1e-5) & (alpha < self.C - 1e-5))[0]
                    candidates = non_bound if len(non_bound) else np.arange(m)
                    j = candidates[np.argmax(np.abs([
                        (float(np.sum(alpha * y * K[:, k]) + b) - y[k]) - E_i
                        for k in candidates
                    ]))]
                    if j == i:
                        j = int(np.random.randint(0, m - 1) if i == 0
                                else (i + 1) % m)

                    a_i, a_j = alpha[i], alpha[j]
                    if y[i] != y[j]:
                        L = max(0.0, a_j - a_i)
                        H = min(self.C, self.C + a_j - a_i)
                    else:
                        L = max(0.0, a_i + a_j - self.C)
                        H = min(self.C, a_i + a_j)
                    if H - L < 1e-12:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    f_j = float(np.sum(alpha * y * K[:, j]) + b)
                    E_j = f_j - y[j]
                    j_new = a_j - y[j] * (E_i - E_j) / eta
                    j_new = min(H, max(L, j_new))
                    if abs(j_new - a_j) < 1e-5:
                        continue

                    i_new = a_i + y[i] * y[j] * (a_j - j_new)
                    b1 = b - E_i - y[i] * (i_new - a_i) * K[i, i] \
                        - y[j] * (j_new - a_j) * K[i, j]
                    b2 = b - E_j - y[i] * (i_new - a_i) * K[i, j] \
                        - y[j] * (j_new - a_j) * K[j, j]
                    if 0 < i_new < self.C:
                        b = b1
                    elif 0 < j_new < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alpha[i], alpha[j] = i_new, j_new
                    num_changed += 1

            iterations += 1
            passes = passes + 1 if num_changed == 0 else 0

        self.alpha = alpha
        self.b = b
        self.support_vectors_ = X[alpha > 1e-5]
        self.support_vector_alpha_ = alpha[alpha > 1e-5]
        self.support_vector_labels_ = y[alpha > 1e-5]
        self.n_support_ = len(self.support_vectors_) 
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        K = self._kernel_matrix(X, self.X_train)
        sv = self.alpha > 1e-5
        return (K[:, sv] @ (self.alpha[sv] * self.y_train[sv])) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def kkt_violations(self, tol=1e-3):
        margins = self.y_train * self.decision_function(self.X_train)
        violations = 0
        for a, margin in zip(self.alpha, margins):
            if a < tol and margin < 1 - tol:
                violations += 1
            elif a > self.C - tol and margin > 1 + tol:
                violations += 1
            elif tol < a < self.C - tol and abs(margin - 1) > tol:
                violations += 1
        return violations


def plot_binary_experiment(results_dir):
    X, y = make_moons(n_samples=300, noise=0.18, random_state=42)
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    kernels = [("linear", 1.0), ("rbf", 0.5), ("rbf", 2.0)]
    print("\nЗадание 1: бинарный SVM")
    for ax, (kernel, param) in zip(axes, kernels):
        model = SMOBinarySVM(C=1.0, kernel=kernel,
                             gamma=(param if kernel == "rbf" else "scale"))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        kkt = model.kkt_violations()
        print(f"  {kernel}, parameter={param}: accuracy={acc:.3f}, "
              f"опорных векторов={model.n_support_}, нарушений KKT={kkt}")

        zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-np.inf, -1, 0, 1, np.inf],
                    colors=["#d8e6fb", "#f7f7f7", "#f7d8d8"], alpha=.8)
        ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["gray", "k", "gray"])
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", edgecolor="k", s=20)
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, facecolors="none", edgecolors="k", linewidths=1)
        ax.set_title(f"{kernel}, param={param}\nacc={acc:.3f}, SV={model.n_support_}")

    fig.tight_layout()
    fig.savefig(results_dir / "binary_svm.png", dpi=160)
    plt.close(fig)


def plot_svr_experiment(results_dir):
    rng = np.random.default_rng(42)
    X = np.linspace(-3, 3, 120).reshape(-1, 1)
    y = np.sinc(X[:, 0]) + rng.normal(0, 0.08, len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                        random_state=42)

    param_grid = {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.05, 0.1, 0.2],
                  "gamma": [0.05, 0.2, 1.0], "kernel": ["rbf"]}
    grid = GridSearchCV(SVR(), param_grid, cv=5,
                        scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    print("\nЗадание 2: epsilon-SVR")
    print("  Лучшие параметры:", grid.best_params_)
    print("  Количество опорных векторов:", len(best.support_),
          "из", len(X_train))

    X_sort = np.sort(X, axis=0)
    prediction = best.predict(X_sort)
    print(f"  MSE на тесте: {np.mean((best.predict(X_test)-y_test)**2):.5f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_train, y_train, color="blue", s=20, label="обучение")
    ax.scatter(X_test, y_test, color="orange", s=20, label="тест")
    ax.plot(X_sort, prediction, color="red", lw=2, label="SVR")
    ax.plot(X_sort, prediction + best.epsilon, "--", color="gray", lw=1)
    ax.plot(X_sort, prediction - best.epsilon, "--", color="gray", lw=1,
            label=r"$\epsilon$-труба")
    ax.scatter(X_train[best.support_, 0], y_train[best.support_], facecolors="none",
               edgecolors="k", s=55, label="опорные векторы")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("epsilon-SVR")
    fig.tight_layout()
    fig.savefig(results_dir / "epsilon_svr.png", dpi=160)
    plt.close(fig)


def crammer_singer_objective(theta, X, y, C, n_classes):
    n_features = X.shape[1]
    W = theta[:n_features * n_classes].reshape(n_features, n_classes)
    b = theta[n_features * n_classes:].reshape(n_classes)
    scores = X @ W + b
    true_scores = scores[np.arange(len(y)), y]

    # Δ(y_i, k)=0 для истинного класса и 1 для остальных.
    competing_scores = scores + (1 - np.eye(n_classes)[y])
    max_competing = np.max(competing_scores, axis=1)
    losses = np.maximum(0.0, max_competing - true_scores)

    return 0.5 * np.sum(W * W) + C * np.mean(losses)


def solve_crammer_singer(X, y, C=1.0):
    X = np.asarray(X)
    y = np.asarray(y)
    n_classes = int(y.max() + 1)
    n_features = X.shape[1]
    theta0 = np.zeros(n_features * n_classes + n_classes)
    result = minimize(crammer_singer_objective, theta0,
                      args=(X, y, C, n_classes), method="L-BFGS-B",
                      options={"maxiter": 500})
    W = result.x[:n_features * n_classes].reshape(n_features, n_classes)
    b = result.x[n_features * n_classes:].reshape(n_classes)

    def predict(Z):
        return np.argmax(np.asarray(Z) @ W + b, axis=1)

    predict.W = W
    predict.b = b
    return predict, result


def plot_multiclass_experiment(results_dir):
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    cs_predict, cs_result = solve_crammer_singer(X_train, y_train, C=1.0)
    cs_train_acc = accuracy_score(y_train, cs_predict(X_train))
    cs_test_acc = accuracy_score(y_test, cs_predict(X_test))

    ovr = LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=42)
    ovr.fit(X_train, y_train)
    ovr_train_acc = accuracy_score(y_train, ovr.predict(X_train))
    ovr_test_acc = accuracy_score(y_test, ovr.predict(X_test))

    print("\nЗадание 3: Crammer-Singer / многоклассовый SVM")
    print(f"  Совместная прямая задача: train={cs_train_acc:.3f}, test={cs_test_acc:.3f}")
    print(f"  scipy success={cs_result.success}, iterations={cs_result.nit}")
    print(f"  LinearSVC One-vs-Rest: train={ovr_train_acc:.3f}, test={ovr_test_acc:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cs_predict.W[:, :10] if cs_predict.W.shape[1] == 10 else cs_predict.W,
                   aspect="auto", cmap="RdBu")
    axes[0].set_title("Матрица весов Crammer-Singer")
    axes[0].set_xlabel("класс")
    axes[0].set_ylabel("признак")

    labels = ["CS train", "CS test", "OvR train", "OvR test"]
    values = [cs_train_acc, cs_test_acc, ovr_train_acc, ovr_test_acc]
    axes[1].bar(labels, values, color=["#4c72b0", "#64b5cd", "#55a868", "#9ecf7a"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Сравнение качества")
    fig.tight_layout()
    fig.savefig(results_dir / "multiclass_svm.png", dpi=160)
    plt.close(fig)


def main():
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot_binary_experiment(results_dir)
    plot_svr_experiment(results_dir)
    plot_multiclass_experiment(results_dir)
    print("\nГотово. Графики сохранены в:", results_dir)


if __name__ == "__main__":
    main()

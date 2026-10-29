from pathlib import Path
import os
import warnings

# На Windows joblib иногда пытается определить число физических ядер
# через системную команду и печатает traceback, хотя выполнение не падает.
# Фиксируем число ядер заранее, чтобы убрать это служебное сообщение.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
# Используем backend Agg, чтобы графики сохранялись в файлы даже без окна GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")
SEED = 42
N_CLUSTERS = 9


# ============================================================
# ШАГ 1. Пути к данным и папкам с результатами
# ============================================================
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
figure_dir = Path(output_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# ШАГ 2. Загрузка и первичное изучение данных
# ============================================================
# train.npz содержит изображения и метки prediction.
# test_features.npz содержит изображения и image_id, но не содержит
# меток классов, поэтому используется только для итогового прогноза.
train_data = np.load('train.npz', allow_pickle=False)
test_data = np.load('test_features.npz', allow_pickle=False)

images = train_data["images"]
labels = train_data["prediction"]
test_images = test_data["images"]
test_image_ids = test_data["image_id"]

print("=== Шаг 1. Исследование данных ===")
print(f"Размер обучающих изображений: {images.shape}")
print(f"Размер тестовых изображений:   {test_images.shape}")
print(f"Количество классов:             {len(np.unique(labels))}")
print(f"Распределение классов:          {np.bincount(labels)}")

# ============================================================
# ШАГ 3. Подготовка признаков
# ============================================================
# Каждое изображение имеет размер 28 x 28 x 3.
# K-Means получает на вход двумерную матрицу:
# одна строка = одно изображение, один столбец = один пиксельный признак.
#
# Делим значения пикселей на 255, чтобы привести их к диапазону [0, 1].
# Метки labels намеренно не включаются в X и не передаются в K-Means.
X = images.reshape(len(images), -1).astype(np.float32) / 255.0
X_test = test_images.reshape(len(test_images), -1).astype(np.float32) / 255.0

print("\n=== Шаг 2. Подготовка признаков ===")
print(f"Матрица X:      {X.shape}")
print(f"Матрица X_test: {X_test.shape}")
print(f"Диапазон X:     [{X.min():.1f}, {X.max():.1f}]")

# ============================================================
# ШАГ 4. Обучение K-Means без учителя
# ============================================================
# В условии требуется использовать 9 кластеров.
# n_init=10 означает, что алгоритм запустится с 10 начальными
# вариантами центроидов и выберет лучший результат.
#
# Здесь labels НЕ используются: кластеризация полностью unsupervised.
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    n_init=10,
)
cluster_labels = kmeans.fit_predict(X)

print("\n=== Шаг 3. Обучение K-Means ===")
print(f"Количество кластеров: {N_CLUSTERS}")
print(f"Количество итераций:   {kmeans.n_iter_}")
print(f"Итоговая инерция:      {kmeans.inertia_:.2f}")
print(f"Размеры кластеров:     {np.bincount(cluster_labels, minlength=N_CLUSTERS)}")

# ============================================================
# ШАГ 5. Оценка кластеризации
# ============================================================
# После завершения обучения можно использовать открытые labels
# для оценки результата.
#
# ARI и NMI сравнивают найденное разбиение с известными классами.
# Silhouette оценивает компактность и разделимость кластеров
# без использования истинных меток.
ari = adjusted_rand_score(labels, cluster_labels)
nmi = normalized_mutual_info_score(labels, cluster_labels)

# Полный silhouette может быть дорогим на больших наборах данных.
# Поэтому используем воспроизводимую случайную подвыборку из 1000 объектов.
rng = np.random.default_rng(SEED)
sample_size = min(1000, len(X))
sample_indices = rng.choice(len(X), size=sample_size, replace=False)
silhouette = silhouette_score(
    X[sample_indices],
    cluster_labels[sample_indices],
)

print("\n=== Шаг 4. Метрики кластеризации ===")
print(f"Adjusted Rand Index (ARI):       {ari:.4f}")
print(f"Normalized Mutual Info (NMI):    {nmi:.4f}")
print(f"Silhouette Score (n={sample_size}): {silhouette:.4f}")

# ============================================================
# ШАГ 6. Предсказание кластеров для тестовой выборки
# ============================================================
# Для тестовых изображений используем уже обученные центроиды.
# Тестовые метки классов недоступны и нигде не используются.
test_cluster_labels = kmeans.predict(X_test).astype(int)

# Проверяем контракт задания перед сохранением результата.
assert len(test_cluster_labels) == len(test_image_ids), (
    "Число кластеров не совпадает с числом тестовых изображений"
)
assert not pd.isnull(test_cluster_labels).any(), (
    "В предсказаниях присутствуют пропуски"
)

submission = pd.DataFrame({
    "image_id": test_image_ids,
    "prediction": test_cluster_labels,
})
submission.to_csv(f"{output_dir}/task3_2_predictions.csv", index=False)

print("\n=== Шаг 5. Submission сохранён ===")
print(f"Файл: outputs/task3_2_predictions.csv")
print(f"Количество строк: {len(submission)}")
print(submission.head())
print("Распределение кластеров:")
print(submission["prediction"].value_counts().sort_index())

# ============================================================
# ШАГ 7. Визуализация t-SNE
# ============================================================
# Сначала уменьшаем размерность PCA, потому что t-SNE плохо работает
# напрямую с 2352 пиксельными признаками.
# PCA используется только для визуализации, а не для обучения K-Means.
pca = PCA(n_components=30, random_state=SEED)
X_pca = pca.fit_transform(X)

tsne = TSNE(
    n_components=2,
    random_state=SEED,
    perplexity=30,
    learning_rate="auto",
    init="pca",
)
X_tsne = tsne.fit_transform(X_pca)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Слева показываем истинные классы.
scatter_true = axes[0].scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=labels, cmap="tab10", s=8, alpha=0.65,
)
axes[0].set_title("Истинные классы")
axes[0].set_xlabel("t-SNE компонент 1")
axes[0].set_ylabel("t-SNE компонент 2")
fig.colorbar(scatter_true, ax=axes[0], ticks=range(N_CLUSTERS))

# Справа показываем кластеры, найденные K-Means.
scatter_clusters = axes[1].scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=cluster_labels, cmap="tab10", s=8, alpha=0.65,
)
axes[1].set_title(f"Кластеры K-Means (ARI={ari:.3f}, NMI={nmi:.3f})")
axes[1].set_xlabel("t-SNE компонент 1")
axes[1].set_ylabel("t-SNE компонент 2")
fig.colorbar(scatter_clusters, ax=axes[1], ticks=range(N_CLUSTERS))

fig.tight_layout()
fig.savefig(figure_dir / "1_tsne_comparison.png", dpi=150)
plt.close(fig)

# ============================================================
# ШАГ 8. Визуализация центроидов кластеров
# ============================================================
# Центроид K-Means — средний вектор изображений внутри кластера.
# Возвращаем ему форму 28 x 28 x 3, чтобы показать как RGB-картинку.
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for cluster_id, ax in enumerate(axes.flat[:N_CLUSTERS]):
    centroid = kmeans.cluster_centers_[cluster_id].reshape(28, 28, 3)
    ax.imshow(np.clip(centroid, 0, 1))
    ax.set_title(f"Кластер {cluster_id}")
    ax.axis("off")
axes.flat[-1].axis("off")
fig.suptitle("Центроиды кластеров K-Means")
fig.tight_layout()
fig.savefig(figure_dir / "2_centroids.png", dpi=150)
plt.close(fig)

# ============================================================
# ШАГ 9. Матрица соответствия классов и кластеров
# ============================================================
# В обычной классификации строки и столбцы матрицы ошибок
# соответствуют классам. Здесь столбцы — это номера кластеров,
# поэтому номера классов и кластеров не обязаны совпадать.
cm = confusion_matrix(labels, cluster_labels, labels=range(N_CLUSTERS))
row_totals = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(
    cm,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0,
)

fig, ax = plt.subplots(figsize=(10, 8))
image = ax.imshow(cm_normalized, cmap="YlOrRd", vmin=0, vmax=1)
for i in range(N_CLUSTERS):
    for j in range(N_CLUSTERS):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

ax.set_xlabel("Номер кластера K-Means")
ax.set_ylabel("Истинный класс")
ax.set_title("Соответствие истинных классов и кластеров")
ax.set_xticks(range(N_CLUSTERS))
ax.set_yticks(range(N_CLUSTERS))
fig.colorbar(image, ax=ax, label="Доля внутри класса")
fig.tight_layout()
fig.savefig(figure_dir / "3_confusion_matrix.png", dpi=150)
plt.close(fig)

# ============================================================
# ШАГ 10. Примеры изображений из каждого кластера
# ============================================================
# Показываем до 5 случайных изображений каждого кластера.
# Истинный класс подписываем только для интерпретации результата.
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(N_CLUSTERS, 5, figsize=(12, 18))

for cluster_id in range(N_CLUSTERS):
    cluster_indices = np.flatnonzero(cluster_labels == cluster_id)
    chosen_indices = rng.choice(
        cluster_indices,
        size=min(5, len(cluster_indices)),
        replace=False,
    )

    for column, ax in enumerate(axes[cluster_id]):
        ax.axis("off")
        if column < len(chosen_indices):
            image_index = chosen_indices[column]
            ax.imshow(images[image_index])
            ax.set_title(f"класс {labels[image_index]}", fontsize=8)

    axes[cluster_id, 0].set_ylabel(
        f"кластер {cluster_id}",
        rotation=0,
        labelpad=28,
    )

fig.suptitle("Примеры изображений внутри кластеров")
fig.tight_layout()
fig.savefig(figure_dir / "4_cluster_examples.png", dpi=150)
plt.close(fig)

# ============================================================
# ШАГ 11. Состав кластеров по истинным классам
# ============================================================
# Эта визуализация также выполняется только после обучения.
# Она помогает понять, какие классы чаще всего попадают в каждый кластер.
class_cluster_counts = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=int)
for class_id in range(N_CLUSTERS):
    class_cluster_counts[class_id] = np.bincount(
        cluster_labels[labels == class_id],
        minlength=N_CLUSTERS,
    )

class_cluster_percentages = (
    class_cluster_counts
    / class_cluster_counts.sum(axis=1, keepdims=True)
    * 100
)

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(N_CLUSTERS)
for class_id in range(N_CLUSTERS):
    ax.bar(
        range(N_CLUSTERS),
        class_cluster_percentages[class_id],
        bottom=bottom,
        label=f"Класс {class_id}",
    )
    bottom += class_cluster_percentages[class_id]

ax.set_xlabel("Номер кластера")
ax.set_ylabel("Процент изображений")
ax.set_title("Состав кластеров по истинным классам")
ax.set_xticks(range(N_CLUSTERS))
ax.legend(ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(figure_dir / "6_cluster_composition.png", dpi=150)
plt.close(fig)

print("\n=== Шаг 6. Визуализации сохранены ===")
print(f"Папка: {figure_dir}")

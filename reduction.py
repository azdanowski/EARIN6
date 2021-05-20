from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def dim_red(features, labels):
    pca = PCA()
    pca.fit_transform(features)
    pca_variance = pca.explained_variance_

    plt.figure(figsize=(8, 6))
    plt.bar(range(100), pca_variance, alpha=0.5,
        label='individual variance')
    plt.legend()
    plt.ylabel('Variance ratio')
    plt.xlabel('Principal components')
    plt.show()

    pca2 = PCA(n_components=20)
    pca2.fit(features)
    x_3d = pca2.transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_3d[:, 0], x_3d[:, 1], c=labels)
    plt.show()

    return x_3d

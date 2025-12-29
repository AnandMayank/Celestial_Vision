import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class EmbeddingVisualizer:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def create_embeddings(self, n_samples=200, embedding_dim=512):
        np.random.seed(42)
        n_classes = 5
        embeddings, labels = [], []
        for cls in range(n_classes):
            emb = np.random.randn(n_samples//n_classes, embedding_dim)
            embeddings.append(emb + cls * 0.5)
            labels.extend([cls] * (n_samples//n_classes))
        return np.vstack(embeddings), np.array(labels)
    
    def plot_2d_scatter(self, embeddings, labels, title="2D PCA Scatter"):
        scaled = self.scaler.fit_transform(embeddings)
        pca = PCA(n_components=2).fit(scaled)
        reduced = pca.transform(scaled)
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(label='Class')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title(title)
        plt.tight_layout()
        return plt
    
    def plot_3d_scatter(self, embeddings, labels, title="3D PCA Scatter"):
        scaled = self.scaler.fit_transform(embeddings)
        pca = PCA(n_components=3).fit(scaled)
        reduced = pca.transform(scaled)
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=labels, cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
        ax.set_title(title)
        return fig
    
    def compare_dimensions(self, embeddings_list, labels):
        fig, axes = plt.subplots(len(embeddings_list), 1, figsize=(10, 5*len(embeddings_list)))
        if len(embeddings_list) == 1:
            axes = [axes]
        
        for idx, (emb, dim, model_name) in enumerate(embeddings_list):
            scaled = self.scaler.fit_transform(emb)
            pca = PCA(n_components=2).fit(scaled)
            reduced = pca.transform(scaled)
            axes[idx].scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            axes[idx].set_title(f'{model_name} (Dim: {dim})')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

if __name__ == '__main__':
    viz = EmbeddingVisualizer()
    emb_128, labels = viz.create_embeddings(n_samples=200, embedding_dim=128)
    emb_512, _ = viz.create_embeddings(n_samples=200, embedding_dim=512)
    
    viz.plot_2d_scatter(emb_512, labels, '2D PCA Scatter (512-dim)')
    viz.plot_3d_scatter(emb_512, labels, '3D PCA Scatter (512-dim)')
    
    embeddings_list = [(emb_128, 128, 'Model A'), (emb_512, 512, 'Model B')]
    viz.compare_dimensions(embeddings_list, labels)

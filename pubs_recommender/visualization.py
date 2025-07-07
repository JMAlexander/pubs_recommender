import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

class Visualizer:
    """
    Provides methods for generating plots and returning them as BytesIO buffers for embedding in reports.
    """
    @staticmethod
    def plot_heatmap(frequencies, window_labels, cluster_ids, figsize=(12, 6)):
        """
        Plot the cluster distribution heatmap and return a BytesIO buffer.
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            frequencies.T,
            cmap='viridis',
            xticklabels=window_labels,
            yticklabels=[f'Cluster {c}' for c in cluster_ids],
            cbar_kws={'label': 'Proportion of Cluster Papers in Time Window', 'shrink': 0.8},
            annot=False,
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
        ax.set_title('Cluster Distribution Over Time', fontsize=16, pad=20)
        ax.set_xlabel('Median Date of Papers in Window', fontsize=12, labelpad=10)
        ax.set_ylabel('Cluster', fontsize=12, labelpad=10)
        ax.set_yticklabels([f'Cluster {i}' for i in cluster_ids], fontsize=10)
        n_labels = len(window_labels)
        max_ticks = 15
        if n_labels > max_ticks:
            tick_indices = np.linspace(0, n_labels - 1, max_ticks, dtype=int)
        else:
            tick_indices = np.arange(n_labels)
        ax.set_xticks(tick_indices + 0.5)
        ax.set_xticklabels([window_labels[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf

    @staticmethod
    def plot_silhouette_elbow(cluster_numbers, scores, best_n_silhouette, num_clusters, silhouette_scores, inertia_clusters, inertia_values, best_n_elbow, inertias, figsize=(15, 6)):
        """
        Plot silhouette and elbow plots side by side and return a BytesIO buffer.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        # Silhouette
        ax1.plot(cluster_numbers, scores, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=best_n_silhouette, color='red', linestyle='--', linewidth=2, label=f'Best Silhouette: {best_n_silhouette} clusters')
        if num_clusters is not None and num_clusters != best_n_silhouette:
            ax1.axvline(x=num_clusters, color='blue', linestyle='--', linewidth=2, label=f'Manual: {num_clusters} clusters')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title(f'Cluster Quality: Silhouette Scores\nBest: {best_n_silhouette} clusters (score: {silhouette_scores[best_n_silhouette]:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Elbow
        ax2.plot(inertia_clusters, inertia_values, 'ro-', linewidth=2, markersize=8)
        ax2.axvline(x=best_n_elbow, color='red', linestyle='--', linewidth=2, label=f'Elbow Point: {best_n_elbow} clusters')
        if num_clusters is not None and num_clusters != best_n_elbow:
            ax2.axvline(x=num_clusters, color='blue', linestyle='--', linewidth=2, label=f'Manual: {num_clusters} clusters')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Within-Cluster Sum of Squares')
        ax2.set_title(f'Cluster Quality: Elbow Method\nOptimal: {best_n_elbow} clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf

    @staticmethod
    def plot_mean_wcss(cluster_ids, plot_vals, ylabel, title, log_scale=True, figsize=(10, 4)):
        """
        Plot the (transformed) mean WCSS for each cluster and return a BytesIO buffer.
        """
        plt.figure(figsize=figsize)
        plt.bar([f'Cluster {c}' for c in cluster_ids], plot_vals, color='teal')
        plt.ylabel(ylabel)
        plt.xlabel('Cluster')
        plt.title(title)
        if log_scale:
            plt.yscale('log')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf

def analyze_library_evolution(library_files, window_size=100, step_size=25, similarity_model=None):
    """
    Analyze the evolution of a library over time using multiple library files.
    
    Args:
        library_files: List of library file paths
        window_size: Number of papers in each analysis window
        step_size: Number of papers to step forward for each window
        similarity_model: Word embedding model for similarity calculations
        
    Returns:
        Dictionary containing evolution analysis results
    """
    from .clustering import RollingClusterAnalyzer
    from .utils import load_library
    
    print(f"Analyzing evolution across {len(library_files)} library files...")
    
    evolution_results = {}
    
    for i, library_file in enumerate(library_files):
        print(f"\nProcessing library {i+1}/{len(library_files)}: {library_file}")
        
        # Load library
        library = load_library(library_file)
        if library is None:
            print(f"Failed to load library: {library_file}")
            continue
        
        # Create rolling analyzer
        analyzer = RollingClusterAnalyzer(window_size=window_size, step_size=step_size)
        
        # Create windows
        analyzer.create_windows(library, verbose=False)
        
        if not analyzer.windows:
            print(f"No valid windows created for library: {library_file}")
            continue
        
        # Compute cluster frequencies
        freq_results = analyzer.compute_cluster_frequencies(library.papers, verbose=False)
        
        evolution_results[library_file] = {
            'frequencies': freq_results['frequencies'],
            'window_labels': freq_results['window_labels'],
            'cluster_ids': freq_results['cluster_ids'],
            'window_dates': analyzer.window_dates
        }
        
        print(f"  - {len(analyzer.windows)} windows created")
        print(f"  - {len(freq_results['cluster_ids'])} clusters found")
    
    return evolution_results 
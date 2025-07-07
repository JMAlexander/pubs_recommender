import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ClusterAnalyzer:
    """
    Provides methods for cluster trend classification and tightness (mean WCSS) calculation.
    """
    @staticmethod
    def classify_trend(window_freqs, recent_n=5):
        """
        Classify the trend of a cluster based on its frequency over time windows.
        Args:
            window_freqs: 1D numpy array of cluster frequencies over time
            recent_n: Number of recent windows to consider as 'recent'
        Returns:
            trend: One of 'Dormant', 'Trending', 'Declining', 'Stable'
        """
        recent = window_freqs[-recent_n:]
        # Use the most recent peak (max in last 2*recent_n windows)
        peak_window = window_freqs[-2*recent_n:] if len(window_freqs) > 2*recent_n else window_freqs
        recent_mean = np.mean(recent)
        peak = np.max(peak_window)
        if peak == 0:
            return "Dormant"
        if recent_mean < 0.1 * peak:
            return "Dormant"
        elif recent_mean > 0.5 * peak and recent_mean > 0.05:
            return "Trending"
        elif recent_mean < 0.5 * peak and peak > 0.05:
            return "Declining"
        else:
            return "Stable"

    @staticmethod
    def mean_wcss(tfidf_dense, cluster_labels, cluster_ids):
        """
        Compute mean WCSS (average squared distance to centroid) for each cluster.
        Args:
            tfidf_dense: 2D numpy array (n_samples, n_features)
            cluster_labels: 1D array of cluster assignments for each sample
            cluster_ids: list/array of unique cluster IDs
        Returns:
            mean_wcss_per_cluster: list of mean WCSS for each cluster
        """
        mean_wcss_per_cluster = []
        for cluster in cluster_ids:
            indices = np.where(cluster_labels == cluster)[0]
            if len(indices) == 0:
                mean_wcss_per_cluster.append(0)
                continue
            cluster_vecs = tfidf_dense[indices]
            centroid = cluster_vecs.mean(axis=0)
            sq_dists = np.sum((cluster_vecs - centroid) ** 2, axis=1)
            mean_wcss = np.mean(sq_dists)
            mean_wcss_per_cluster.append(mean_wcss)
        return mean_wcss_per_cluster

def cluster_library(bow_corpus, termsim_matrix, papers_library, method='ward', verbose=False, optimization_algorithm='silhouette', max_clusters=20, num_clusters=None):
    """
    Cluster the library and find optimal number of clusters.
    
    Args:
        bow_corpus: Bag-of-words corpus
        termsim_matrix: Term similarity matrix
        papers_library: Library object containing papers to cluster
        method: Clustering method ('ward' or 'complete')
        verbose: Whether to print detailed debugging information
        optimization_algorithm: Method for selecting optimal clusters ('silhouette' or 'elbow')
        max_clusters: Maximum number of clusters to try
        num_clusters: If set, use this number of clusters directly (skip optimization)
    
    Returns:
        optimal_n: Optimal number of clusters
        silhouette_scores: Silhouette scores for different cluster numbers
        inertias: Inertia values for different cluster numbers
        best_n_silhouette: Best number of clusters according to silhouette scores
        best_n_elbow: Best number of clusters according to elbow method
    """
    # Get similarity matrix
    similarity_matrix = get_cosine_matrix(termsim_matrix, bow_corpus, bow_corpus)
    if verbose:
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    if num_clusters is not None:
        # Use the specified number of clusters directly
        optimal_n = num_clusters
        silhouette_scores = {}
        inertias = {}
        best_n_silhouette = None
        best_n_elbow = None
    else:
        # Find optimal clusters
        optimal_n, silhouette_scores, inertias, best_n_silhouette, best_n_elbow = find_optimal_clusters(
            similarity_matrix, max_clusters=max_clusters, method=optimization_algorithm, linkage_method=method)
    
    # Compute topics and assign to papers
    compute_topics(similarity_matrix, papers_library.papers, method=method, num_topics=optimal_n, verbose=verbose)
    
    return optimal_n, silhouette_scores, inertias, best_n_silhouette, best_n_elbow

def get_cosine_matrix(termsim_matrix, bow_i, bow_j, normalized_value=(True, True)):
    """
    Compute cosine similarity matrix using dense matrices.
    
    Args:
        termsim_matrix: Term similarity matrix
        bow_i: First bag-of-words corpus
        bow_j: Second bag-of-words corpus
        normalized_value: Tuple of booleans for normalization
    
    Returns:
        Dense cosine similarity matrix
    """
    # Initialize dense matrix
    matrix_size = (len(bow_i), len(bow_j))
    cosine_similarities = np.zeros(matrix_size)
    
    # Compute cosine similarities
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            cosine_similarities[i,j] = termsim_matrix.inner_product(bow_i[i], bow_j[j], normalized=normalized_value)
    
    return cosine_similarities

def compute_topics(cosine_matrix, papers, method='average', num_topics=10, verbose=False):
    """
    Compute topics using hierarchical clustering and assign cluster IDs to papers.
    
    Args:
        cosine_matrix: Similarity matrix (dense)
        papers: List of Paper objects to assign clusters to
        method: Linkage method for hierarchical clustering
        num_topics: Number of topics to create
        verbose: Whether to print detailed debugging information
        
    Returns:
        None (updates papers directly)
    """
    # Convert to distance matrix
    distance_matrix = 1 - cosine_matrix
    
    # Ensure diagonal is zero
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed form (upper triangular part as a vector)
    condensed_distance = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method=method, optimal_ordering=False)
    topic_labels = fcluster(linkage_matrix, num_topics, criterion='maxclust')
    
    # Assign cluster IDs to papers
    for paper, cluster_id in zip(papers, topic_labels):
        paper.cluster_id = cluster_id
    
    # Calculate individual silhouette scores using scikit-learn
    silhouette_scores = silhouette_samples(distance_matrix, topic_labels, metric='precomputed')
    
    # Assign silhouette scores to papers
    for paper, score in zip(papers, silhouette_scores):
        paper.silhouette_score = score
    
    return None

def get_topic_similarity(cosine_matrix, cluster_labels):
    """
    Calculate similarity between topics and documents.
    
    Args:
        cosine_matrix: Similarity matrix between documents
        cluster_labels: Cluster assignments for documents
        
    Returns:
        Topic similarity matrix
    """
    topic_ids = np.unique(cluster_labels)
    matrix_size = (len(topic_ids), cosine_matrix.shape[1])
    topic_sim_matrix = np.zeros(matrix_size)

    for i, topic in enumerate(topic_ids):
        for j in range(cosine_matrix.shape[1]):
            topic_indices = np.where(cluster_labels == topic)[0]
            topic_sim_matrix[i, j] = np.mean(cosine_matrix[topic_indices, j])

    return topic_sim_matrix

def detect_elbow_point(inertias):
    """
    Detect the elbow point in inertia values using the knee/elbow detection algorithm.
    
    Args:
        inertias: Dictionary of {n_clusters: inertia_value}
        
    Returns:
        optimal_n: Number of clusters at the elbow point
    """
    cluster_numbers = sorted(inertias.keys())
    inertia_values = [inertias[n] for n in cluster_numbers]
    
    # Calculate the rate of change (first derivative)
    first_derivative = np.diff(inertia_values)
    
    # Calculate the rate of change of the rate of change (second derivative)
    second_derivative = np.diff(first_derivative)
    
    # Find the point of maximum curvature (minimum second derivative)
    # Add 2 to account for the two diff operations and 1-indexing
    elbow_idx = np.argmin(second_derivative) + 2
    
    # Ensure we don't go out of bounds
    elbow_idx = min(elbow_idx, len(cluster_numbers) - 1)
    
    optimal_n = cluster_numbers[elbow_idx]
    
    return optimal_n

def find_optimal_clusters(similarity_matrix, min_clusters=2, max_clusters=20, method='silhouette', linkage_method='average'):
    """
    Find the optimal number of clusters using silhouette scores and inertia.
    
    Args:
        similarity_matrix: Similarity matrix (dense)
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        method: Method for selecting optimal clusters ('silhouette' or 'elbow')
        linkage_method: Linkage method for hierarchical clustering
        
    Returns:
        Tuple of (optimal number of clusters, dictionary of silhouette scores, dictionary of inertia values, best_n_silhouette, best_n_elbow)
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Ensure diagonal is zero
    np.fill_diagonal(distance_matrix, 0)

    # Convert to condensed form (upper triangular part as a vector)
    condensed_distance = squareform(distance_matrix, checks=False)
    
    # Try different numbers of clusters
    silhouette_scores = {}
    inertias = {}
    best_score = -1
    best_n_silhouette = min_clusters
    
    for n in range(min_clusters, max_clusters + 1):
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distance, method=linkage_method, optimal_ordering=True)
        clusters = fcluster(linkage_matrix, n, criterion='maxclust')
        
        # Compute silhouette score
        score = silhouette_score(distance_matrix, clusters, metric='precomputed')
        silhouette_scores[n] = score
        
        # Compute inertia (within-cluster sum of squares)
        inertia = 0
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 1:
                cluster_distances = distance_matrix[cluster_indices][:, cluster_indices]
                inertia += np.sum(cluster_distances) / 2  # Divide by 2 since matrix is symmetric
        inertias[n] = inertia
        
        # Update best silhouette score
        if score > best_score:
            best_score = score
            best_n_silhouette = n
    
    # Calculate best_n_elbow
    best_n_elbow = detect_elbow_point(inertias)
    
    # Choose optimal number of clusters based on method
    if method == 'silhouette':
        optimal_n = best_n_silhouette
        print(f"\nOptimal number of clusters (silhouette): {optimal_n} (score: {best_score:.3f})")
    elif method == 'elbow':
        optimal_n = best_n_elbow
        print(f"\nOptimal number of clusters (elbow): {optimal_n}")
        print(f"Silhouette score at elbow: {silhouette_scores[optimal_n]:.3f}")
    else:
        raise ValueError("Method must be 'silhouette' or 'elbow'")
    
    return optimal_n, silhouette_scores, inertias, best_n_silhouette, best_n_elbow

def compare_clusterings(library1, clusters1, library2, clusters2, similarity_threshold=0.5):
    """
    Compare two clusterings of similar libraries to assess stability and similarity.
    
    Args:
        library1: First library object
        clusters1: Cluster assignments for first library
        library2: Second library object
        clusters2: Cluster assignments for second library
        similarity_threshold: Minimum Jaccard similarity to consider clusters related
        
    Returns:
        dict: Dictionary containing:
            - 'cluster_matches': List of matching cluster pairs with similarity scores
            - 'stability_score': Average similarity of matched clusters
            - 'unmatched_clusters': Count of clusters without matches
    """
    # Get unique cluster IDs
    unique_clusters1 = np.unique(clusters1)
    unique_clusters2 = np.unique(clusters2)
    
    # Create dictionaries mapping cluster IDs to paper PMIDs
    cluster_papers1 = {}
    cluster_papers2 = {}
    
    for cluster_id in unique_clusters1:
        cluster_papers1[cluster_id] = set(p.pmid for p, c in zip(library1.papers, clusters1) if c == cluster_id)
    
    for cluster_id in unique_clusters2:
        cluster_papers2[cluster_id] = set(p.pmid for p, c in zip(library2.papers, clusters2) if c == cluster_id)
    
    # Find matching clusters
    cluster_matches = []
    matched_clusters2 = set()
    
    for cluster1 in unique_clusters1:
        best_match = None
        best_similarity = 0
        
        for cluster2 in unique_clusters2:
            if cluster2 in matched_clusters2:
                continue
                
            papers1 = cluster_papers1[cluster1]
            papers2 = cluster_papers2[cluster2]
            
            if papers1 and papers2:
                intersection = len(papers1 & papers2)
                union = len(papers1 | papers2)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cluster2
        
        if best_match and best_similarity >= similarity_threshold:
            cluster_matches.append({
                'cluster1': cluster1,
                'cluster2': best_match,
                'similarity': best_similarity,
                'size1': len(cluster_papers1[cluster1]),
                'size2': len(cluster_papers2[best_match]),
                'overlap': len(cluster_papers1[cluster1] & cluster_papers2[best_match])
            })
            matched_clusters2.add(best_match)
    
    # Calculate stability score
    stability_score = np.mean([m['similarity'] for m in cluster_matches]) if cluster_matches else 0
    
    # Count unmatched clusters
    unmatched_clusters = len(unique_clusters2) - len(matched_clusters2)
    
    # Print results
    print("\nCluster Comparison Results:")
    print(f"Total clusters in first run: {len(unique_clusters1)}")
    print(f"Total clusters in second run: {len(unique_clusters2)}")
    print(f"Matched clusters: {len(cluster_matches)}")
    print(f"Unmatched clusters: {unmatched_clusters}")
    print(f"Average cluster similarity: {stability_score:.3f}")
    
    print("\nCluster Matches:")
    for match in sorted(cluster_matches, key=lambda x: x['similarity'], reverse=True):
        print(f"Cluster {match['cluster1']} → Cluster {match['cluster2']}:")
        print(f"  Similarity: {match['similarity']:.3f}")
        print(f"  Sizes: {match['size1']} → {match['size2']} papers")
        print(f"  Overlap: {match['overlap']} papers")
    
    return {
        'cluster_matches': cluster_matches,
        'stability_score': stability_score,
        'unmatched_clusters': unmatched_clusters
    }

class ClusterTracker:
    """
    Track cluster compositions over time to assess stability.
    """
    def __init__(self):
        self.cluster_history = {}  # {cluster_id: [(timestamp, papers_set), ...]}
        self.stability_scores = {}  # {cluster_id: [stability_score, ...]}
    
    def update(self, timestamp, papers, clusters, similarity_matrix):
        """
        Update cluster compositions with current state.
        
        Args:
            timestamp: Current timestamp
            papers: List of Paper objects
            clusters: Cluster assignments
            similarity_matrix: Similarity matrix for stability calculation
        """
        # Group papers by cluster
        cluster_papers = {}
        for paper, cluster_id in zip(papers, clusters):
            if cluster_id not in cluster_papers:
                cluster_papers[cluster_id] = set()
            cluster_papers[cluster_id].add(paper.pmid)
        
        # Update history for each cluster
        for cluster_id, current_papers in cluster_papers.items():
            if cluster_id not in self.cluster_history:
                self.cluster_history[cluster_id] = []
            
            self.cluster_history[cluster_id].append((timestamp, current_papers))
            
            # Calculate stability if we have previous state
            if len(self.cluster_history[cluster_id]) > 1:
                stability = self._calculate_stability(cluster_id, current_papers)
                if cluster_id not in self.stability_scores:
                    self.stability_scores[cluster_id] = []
                self.stability_scores[cluster_id].append(stability)
    
    def _calculate_stability(self, cluster_id, current_papers):
        """
        Calculate stability score for a cluster.
        
        Args:
            cluster_id: Cluster ID
            current_papers: Set of current paper PMIDs
            
        Returns:
            Stability score (Jaccard similarity with previous state)
        """
        if len(self.cluster_history[cluster_id]) < 2:
            return 1.0
        
        # Get previous state
        _, previous_papers = self.cluster_history[cluster_id][-2]
        
        if not previous_papers or not current_papers:
            return 0.0
        
        intersection = len(previous_papers & current_papers)
        union = len(previous_papers | current_papers)
        
        return intersection / union if union > 0 else 0.0

class RollingClusterAnalyzer:
    """
    Analyze clusters over rolling time windows.
    """
    def __init__(self, window_size=50, step_size=50):
        self.window_size = window_size
        self.step_size = step_size
        self.windows = []
        self.window_dates = []
    
    def create_sorted_library(self, papers_library, verbose=False):
        """
        Create a sorted version of the library by creation date.
        
        Args:
            papers_library: Library object
            verbose: Whether to print debugging information
            
        Returns:
            List of papers sorted by creation date
        """
        # Filter out papers without creation dates
        valid_papers = [p for p in papers_library.papers if p.date_created is not None]
        
        if verbose:
            print(f"Total papers: {len(papers_library.papers)}")
            print(f"Papers with creation dates: {len(valid_papers)}")
        
        # Sort by creation date
        sorted_papers = sorted(valid_papers, key=lambda p: pd.to_datetime(p.date_created))
        
        if verbose:
            print(f"Date range: {sorted_papers[0].date_created} to {sorted_papers[-1].date_created}")
        
        return sorted_papers
    
    def create_windows(self, papers_library, verbose=False):
        """
        Create rolling windows for analysis.
        
        Args:
            papers_library: Library object
            verbose: Whether to print debugging information
        """
        sorted_papers = self.create_sorted_library(papers_library, verbose)
        
        if verbose:
            print("\nVerifying paper sorting and window creation:")
            print(f"Total number of papers: {len(sorted_papers)}")
            print("\nFirst 5 papers by creation date:")
            for i, paper in enumerate(sorted_papers[:5]):
                print(f"{i+1}. Created: {paper.date_created}, Title: {paper.title[:50]}...")
            
            print("\nLast 5 papers by creation date:")
            for i, paper in enumerate(sorted_papers[-5:]):
                print(f"{i+1}. Created: {paper.date_created}, Title: {paper.title[:50]}...")
        
        # Create windows based on sorted papers
        self.windows = []
        self.window_dates = []
        
        for i in range(0, len(sorted_papers), self.step_size):
            end_idx = min(i + self.window_size, len(sorted_papers))
            if end_idx - i >= self.window_size // 2:  # Only include windows with at least half the size
                self.windows.append((i, end_idx))
                window_date = pd.to_datetime(sorted_papers[i].date_created)
                self.window_dates.append(window_date)
        
        if verbose:
            print(f"\nCreated {len(self.windows)} windows")
            print("\nWindow details:")
            for i, (start_idx, end_idx) in enumerate(self.windows):
                print(f"\nWindow {i+1}:")
                print(f"  Papers {start_idx} to {end_idx-1} (size: {end_idx-start_idx})")
                print(f"  Start date: {sorted_papers[start_idx].date_created}")
                print(f"  End date: {sorted_papers[end_idx-1].date_created}")
    
    def compute_cluster_frequencies(self, papers, verbose=False):
        """
        Compute cluster frequencies across time windows.
        
        Args:
            papers: List of Paper objects
            verbose: Whether to print debugging information
            
        Returns:
            Dictionary with frequencies, window labels, and cluster IDs
        """
        if not self.windows:
            raise ValueError("Windows must be created before computing frequencies")
        
        # Get unique cluster IDs
        cluster_ids = np.unique([p.cluster_id for p in papers if p.cluster_id is not None])
        
        if verbose:
            print(f"Found {len(cluster_ids)} unique clusters: {cluster_ids}")
        
        # Initialize frequency matrix
        frequencies = np.zeros((len(self.windows), len(cluster_ids)))
        
        # Sort papers by creation date
        sorted_papers = sorted(papers, key=lambda p: pd.to_datetime(p.date_created))
        
        # First pass: count papers in each cluster for each window
        for window_idx, (start_idx, end_idx) in enumerate(self.windows):
            window_papers = sorted_papers[start_idx:end_idx]
            
            # Count papers in each cluster
            for paper in window_papers:
                if paper.cluster_id is not None:
                    cluster_idx = np.where(cluster_ids == paper.cluster_id)[0][0]
                    frequencies[window_idx, cluster_idx] += 1
        
        # Second pass: normalize by total papers in each cluster across all windows
        for cluster_idx, cluster_id in enumerate(cluster_ids):
            total_papers_in_cluster = np.sum(frequencies[:, cluster_idx])
            if total_papers_in_cluster > 0:
                frequencies[:, cluster_idx] /= total_papers_in_cluster
        
        # Create window labels
        window_labels = [f"Window {i+1}" for i in range(len(self.windows))]
        
        return {
            'frequencies': frequencies,
            'window_labels': window_labels,
            'cluster_ids': cluster_ids
        }
    
    def visualize_cluster_frequencies(self, papers, figsize=(12, 6), verbose=False):
        """
        Visualize cluster frequencies as a heatmap.
        
        Args:
            papers: List of Paper objects
            figsize: Figure size
            verbose: Whether to print debugging information
        """
        freq_results = self.compute_cluster_frequencies(papers, verbose)
        frequencies = freq_results['frequencies']
        window_labels = freq_results['window_labels']
        cluster_ids = freq_results['cluster_ids']
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(frequencies.T, 
                   xticklabels=window_labels,
                   yticklabels=[f'Cluster {c}' for c in cluster_ids],
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Cluster Frequency'})
        
        plt.title('Cluster Frequencies Over Time Windows')
        plt.xlabel('Time Window')
        plt.ylabel('Cluster')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Limit x-axis tick labels to 15
        if len(window_labels) > 15:
            plt.xticks(range(len(window_labels)), window_labels, rotation=45, ha='right')
            plt.gca().set_xticks(range(0, len(window_labels), max(1, len(window_labels)//15)))
            plt.gca().set_xticklabels([window_labels[i] for i in range(0, len(window_labels), max(1, len(window_labels)//15))], rotation=45, ha='right')
        
        plt.tight_layout()
    
    def debug_window_creation(self, papers_library, verbose=False):
        """
        Debug window creation process.
        
        Args:
            papers_library: Library object
            verbose: Whether to print debugging information
        """
        if not verbose:
            return
        
        print("\n=== Debugging Window Creation ===")
        print(f"Window size: {self.window_size}")
        print(f"Step size: {self.step_size}")
        
        # Create sorted library
        sorted_papers = self.create_sorted_library(papers_library, verbose=False)
        
        print(f"\nTotal papers after sorting: {len(sorted_papers)}")
        
        # Show date range
        if sorted_papers:
            start_date = pd.to_datetime(sorted_papers[0].date_created)
            end_date = pd.to_datetime(sorted_papers[-1].date_created)
            print(f"Date range: {start_date} to {end_date}")
            print(f"Total days: {(end_date - start_date).days}")
        
        # Show window creation
        print(f"\nWindow creation:")
        for i in range(0, len(sorted_papers), self.step_size):
            end_idx = min(i + self.window_size, len(sorted_papers))
            window_size = end_idx - i
            if window_size >= self.window_size // 2:
                start_date = pd.to_datetime(sorted_papers[i].date_created)
                end_date = pd.to_datetime(sorted_papers[end_idx-1].date_created)
                print(f"  Window {len(self.windows)+1}: Papers {i}-{end_idx-1} ({window_size} papers), {start_date} to {end_date}")
        
        print("=== End Debug Output ===\n") 
"""
Analysis package for literature analysis.

This package contains modules for:
- clustering: Cluster analysis and optimization
- visualization: Plotting and visualization
- reporting: Report generation
- models: Data models and model management
- utils: Utility functions
- feeds: Feed processing and publication matching
- email: Email functionality
- language: Natural language processing
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .clustering import ClusterAnalyzer, cluster_library, find_optimal_clusters, get_cosine_matrix, compute_topics, get_topic_similarity
from .visualization import Visualizer, analyze_library_evolution
from .reporting import ReportGenerator  # Requires reportlab
from .models import Paper, Library, load_word_model, load_or_create_models_for_clustering, save_models, save_papers_library
from .utils import clean_text, parse_date, safe_int, safe_float, filter_library, tokenize_library, get_recent_library, load_library
from .feeds import tokenize_feeds, tokenize_pubmed, check_feeds, get_topic_matching_pubs, load_sent_papers, save_sent_papers, get_paper_id, match_topics_to_publications
from .email import send_email_with_SMTP, send_email_with_Web_API, draft_plaintext_email, draft_html_email
from .language import extract_tfidf_keywords, extract_lda_keywords, extract_cluster_docs

__all__ = [
    # Clustering
    'ClusterAnalyzer', 'cluster_library', 'find_optimal_clusters', 'get_cosine_matrix', 'compute_topics', 'get_topic_similarity',
    # Visualization
    'Visualizer', 'analyze_library_evolution',
    # Reporting
    'ReportGenerator',  # Requires reportlab
    # Models
    'Paper', 'Library', 'load_word_model', 'load_or_create_models_for_clustering', 'save_models', 'save_papers_library',
    # Utils
    'clean_text', 'parse_date', 'safe_int', 'safe_float', 'filter_library', 'tokenize_library', 'get_recent_library', 'load_library',
    # Feeds
    'tokenize_feeds', 'tokenize_pubmed', 'check_feeds', 'get_topic_matching_pubs', 'load_sent_papers', 'save_sent_papers', 'get_paper_id', 'match_topics_to_publications',
    # Email
    'send_email_with_SMTP', 'send_email_with_Web_API', 'draft_plaintext_email', 'draft_html_email',
    # Language
    'extract_tfidf_keywords', 'extract_lda_keywords', 'extract_cluster_docs'
] 
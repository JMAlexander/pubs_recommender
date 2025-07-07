#!/usr/lib//venv/bin/python3

from gensim import models
from gensim import corpora
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import nltk
from nltk.corpus import stopwords
import pickle
import numpy as np
import pandas as pd
import os
import argparse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.platypus import Image
from datetime import datetime
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import base64
import analysis.clustering as clustering
import analysis.visualization as visualization
import analysis.reporting as reporting
import analysis.models as models
import analysis.utils as utils

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze library clusters and generate reports.')
    parser.add_argument('--library', type=str, help='Path to specific library file (optional)')
    parser.add_argument('--clustering_method', type=str, default='ward', help='Clustering method (ward or complete)')
    parser.add_argument('--max-clusters', type=int, default=20,
                       help='Maximum number of clusters to try during optimization analysis(default: 20)')
    parser.add_argument('--window_size', type=int, default=50, help='Number of papers in each analysis window')
    parser.add_argument('--step_size', type=int, default=50, help='Number of papers to step forward for each window')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing models')
    parser.add_argument('--num-clusters', type=str, default='auto',
                       help="Number of clusters to use. If 'auto', will optimize using --optimization-algorithm. Otherwise, provide an integer.")
    parser.add_argument('--optimization-algorithm', choices=['silhouette', 'elbow'], default='silhouette',
                       help="Algorithm to use for cluster number optimization if --num-clusters is 'auto'.")
    parser.add_argument('--report-mode', type=str, choices=['pdf', 'html'], default='pdf',
                       help="Report output format: 'pdf' (default) or 'html'.")
    args = parser.parse_args()

    # Set default paths for library file and model path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    # Load library
    if args.library:
        # Use user-provided library path
        papers_library = utils.load_library(args.library)
    else:
        # Use default data directory
        library_path = utils.get_recent_library(data_dir)
        if library_path is None:
            print(f"Error: Could not find library in {data_dir}")
            return
        papers_library = utils.load_library(library_path)
    
    if papers_library is None:
        print(f"Error: Could not load library")
        return
    
    # Load word model
    pubmed_wordmodel = models.load_word_model(model_path)
    if pubmed_wordmodel is None:
        print(f"Error: Could not load word model")
        return
    
    # Load or create models
    dictionary, bow_corpus, model_tfidf, termsim_matrix = models.load_or_create_models_for_clustering(papers_library, pubmed_wordmodel, model_path, args.overwrite)
    
    # Filter library and tokenize
    filtered_library = utils.filter_library(papers_library, field='title')
    tokenized_library = utils.tokenize_library(filtered_library, wordmodel=pubmed_wordmodel, field='title')
    bow_corpus = [dictionary.doc2bow(text) for text in tokenized_library]
    
    # Compute similarity matrix once
    similarity_matrix = clustering.get_cosine_matrix(termsim_matrix, bow_corpus, bow_corpus)

    # Always run optimization to get curves for both methods
    silhouette_opt_n, silhouette_scores, inertias, best_n_silhouette, best_n_elbow = clustering.find_optimal_clusters(
        similarity_matrix,
        max_clusters=args.max_clusters, method='silhouette', linkage_method=args.clustering_method)

    # Determine number of clusters for actual clustering
    if args.num_clusters != 'auto':
        try:
            num_clusters = int(args.num_clusters)
        except ValueError:
            print("Error: --num-clusters must be an integer or 'auto'.")
            return
        optimal_n = num_clusters
    else:
        num_clusters = None
        optimal_n = best_n_silhouette if args.optimization_algorithm == 'silhouette' else best_n_elbow

    # Cluster library using the chosen number of clusters
    clustering.compute_topics(similarity_matrix, filtered_library.papers, method=args.clustering_method, num_topics=optimal_n, verbose=args.verbose)

    # Print clustering results
    print(f"Number of clusters used: {optimal_n}")
 

    # Save models (only if overwrite is True)
    if args.overwrite:
        print("Saving models...")
        models.save_models(dictionary, model_tfidf, termsim_matrix, model_path)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped folder inside output directory
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Save run parameters to a text file
    run_params_file = os.path.join(run_dir, 'run_parameters.txt')
    with open(run_params_file, 'w') as f:
        f.write('Run parameters for this analysis:\n')
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')

    # Generate report
    print("Generating analysis report...")
    # Use the new ReportGenerator from analysis.reporting
    report_generator = reporting.ReportGenerator(
        filtered_library,
        bow_corpus,
        dictionary,
        run_dir,
        window_size=args.window_size,
        step_size=args.step_size,
        verbose=args.verbose,
        silhouette_scores=silhouette_scores,
        inertias=inertias,
        best_n_silhouette=best_n_silhouette,
        best_n_elbow=best_n_elbow,
        num_clusters=num_clusters,
        model_tfidf=model_tfidf,
        report_mode=args.report_mode
    )
    report_generator.generate_report()

    # Save papers library to output directory
    print("Saving papers library...")
    models.save_papers_library(filtered_library, run_dir)

    print("Analysis complete!")

if __name__ == "__main__":
    main() 
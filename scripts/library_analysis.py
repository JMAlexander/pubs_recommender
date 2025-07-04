#!/usr/lib//venv/bin/python3

from gensim import models
from gensim import corpora
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import nltk
from nltk.corpus import stopwords
import pickle
import numpy as np
import pandas as pd
import source as lang_helper
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

def generate_cluster_report(papers_library, bow_corpus, dictionary, run_dir, window_size=50, step_size=50, verbose=False, silhouette_scores=None, inertias=None, best_n_silhouette=None, best_n_elbow=None, num_clusters=None, model_tfidf=None, report_mode='pdf'):
    """
    Generate a report analyzing the library clusters and a simple text file for feed search preferences.
    
    Args:
        papers_library: List of papers to analyze
        bow_corpus: Bag-of-words corpus
        dictionary: Gensim dictionary
        output_dir: Directory to save the report
        window_size: Number of papers in each analysis window
        step_size: Number of papers to step forward for each window
        verbose: Whether to print detailed debugging information
        silhouette_scores: Dictionary of silhouette scores for different cluster numbers
        inertias: Dictionary of inertia values for different cluster numbers
        best_n_silhouette: Best silhouette score and corresponding cluster number
        best_n_elbow: Elbow method suggests this number of clusters
        num_clusters: Optional number of clusters to add to plots
        model_tfidf: Trained Gensim TfidfModel for feature extraction
        report_mode: 'pdf' (default) or 'html' for output format
    """
    
    # Initialize ClusterTracker and RollingClusterAnalyzer
    tracker = lang_helper.ClusterTracker()
    analyzer = lang_helper.RollingClusterAnalyzer(window_size=window_size, step_size=step_size)
    
    # Debug window creation
    if verbose:
        print("\n=== Debugging Window Creation ===")
        analyzer.debug_window_creation(papers_library, verbose=verbose)
        print("=== End Debug Output ===\n")
    
    # Create windows for analysis
    analyzer.create_windows(papers_library, verbose=verbose)
    
    # Compute cluster frequencies
    freq_results = analyzer.compute_cluster_frequencies(papers_library.papers, verbose=verbose)
    frequencies = freq_results['frequencies']
    window_labels = freq_results['window_labels']
    cluster_ids = freq_results['cluster_ids']
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    
    # Create PDF with landscape orientation
    doc = SimpleDocTemplate(
        os.path.join(run_dir, f"library_analysis_report_{timestamp}.pdf"),
        pagesize=letter,
        rightMargin=36,  # Reduced margins for landscape
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Modern styles
    modern_title_style = ParagraphStyle(
        'ModernTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor("#22223b"),
        spaceAfter=18,
        keepWithNext=True,
    )
    modern_section_style = ParagraphStyle(
        'ModernSection',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.HexColor("#4a4e69"),
        backColor=colors.HexColor("#f2e9e4"),
        leftIndent=0,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True,
    )
    modern_normal = ParagraphStyle(
        'ModernNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=colors.HexColor("#22223b"),
        spaceAfter=6,
    )

    # Title
    story.append(Paragraph("Library Analysis Report", modern_title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8))
    
    # Create a two-column layout for the overview section
    overview_data = []
    
    # Cluster Overview
    overview_data.append([Paragraph("Cluster Overview", modern_section_style), ""])
    overview_data.append([Spacer(1, 12), ""])
    
    # Get cluster sizes from papers
    cluster_sizes = pd.Series([p.cluster_id for p in papers_library.papers if p.cluster_id is not None]).value_counts().sort_index()
    
    # Create combined table data
    table_data = [['Cluster', 'Number of Papers', 'Keywords', 'Trend']]
    
    # Create list to store cluster lines for text file
    cluster_lines = []
    
    for cluster in cluster_ids:  # Use cluster_ids from freq_results
        # Get documents in this cluster
        cluster_indices = [i for i, p in enumerate(papers_library.papers) if p.cluster_id == cluster]
        cluster_corpus = [bow_corpus[i] for i in cluster_indices]
        
        # Get keywords using LDA
        keywords = lang_helper.extract_lda_keywords(dictionary, cluster_corpus)
        
        # Format keywords (without scores)
        keywords_str = ", ".join([k for k, _ in keywords])
        
        # Calculate trend based on frequency changes (improved logic)
        cluster_idx = np.where(cluster_ids == cluster)[0][0]
        window_freqs = frequencies[:, cluster_idx]
        recent_n = 5
        past_n = 15
        recent = window_freqs[-recent_n:]
        if len(window_freqs) > past_n:
            past = window_freqs[-past_n:-recent_n]
        elif len(window_freqs) > recent_n:
            past = window_freqs[:-recent_n]
        else:
            past = window_freqs
        recent_mean = np.mean(recent)
        past_mean = np.mean(past) if len(past) > 0 else 0
        peak = np.max(window_freqs)
        if recent_mean < 0.04 * peak:
            trend = "Dormant"
        elif recent_mean > 1.5 * past_mean and recent_mean > 0.04:
            trend = "Trending"
        elif recent_mean < 0.5 * past_mean and past_mean > 0.04:
            trend = "Declining"
        else:
            trend = "Stable"
        
        # Add row to table
        table_data.append([
            f'Cluster {cluster}',
            str(cluster_sizes[cluster]),
            keywords_str,
            trend
        ])
        
        # Add to cluster lines for text file
        cluster_lines.append(f"Cluster {cluster}: {cluster_sizes[cluster]} papers - {keywords_str}")
    
    # Create table with adjusted column widths for landscape
    # Adjusted widths to prevent header overlap
    t = Table(table_data, colWidths=[1.2*inch, 1.5*inch, 3.5*inch, 1.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a4e69")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f2e9e4")),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor("#22223b")),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#c9ada7")),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    # Cluster Overview explainer
    cluster_overview_explainer = Paragraph(
        "<b>What is this?</b> This table summarizes each cluster, showing its size, top keywords, and trend over time. "
        "<b>Why look at it?</b> It gives a quick overview of the main topics in your library and how they are changing.",
        modern_normal)
    story.append(KeepTogether([
        Paragraph("Cluster Overview", modern_section_style),
        cluster_overview_explainer,
        Spacer(1, 6),
        t,
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8)
    ]))
    
    # Topic Identification Analysis plot (silhouette and elbow)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Left subplot: Silhouette scores
    cluster_numbers = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    ax1.plot(cluster_numbers, scores, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=best_n_silhouette, color='red', linestyle='--', linewidth=2, label=f'Best Silhouette: {best_n_silhouette} clusters')
    if num_clusters is not None and num_clusters != best_n_silhouette:
        ax1.axvline(x=num_clusters, color='blue', linestyle='--', linewidth=2, label=f'Manual: {num_clusters} clusters')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title(f'Cluster Quality: Silhouette Scores\nBest: {best_n_silhouette} clusters (score: {silhouette_scores[best_n_silhouette]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Right subplot: Elbow plot (inertia/within-cluster sum of squares)
    inertia_clusters = list(inertias.keys())
    inertia_values = list(inertias.values())
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
    silhouette_buffer = BytesIO()
    plt.savefig(silhouette_buffer, format='png', dpi=300, bbox_inches='tight')
    silhouette_buffer.seek(0)
    plt.close()
    # Topic Identification Analysis explainer
    silhouette_explainer = Paragraph(
        "<b>What is this?</b> These plots show how well your data clusters for different numbers of clusters. "
        "The left plot shows the silhouette score (higher is better), and the right shows the elbow method (lower is better). "
        "<b>Why look at it?</b> It helps you choose the best number of clusters for your data.",
        modern_normal)
    story.append(KeepTogether([
        Paragraph("Topic Identification Analysis", modern_section_style),
        silhouette_explainer,
        Spacer(1, 6),
        Image(silhouette_buffer, width=500, height=180),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8)
    ]))

    # Insert WCSS block here (if model_tfidf is not None)
    if model_tfidf is not None:
        cluster_ids = np.unique([p.cluster_id for p in papers_library.papers if p.cluster_id is not None])
        tfidf_vectors = [model_tfidf[bow] for bow in bow_corpus]
        vocab_size = len(dictionary)
        tfidf_dense = np.zeros((len(tfidf_vectors), vocab_size))
        for i, vec in enumerate(tfidf_vectors):
            for idx, val in vec:
                tfidf_dense[i, idx] = val
        mean_wcss_per_cluster = []
        cluster_labels = np.array([p.cluster_id for p in papers_library.papers])
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
        # Transform for plotting
        plot_vals = 1 - np.array(mean_wcss_per_cluster)
        plt.figure(figsize=(10, 4))
        plt.bar([f'Cluster {c}' for c in cluster_ids], plot_vals, color='teal')
        plt.ylabel('1 - Mean Squared Distance to Centroid (log scale)')
        plt.xlabel('Cluster')
        plt.title('Transformed Mean WCSS for Each Cluster')
        plt.yscale('log')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        wcss_buffer = BytesIO()
        plt.savefig(wcss_buffer, format='png', dpi=300, bbox_inches='tight')
        wcss_buffer.seek(0)
        plt.close()
        wcss_explainer = Paragraph(
            "<b>What is this?</b> This bar chart shows 1 minus the mean within-cluster sum of squares (mean WCSS) for each cluster, on a log scale. "
            "<b>Why look at it?</b> Higher values indicate tighter, more specific clusters; lower values indicate more spread out or miscellaneous clusters. The log scale helps visualize differences when values are close to 1.",
            modern_normal)
        story.append(KeepTogether([
            Paragraph("Per-Cluster Tightness (1 - Mean WCSS, log scale)", modern_section_style),
            wcss_explainer,
            Spacer(1, 6),
            Image(wcss_buffer, width=500, height=180),
            Spacer(1, 8),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8)
        ]))

    # Topic Evolution Analysis plot
    plt.figure(figsize=(12, 6))
    analyzer.visualize_cluster_frequencies(papers_library.papers, verbose=verbose)
    plt.tight_layout()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    # Topic Evolution Analysis explainer
    te_explainer = Paragraph(
        "<b>What is this?</b> This heatmap shows how the distribution of clusters changes over time. "
        "Each row is a cluster, and each column is a time window. "
        "<b>Why look at it?</b> It helps you see which topics are emerging, stable, or fading in your library.",
        modern_normal)
    story.append(KeepTogether([
        Paragraph("Topic Evolution Analysis", modern_section_style),
        te_explainer,
        Spacer(1, 6),
        Image(img_buffer, width=500, height=180),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8)
    ]))

    # Build PDF
    doc.build(story)
    
    if report_mode == 'html':
        def img_to_base64(buffer):
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

        html = [
            '<html><head><meta charset="utf-8"><title>Library Analysis Report</title>',
            '<style>body{font-family:Helvetica,Arial,sans-serif;background:#f8f9fa;color:#22223b;margin:0;padding:0;} .container{max-width:900px;margin:30px auto;background:#fff;border-radius:10px;box-shadow:0 2px 8px #ccc;padding:32px;} h1{color:#22223b;} h2{color:#4a4e69;background:#f2e9e4;padding:8px 12px;border-radius:6px;} .explainer{margin:8px 0 18px 0;color:#555;font-size:1.05em;} table{border-collapse:collapse;width:100%;margin-bottom:24px;} th,td{border:1px solid #c9ada7;padding:8px;text-align:center;} th{background:#4a4e69;color:#fff;} tr:nth-child(even){background:#f2e9e4;} .section{margin-bottom:40px;} img{display:block;margin:0 auto 12px auto;max-width:100%;border-radius:8px;box-shadow:0 1px 4px #bbb;} hr{border:none;border-top:1px solid #c9ada7;margin:32px 0;}</style></head><body><div class="container">'
        ]
        html.append('<h1>Library Analysis Report</h1><hr>')
        # Cluster Overview
        html.append('<div class="section"><h2>Cluster Overview</h2>')
        html.append('<div class="explainer"><b>What is this?</b> This table summarizes each cluster, showing its size, top keywords, and trend over time. <b>Why look at it?</b> It gives a quick overview of the main topics in your library and how they are changing.</div>')
        html.append('<table><tr>' + ''.join(f'<th>{col}</th>' for col in table_data[0]) + '</tr>')
        for row in table_data[1:]:
            html.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>')
        html.append('</table></div><hr>')
        # Topic Identification Analysis
        html.append('<div class="section"><h2>Topic Identification Analysis</h2>')
        html.append('<div class="explainer"><b>What is this?</b> These plots show how well your data clusters for different numbers of clusters. The left plot shows the silhouette score (higher is better), and the right shows the elbow method (lower is better). <b>Why look at it?</b> It helps you choose the best number of clusters for your data.</div>')
        html.append(f'<img src="data:image/png;base64,{img_to_base64(silhouette_buffer)}" alt="Silhouette and Elbow Plot">')
        html.append('</div><hr>')
        # Per-Cluster Tightness (Mean WCSS)
        if model_tfidf is not None:
            html.append('<div class="section"><h2>Per-Cluster Tightness (1 - Mean WCSS, log scale)</h2>')
            html.append('<div class="explainer"><b>What is this?</b> This bar chart shows 1 minus the mean within-cluster sum of squares (mean WCSS) for each cluster, on a log scale. <b>Why look at it?</b> Higher values indicate tighter, more specific clusters; lower values indicate more spread out or miscellaneous clusters. The log scale helps visualize differences when values are close to 1.</div>')
            html.append(f'<img src="data:image/png;base64,{img_to_base64(wcss_buffer)}" alt="1 - Mean WCSS Bar Chart (log scale)">')
            html.append('</div><hr>')
        # Topic Evolution Analysis
        html.append('<div class="section"><h2>Topic Evolution Analysis</h2>')
        html.append('<div class="explainer"><b>What is this?</b> This heatmap shows how the distribution of clusters changes over time. Each row is a cluster, and each column is a time window. <b>Why look at it?</b> It helps you see which topics are emerging, stable, or fading in your library.</div>')
        html.append(f'<img src="data:image/png;base64,{img_to_base64(img_buffer)}" alt="Cluster Distribution Over Time Heatmap">')
        html.append('</div></div></body></html>')
        html_path = os.path.join(run_dir, f"library_analysis_report_{timestamp}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))
        print(f"HTML report saved to {html_path}")
        # Save cluster preferences to text file as before
        preferences_file = os.path.join(run_dir, 'feed_search_preferences.txt')
        with open(preferences_file, 'w') as f:
            f.write('\n'.join(cluster_lines))
        print(f"Feed preferences saved to {preferences_file}")
        return
    
    # Save cluster preferences to text file
    preferences_file = os.path.join(run_dir, 'feed_search_preferences.txt')
    with open(preferences_file, 'w') as f:
        f.write('\n'.join(cluster_lines))
    
    print(f"Report saved to {run_dir}")
    print(f"Feed preferences saved to {preferences_file}")

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
        papers_library = lang_helper.load_library(args.library)
    else:
        # Use default data directory
        library_path = lang_helper.get_recent_library(data_dir)
        if library_path is None:
            print(f"Error: Could not find library in {data_dir}")
            return
        papers_library = lang_helper.load_library(library_path)
    
    if papers_library is None:
        print(f"Error: Could not load library")
        return
    
    # Load word model
    pubmed_wordmodel = lang_helper.load_word_model(model_path)
    if pubmed_wordmodel is None:
        print(f"Error: Could not load word model")
        return
    
    # Load or create models
    dictionary, bow_corpus, model_tfidf, termsim_matrix = lang_helper.load_or_create_models(papers_library, pubmed_wordmodel, model_path, args.overwrite)
    
    # Filter library and tokenize
    filtered_library = lang_helper.filter_library(papers_library, type='title')
    tokenized_library = lang_helper.tokenize_library(filtered_library, wordmodel=pubmed_wordmodel)
    bow_corpus = [dictionary.doc2bow(text) for text in tokenized_library]
    
    # Compute similarity matrix once
    similarity_matrix = lang_helper.get_cosine_matrix(termsim_matrix, bow_corpus, bow_corpus)

    # Always run optimization to get curves for both methods
    silhouette_opt_n, silhouette_scores, inertias, best_n_silhouette, best_n_elbow = lang_helper.find_optimal_clusters(
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
    lang_helper.compute_topics(similarity_matrix, filtered_library.papers, method=args.clustering_method, num_topics=optimal_n, verbose=args.verbose)

    # Print clustering results
    print(f"Number of clusters used: {optimal_n}")
 

    # Save models (only if overwrite is True)
    if args.overwrite:
        print("Saving models...")
        lang_helper.save_models(dictionary, model_tfidf, termsim_matrix, model_path)

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
    generate_cluster_report(filtered_library, bow_corpus, dictionary, run_dir,
                          window_size=args.window_size, step_size=args.step_size, verbose=args.verbose,
                          silhouette_scores=silhouette_scores, inertias=inertias, 
                          best_n_silhouette=best_n_silhouette, best_n_elbow=best_n_elbow,
                          num_clusters=num_clusters, model_tfidf=model_tfidf, report_mode=args.report_mode)

    # Save papers library to output directory
    print("Saving papers library...")
    lang_helper.save_papers_library(filtered_library, run_dir)

    print("Analysis complete!")

if __name__ == "__main__":
    main() 
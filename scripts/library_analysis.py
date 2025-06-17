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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.platypus import Image
from datetime import datetime

def generate_cluster_report(papers_library, bow_corpus, dictionary, window_size=50, step_size=50, verbose=False):
    """
    Generate a PDF report analyzing the library clusters and a simple text file for feed search preferences.
    
    Args:
        papers_library: List of papers to analyze
        bow_corpus: Bag-of-words corpus
        dictionary: Gensim dictionary
        window_size: Number of papers in each analysis window
        step_size: Number of papers to step forward for each window
        verbose: Whether to print detailed debugging information
    """
    # Set default output directory to 'output' in the program directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped folder inside output directory
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
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
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Library Analysis Report", title_style))
    
    # Create a two-column layout for the overview section
    overview_data = []
    
    # Cluster Overview
    overview_data.append([Paragraph("Cluster Overview", styles['Heading2']), ""])
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
        
        # Calculate trend based on frequency changes
        cluster_idx = np.where(cluster_ids == cluster)[0][0]
        window_freqs = frequencies[:, cluster_idx]
        if len(window_freqs) > 1:
            growth_rate = (window_freqs[-1] - window_freqs[0]) / len(window_freqs)
            if growth_rate > 0.1:
                trend = "Emerging"
            elif growth_rate < -0.1:
                trend = "Fading"
            else:
                trend = "Stable"
        else:
            trend = "N/A"
        
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
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Reduced font size for headers
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('WORDWRAP', (0, 0), (-1, -1), True),  # Enable word wrap for all cells
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Smaller font for data rows
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Center content vertically
        ('LEFTPADDING', (0, 0), (-1, -1), 6),  # Add some padding
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Topic Evolution Analysis
    story.append(Paragraph("Topic Evolution Analysis", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Create a two-column layout for evolution analysis
    evolution_data = []
    evolution_data.append([Paragraph("Cluster Frequencies by Window:", styles['Heading3']), ""])
    
    for i, window in enumerate(window_labels):
        window_text = f"\n{window}:"
        for j, cluster_id in enumerate(cluster_ids):
            if frequencies[i, j] > 0:  # Only show clusters with papers
                window_text += f"\n  Cluster {cluster_id}: {frequencies[i, j]:.2%}"
        evolution_data.append([Paragraph(window_text, styles['Normal']), ""])
    
    # Add evolution data to story
    for row in evolution_data:
        story.append(row[0])
        story.append(Spacer(1, 6))
    
    # Visualizations
    story.append(Paragraph("Visualizations", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Create cluster frequencies visualization with larger size for landscape
    plt.figure(figsize=(12, 6))
    analyzer.visualize_cluster_frequencies(papers_library.papers, verbose=verbose)
    plt.tight_layout()
    
    # Save to BytesIO buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    # Add to PDF with adjusted size for landscape
    story.append(Paragraph("Cluster Frequencies", styles['Heading3']))
    story.append(Spacer(1, 6))
    story.append(Image(img_buffer, width=650, height=325))
    
    # Build PDF
    doc.build(story)
    
    # Save cluster preferences to text file
    preferences_file = os.path.join(run_dir, 'feed_search_preferences.txt')
    with open(preferences_file, 'w') as f:
        f.write('\n'.join(cluster_lines))
    
    print(f"Report saved to {run_dir}")
    print(f"Feed preferences saved to {preferences_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze library clusters and generate reports.')
    parser.add_argument('--clustering_method', type=str, default='ward', help='Clustering method (ward or complete)')
    parser.add_argument('--window_size', type=int, default=50, help='Number of papers in each analysis window')
    parser.add_argument('--step_size', type=int, default=50, help='Number of papers to step forward for each window')
    parser.add_argument('--verbose', action='store_true', help='Print detailed debugging information')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing models')
    args = parser.parse_args()
    
    # Set default paths for library file and model path
    library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # Load library
    papers_library = lang_helper.load_recent_library(library_file)
    
    # Load or create models
    dictionary, model_tfidf, termsim_matrix = lang_helper.load_or_create_models(papers_library, model_path, args.overwrite)
    
    # Tokenize library
    tokenized_library = lang_helper.tokenize_library(papers_library)
    bow_corpus = [dictionary.doc2bow(text) for text in tokenized_library]
    
    # Cluster library
    optimal_n, silhouette_scores = lang_helper.cluster_library(bow_corpus, termsim_matrix, papers_library,
                                                     method=args.clustering_method, verbose=args.verbose)
    
    # Print clustering results
    print(f"Optimal number of clusters: {optimal_n}")
    print(f"Silhouette score for {optimal_n} clusters: {silhouette_scores[optimal_n]:.3f}")
    
    # Save models and papers library
    if args.overwrite:
        print("Saving models and papers library...")
        lang_helper.save_models(dictionary, model_tfidf, termsim_matrix, model_path, papers_library)
    
    # Generate report
    print("Generating analysis report...")
    generate_cluster_report(papers_library, bow_corpus, dictionary,
                          window_size=args.window_size, step_size=args.step_size, verbose=args.verbose)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 
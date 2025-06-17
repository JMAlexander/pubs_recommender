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
from dotenv import load_dotenv
import json

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
print(f"\nLoading environment variables from {env_path}")
load_dotenv(env_path)

def load_models(model_dir):
    """Load the necessary models for analysis."""
    dictionary = corpora.Dictionary.load(os.path.join(model_dir, 'dictionary.gensim'))

    # Download it if it doesn't exist locally
    expected_model_location = os.path.join(model_dir, 'pubmed2018_w2v_200D.bin')
    if (not os.path.exists(expected_model_location)):
        print("Downloading the Pubmed model...")
    lang_helper.load_file_from_s3(origin_path='lang_processing/gensim-data/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin',
                       final_path=expected_model_location)

    # Load the Pubmed model
    print("Loading the Pubmed model and building the similarity index...")
    pubmed_wordmodel = models.KeyedVectors.load_word2vec_format(expected_model_location, binary=True)
    termsim_index = WordEmbeddingSimilarityIndex(pubmed_wordmodel)

    return dictionary, termsim_index

def create_models(dictionary, combined_corpus, termsim_index):
    """
    Create TF-IDF model and term similarity matrix using both library and feed corpora.
    
    Args:
        dictionary: Gensim dictionary
        library_corpus: BOW corpus of the library papers
        feed_corpus: BOW corpus of the new feed publications
    
    Returns:
        model_tfidf: TF-IDF model
        termsim_matrix: Term similarity matrix
    """
    
    # Create TF-IDF model
    model_tfidf = models.TfidfModel(combined_corpus)
    
    # Create term similarity matrix
    termsim_matrix = SparseTermSimilarityMatrix(
        source=termsim_index,
        dictionary=dictionary,
        tfidf=model_tfidf
    )
    
    return model_tfidf, termsim_matrix

def process_feeds(feeds, include_read):
    """
    Process the RSS feeds and tokenize the publications.
    """
    # Check feeds
    lang_helper.check_feeds(feeds)
    
    # Tokenize feeds
    pubs, tokenized_pubs = lang_helper.tokenize_feeds(feeds, type='title', include_read=include_read)
    print(f"\nInitial number of papers from feeds: {len(pubs)}")
    
    return pubs, tokenized_pubs

def generate_and_send_email(email, pubs, papers, bow_corpus, similarity_matrix, dictionary, include_read_items=False):
    """
    Generate and send an email with publication recommendations.
    
    Args:
        pubs: List of publications to recommend
        papers: List of Paper objects with cluster IDs
        bow_corpus: Bag of words corpus for new publications
        similarity_matrix: Similarity matrix between papers and publications
        dictionary: Gensim dictionary
        include_read_items: Whether to include already read items
    """
    # Generate email message
    
    # Send email


def load_analyzed_library(archive_dir):
    """
    Load the analyzed library from papers_library_with_clusters.json in the data directory.
    
    Args:
        archive_dir: Directory containing the paper archive
        
    Returns:
        Library object with cluster information
        
    Raises:
        FileNotFoundError: If papers_library_with_clusters.json is not found in data directory
    """
    # Look for the JSON file in the data directory
    library_path = os.path.join(archive_dir, 'papers_library_with_clusters.json')
    if not os.path.exists(library_path):
        raise FileNotFoundError(f"Analyzed library not found at {library_path}. Please run analyze_cluster.py first.")
    
    # Create new library object
    library = lang_helper.Library()
    
    # Load JSON data
    with open(library_path, 'r') as f:
        papers_data = json.load(f)
    
    # Create Paper objects from JSON data
    for paper_dict in papers_data:
        paper = lang_helper.Paper()
        paper.title = paper_dict['title']
        paper.abstract = paper_dict['abstract']
        paper.authors = paper_dict['authors']
        paper.date_created = paper_dict['date_created']
        paper.date_updated = paper_dict['date_updated']
        paper.read = paper_dict['read']
        paper.notes = paper_dict['notes']
        paper.ratings = paper_dict['ratings']
        paper.doi = paper_dict['doi']
        paper.pmid = paper_dict['pmid']
        paper.pmcid = paper_dict['pmcid']
        paper.year = paper_dict['year']
        paper.journal = paper_dict['journal']
        paper.date = paper_dict['date']
        paper.pages = paper_dict['pages']
        paper.issue = paper_dict['issue']
        paper.volume = paper_dict['volume']
        paper.cluster_id = paper_dict['cluster_id']
        paper.silhouette_score = paper_dict['silhouette_score']
        library.papers.append(paper)
    
    return library

def load_cluster_preferences(config_dir):
    """Load cluster preferences from CSV file and return list of included cluster IDs."""
    preferences_file = os.path.join(config_dir, 'cluster_search_preferences.csv')
    if not os.path.exists(preferences_file):
        return []  # Return empty list if file doesn't exist
    preferences_df = pd.read_csv(preferences_file)
    return preferences_df[preferences_df['include_in_search'] == 'YES']['cluster_id'].tolist()

def main():
    parser = argparse.ArgumentParser(description='Analyze new publications and send recommendations.')
    parser.add_argument('--include-read', action='store_true', default=False,
                        help='Include already read items in recommendations')
    parser.add_argument('--target-email', type=str, default=os.getenv('TARGET_EMAIL'),
                        help='Email address to send the recommendations to')
    args = parser.parse_args()

    # Set up directory paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    archive_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    config_dir = os.path.join(base_dir, 'config')
    expected_model_location = os.path.join(models_dir, 'pubmed2018_w2v_200D.bin')
    
    lang_helper.load_file_from_s3(origin_path='lang_processing/gensim-data/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin',
                                 final_path=expected_model_location)

    # Load dictionary
    dictionary, termsim_index = load_models(models_dir)
    
    # Read feeds from file
    feeds_file = os.path.join(config_dir, 'feeds.txt')
    with open(feeds_file, 'r') as f:
        # Skip lines starting with # or ##, only keep URLs
        feeds = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    print(f"\nNumber of feeds: {len(feeds)}")
    
    # Get new publications
    pubs, tokenized_pubs = process_feeds(feeds, args.include_read)
    
    # Save new publications to pubs_reviewed.csv
    lang_helper.save_sent_papers(pubs)

    # Add new documents to dictionary
    dictionary.add_documents(tokenized_pubs)
    
    # Load library with cluster information
    papers_library = load_analyzed_library(archive_dir)
    tokenized_library = lang_helper.tokenize_library(papers_library, type='title')
    print(f"\nNumber of papers in library: {len(papers_library.papers)}")
    
    # Create bow corpora
    corpus_pubs = [dictionary.doc2bow(text) for text in tokenized_pubs]
    corpus_library = [dictionary.doc2bow(text) for text in tokenized_library]
    combined_corpus = corpus_pubs + corpus_library

    # Create models
    model_tfidf, termsim_matrix = create_models(dictionary, combined_corpus, termsim_index)
    
    # Compare publications and get similarity matrix
    library_pubs_similarity_matrix = lang_helper.get_cosine_matrix(termsim_matrix, corpus_library, corpus_pubs)
    
    # Get cluster IDs from papers
    topic_labels = [paper.cluster_id for paper in papers_library.papers]
    
    # Get topic similarity matrix
    topic_pubs_similarity_matrix = lang_helper.get_topic_similarity(library_pubs_similarity_matrix, topic_labels)
    print(f"\nTopic Pubs Matrix Size: {topic_pubs_similarity_matrix.shape}")

    # Load cluster preferences
    included_clusters = load_cluster_preferences(config_dir)
    if included_clusters:
        print(f"\nIncluding only clusters: {included_clusters}")
    else:
        print("\nNo cluster preferences found, including all clusters")
    
    # Generate email
    email_message = lang_helper.draft_email(pubs, papers_library.papers, corpus_library, topic_pubs_similarity_matrix, dictionary, included_clusters=included_clusters)

    # Send email
    if not args.target_email:
        raise ValueError("TARGET_EMAIL environment variable not set")
    else:
        lang_helper.send_email_with_Web_API(email_message, args.target_email)

if __name__ == "__main__":
    main() 
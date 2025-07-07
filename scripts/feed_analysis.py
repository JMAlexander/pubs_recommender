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
from dotenv import load_dotenv
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from analysis modules
from pubs_recommender import models as pr_models
from pubs_recommender import feeds as pr_feeds
from pubs_recommender import utils as pr_utils
from pubs_recommender import email as pr_email
from pubs_recommender import get_cosine_matrix, get_topic_similarity, match_topics_to_publications

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
print(f"\nLoading environment variables from {env_path}")
load_dotenv(env_path)

def main():
    parser = argparse.ArgumentParser(description='Analyze new publications and send recommendations.')
    parser.add_argument('--include-read', action='store_true', default=False,
                        help='Include already read items in recommendations')
    parser.add_argument('--target-email', type=str, default=os.getenv('TARGET_EMAIL'),
                        help='Email address to send the recommendations to')
    parser.add_argument('--html-email', action='store_true', default=False,
                        help='Send email in HTML format instead of plain text')
    args = parser.parse_args()

    # Set up directory paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    archive_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    config_dir = os.path.join(base_dir, 'config')

    # Load dictionary and models
    dictionary, termsim_index, pubmed_wordmodel = pr_models.load_existing_models_for_feeds(models_dir)
    
    # Read feeds from file
    feeds_file = os.path.join(config_dir, 'feeds.txt')
    with open(feeds_file, 'r') as f:
        # Skip lines starting with # or ##, only keep URLs
        feeds = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    print(f"\nNumber of feeds: {len(feeds)}")
    
    # Get new publications
    pubs, tokenized_pubs = pr_feeds.process_feeds(feeds, args.include_read, pubmed_wordmodel)
    
    # Save new publications to pubs_reviewed.csv
    pr_feeds.save_sent_papers(pubs)

    # Add new documents to dictionary
    dictionary.add_documents(tokenized_pubs)
    
    # Filter dictionary to only include terms present in the PubMed word model
    # This prevents KeyError when SparseTermSimilarityMatrix tries to access tfidf.idfs
    dictionary.filter_tokens(bad_ids=[tokenid for tokenid, term in dictionary.items() if term not in pubmed_wordmodel])
    
    # Load library with cluster information
    papers_library = pr_models.load_analyzed_library(archive_dir)
    tokenized_library = pr_utils.tokenize_library(papers_library, field='title', wordmodel=pubmed_wordmodel)
    print(f"\nNumber of papers in library: {len(papers_library.papers)}")
    
    # Create bow corpora
    corpus_pubs = [dictionary.doc2bow(text) for text in tokenized_pubs]
    corpus_library = [dictionary.doc2bow(text) for text in tokenized_library]
    combined_corpus = corpus_pubs + corpus_library

    # Create models
    model_tfidf, termsim_matrix = pr_models.create_tfidf_and_similarity_matrix(dictionary, combined_corpus, termsim_index)
    
    # Compare publications and get similarity matrix
    library_pubs_similarity_matrix = get_cosine_matrix(termsim_matrix, corpus_library, corpus_pubs)
    
    # Get cluster IDs from papers
    topic_labels = [paper.cluster_id for paper in papers_library.papers]
    
    # Get topic similarity matrix
    topic_pubs_similarity_matrix = get_topic_similarity(library_pubs_similarity_matrix, topic_labels)
    print(f"\nTopic Pubs Matrix Size: {topic_pubs_similarity_matrix.shape}")

    # Load cluster preferences
    included_clusters = pr_utils.load_cluster_preferences(config_dir)
    if included_clusters:
        print(f"\nIncluding only clusters: {included_clusters}")
    else:
        print("\nNo cluster preferences found, including all clusters")
    
    # Match topics to publications
    topic_results = match_topics_to_publications(pubs, papers_library.papers, topic_pubs_similarity_matrix, included_clusters=included_clusters)
    
    # Generate email
    if args.html_email:
        email_message = pr_email.draft_html_email(pubs, papers_library.papers, corpus_library, topic_pubs_similarity_matrix, dictionary, topic_results)
    else:
        email_message = pr_email.draft_plaintext_email(pubs, papers_library.papers, corpus_library, topic_pubs_similarity_matrix, dictionary, topic_results)

    # Send email
    if not args.target_email:
        raise ValueError("TARGET_EMAIL environment variable not set")
    else:
        if args.html_email:
            pr_email.send_email_with_Web_API(email_message, args.target_email, html_content=email_message)
        else:
            pr_email.send_email_with_Web_API(email_message, args.target_email)

if __name__ == "__main__":
    main() 
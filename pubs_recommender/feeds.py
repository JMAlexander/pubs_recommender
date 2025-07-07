import feedparser
from Bio import Entrez
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
from .utils import get_stop, clean_and_split_term

def tokenize_feeds(feeds, stop_method='nltk', type='title', include_read=False, wordmodel=None):
    """
    Tokenize publications from RSS feeds.
    
    Args:
        feeds: List of RSS feed URLs
        stop_method: Method for stopword removal
        type: Type of content to tokenize ('title' or 'abstract')
        include_read: Whether to include already read papers
        wordmodel: Word embedding model for cleaning and splitting terms
        
    Returns:
        tuple: (included_pubs, tokenized_texts)
    """
    # Get stopword corpus
    stop_words = get_stop(stop_method=stop_method)
    
    pubs = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            pubs.append(entry)
    
    print(f"\nTotal papers from feeds: {len(pubs)}")
    
    # Load previously sent papers
    sent_papers = load_sent_papers()
    
    # Filter out papers that have already been sent
    if include_read:
        included_pubs = pubs
    else:
        included_pubs = [pub for pub in pubs if get_paper_id(pub) not in sent_papers]
        
    print(f"Number of included papers: {len(included_pubs)}")
    tokenized_texts = []
    if (type == 'title'):
        for pub in included_pubs:
            words = pub.title.lower().split()
            if wordmodel is not None:
                cleaned_words = []
                for word in words:
                    cleaned_words.extend(clean_and_split_term(word, wordmodel))
                words = cleaned_words
            tokenized_texts.append([word for word in words if word not in stop_words])
    elif (type == 'abstract'):
        for pub in included_pubs:
            words = pub.abstract.lower().split()
            if wordmodel is not None:
                cleaned_words = []
                for word in words:
                    cleaned_words.extend(clean_and_split_term(word, wordmodel))
                words = cleaned_words
            tokenized_texts.append([word for word in words if word not in stop_words])
    return included_pubs, tokenized_texts

def tokenize_pubmed(email, past_days, stop_method='nltk', max_results=1000, pub_types=None):
    """
    Fetch abstracts of articles published in the last week from PubMed.

    Args:
        email (str): Your email address (required by NCBI Entrez).
        past_days (int): Number of days to look back
        max_results (int): Maximum number of articles to fetch.
        pub_types (list): List of publication types to search for

    Returns:
        tuple: (abstracts, tokenized_texts)
    """
    stop_words = get_stop(stop_method=stop_method)
    
    Entrez.email = email
    # Build the query with publication types
    pub_type_query = " OR ".join([f'"{pub_type}"[Publication Type]' for pub_type in pub_types])
    
    # Search for articles published in the last 7 days
    today = datetime.today()
    one_week_ago = today - timedelta(days=past_days)

    # Format the dates for PubMed's search syntax
    today_str = today.strftime("%Y/%m/%d")
    one_week_ago_str = one_week_ago.strftime("%Y/%m/%d")

    # Construct the search term
    search_term = f"(({pub_type_query}) AND ({one_week_ago_str}[Date - Publication] : {today_str}[Date - Publication]))"
    
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    
    ids = record["IdList"]
    if not ids:
        print("No articles found.")
        return []

    # Fetch article details
    handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    abstracts = []
    for article in records["PubmedArticle"]:
        try:
            abstract = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
            abstracts.append(" ".join(abstract))
        except KeyError:
            # Skip articles with no abstract
            continue

    tokenized_texts = [text.lower().split() for text in abstracts]
    tokenized_texts =[[word for word in doc if word not in stop_words] for doc in tokenized_texts]
    
    return abstracts, tokenized_texts

def check_feeds(feeds):
    """
    Check if RSS feeds are accessible.
    
    Args:
        feeds: List of RSS feed URLs
        
    Returns:
        List of accessible feeds
    """
    accessible_feeds = []
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            if feed.entries:
                accessible_feeds.append(feed_url)
                print(f"✓ {feed_url}: {len(feed.entries)} entries")
            else:
                print(f"✗ {feed_url}: No entries found")
        except Exception as e:
            print(f"✗ {feed_url}: Error - {str(e)}")
    
    return accessible_feeds

def get_topic_matching_pubs(pubs, similarity_matrix, topic_num, threshold_similarity=0.2):
    """
    Get publications matching a specific topic.
    
    Args:
        pubs: List of publications
        similarity_matrix: Similarity matrix between topics and publications
        topic_num: Topic number to match
        threshold_similarity: Minimum similarity threshold
        
    Returns:
        List of matching publications
    """
    print(f"\nDebug get_topic_matching_pubs for cluster {topic_num}:")
    print(f"Looking at row {topic_num-1} of similarity matrix")
    print(f"Row values: {similarity_matrix[topic_num - 1]}")
    topic_pub_indices = np.where(similarity_matrix[topic_num - 1] > threshold_similarity)[0]
    print(f"Indices with score > {threshold_similarity}: {topic_pub_indices}")
    topic_pub_indices = np.sort(topic_pub_indices)[::-1]
    matching_pubs = [pubs[idx] for idx in topic_pub_indices]
    print(f"Returning {len(matching_pubs)} papers for topic {topic_num}")
    for idx, pub in zip(topic_pub_indices, matching_pubs):
        print(f"Index {idx}: {pub.title}")
    return matching_pubs

def load_sent_papers():
    """
    Load the list of papers that have already been sent in digests.
    
    Returns:
        Set of paper IDs that have been sent
    """
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        df = pd.read_csv(os.path.join(data_dir, 'pubs_reviewed.csv'))
        return set(df['paper_id'].tolist())
    except FileNotFoundError:
        return set()

def save_sent_papers(new_pubs):
    """
    Save the list of papers that have already been sent in digests.
    
    Args:
        new_pubs: List of publications to mark as sent
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a list of dictionaries with paper metadata
    papers_data = []
    for pub in new_pubs:
        paper_data = {
            'paper_id': get_paper_id(pub),
            'title': pub.title,
            'date_sent': datetime.now().strftime('%Y-%m-%d'),
            'doi': pub.prism_doi if hasattr(pub, 'prism_doi') else '',
            'link': pub.link if hasattr(pub, 'link') else ''
        }
        papers_data.append(paper_data)
    
    # Convert to DataFrame
    new_df = pd.DataFrame(papers_data)
    
    try:
        # Try to load existing file
        existing_df = pd.read_csv(os.path.join(data_dir, 'pubs_reviewed.csv'))
        # Concatenate with new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Remove duplicates based on paper_id
        combined_df = combined_df.drop_duplicates(subset=['paper_id'], keep='last')
    except FileNotFoundError:
        # If file doesn't exist, use only new data
        combined_df = new_df
    
    # Save to CSV
    combined_df.to_csv(os.path.join(data_dir, 'pubs_reviewed.csv'), index=False)
    print(f"Saved {len(new_pubs)} new papers to pubs_reviewed.csv")

def get_paper_id(pub):
    """
    Get a unique identifier for a publication.
    
    Args:
        pub: Publication object
        
    Returns:
        Unique paper identifier
    """
    # Try to get DOI first, then title as fallback
    if hasattr(pub, 'prism_doi') and pub.prism_doi:
        return pub.prism_doi
    else:
        return pub.title

def match_topics_to_publications(pubs, papers, similarity_matrix, included_clusters=None):
    """
    Match topics to publications based on similarity.
    
    Args:
        pubs: List of publications
        papers: List of papers from library
        similarity_matrix: Similarity matrix between topics and publications
        included_clusters: List of cluster IDs to include (None for all)
        
    Returns:
        Dictionary mapping cluster IDs to matching publications
    """
    topic_results = {}
    
    # Get unique cluster IDs from papers
    cluster_ids = np.unique([p.cluster_id for p in papers if p.cluster_id is not None])
    
    # Filter clusters if specified
    if included_clusters is not None:
        cluster_ids = [c for c in cluster_ids if c in included_clusters]
    
    print(f"\nMatching {len(cluster_ids)} clusters to {len(pubs)} publications...")
    
    for cluster_id in cluster_ids:
        print(f"\nProcessing cluster {cluster_id}...")
        
        # Get publications matching this topic
        matching_pubs = get_topic_matching_pubs(pubs, similarity_matrix, cluster_id, threshold_similarity=0.2)
        
        if matching_pubs:
            topic_results[cluster_id] = matching_pubs
            print(f"Found {len(matching_pubs)} matching publications for cluster {cluster_id}")
        else:
            print(f"No matching publications found for cluster {cluster_id}")
    
    return topic_results

def process_feeds(feeds, include_read, pubmed_wordmodel):
    """
    Process the RSS feeds and tokenize the publications.
    
    Args:
        feeds: List of RSS feed URLs
        include_read: Whether to include already read papers
        pubmed_wordmodel: Word embedding model for cleaning and splitting terms
        
    Returns:
        tuple: (pubs, tokenized_pubs)
    """
    check_feeds(feeds)
    pubs, tokenized_pubs = tokenize_feeds(feeds, type='title', include_read=include_read, wordmodel=pubmed_wordmodel)
    print(f"\nInitial number of papers from feeds: {len(pubs)}")
    return pubs, tokenized_pubs 
##Necessary dependencies
import pandas as pd
import numpy as np
from gensim import corpora
from gensim import models
from gensim import similarities
import feedparser
from Bio import Entrez
from datetime import datetime, timedelta
import glob
import os
import source as lang_helper
import boto3
import nltk
from scipy.cluster.hierarchy import linkage, fcluster
import smtplib
from email.mime.text import MIMEText
import requests
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import squareform

load_dotenv('/home/jeff/data_server_home/DS_projects/literature_analysis/.env')

class Paper:
    def __init__(self):
        self.title = None
        self.abstract = None
        self.authors = None
        self.date_created = None
        self.date_updated = None
        self.read = None
        self.notes = None
        self.ratings = None
        self.doi = None
        self.pmid = None
        self.pmcid = None
        self.year = None
        self.author = None
        self.journal = None
        self.date = None
        self.pages = None
        self.issue = None
        self.volume = None
        self.cluster_id = None
        self.silhouette_score = None

    def load_from_row(self, row):
        self.title = row['title']
        self.abstract = row['abstract']
        self.authors = row['author']
        self.date_created = row['created (Read-Only)']
        self.date_updated = row['updated (Read-Only)']
        self.read = row['read']
        self.notes = row['notes']
        self.ratings = row['ratings']
        self.doi = row['doi']
        self.pmid = row['pmid']
        self.pmcid = row['pmcid']
        self.year = row['year']
        self.author = row['author']
        self.journal = row['journal']
        self.date = row['date']
        self.pages = row['pages']
        self.issue = row['issue']
        self.volume = row['volume']
        self.cluster_id = None
        self.silhouette_score = None

class Library:
    def __init__(self):
        self.papers = []  # List of Paper objects

    def load(self, file):
        lib = pd.read_csv(file)
        for _, row in lib.iterrows():
            paper = Paper()
            paper.load_from_row(row)
            self.papers.append(paper)

def get_stop(stop_method='nltk'):
    if (stop_method == 'nltk'):
        stop_words = set(nltk.corpus.stopwords.words('english'))

    return stop_words
        

def tokenize_library(library, stop_method='nltk', type='title'):
    ##Get stopword corpus
    stop_words = get_stop(stop_method=stop_method)
        
    tokenized_texts = []
    if (type == 'abstract'):
        for paper in library.papers:
            if not(isinstance(paper.abstract, float) and np.isnan(paper.abstract)): 
                tokenized_texts.append(paper.abstract.lower().split())
    elif (type == 'title'):
        for paper in library.papers:
            if not(isinstance(paper.title, float) and np.isnan(paper.title)): 
                tokenized_texts.append(paper.title.lower().split())
    tokenized_texts =[[word for word in doc if word not in stop_words] for doc in tokenized_texts]
    
    return tokenized_texts

def tokenize_feeds(feeds, stop_method='nltk', type='title', include_read=False):
    ##Get stopword corpus
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

    if (type == 'title'):
        tokenized_texts = [pub.title.lower().split() for pub in included_pubs]
    elif (type == 'abstract'):
        tokenized_text = [pub.abstract.lower().split() for pub in included_pubs]
    tokenized_texts =[[word for word in doc if word not in stop_words] for doc in tokenized_texts]
    
    return included_pubs, tokenized_texts

def tokenize_pubmed(email, past_days, stop_method='nltk', max_results=1000, pub_types=None):
    """
    Fetch abstracts of articles published in the last week from PubMed.

    Args:
        email (str): Your email address (required by NCBI Entrez).
        max_results (int): Maximum number of articles to fetch.

    Returns:
        list: A list of article abstracts.
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

def load_recent_library(directory):
    today = datetime.today().date()
    library_paths = glob.glob(f"{directory}/????_??_??_papers_library.csv")

    ##Initalize variables for selecting most recent archive
    closest_archive = None
    min_diff = float("inf")
    for path in library_paths:
        archive = os.path.basename(path)
        date_str = archive[:10]  # First 10 characters (YYYY_MM_DD)
        archive_date = datetime.strptime(date_str, "%Y_%m_%d").date()
        # Compute days since the archive
        diff = abs((archive_date - today).days)

        if diff < min_diff:
            min_diff = diff
            closest_archive = path

    loaded_library = Library()
    loaded_library.load(closest_archive)

    return loaded_library

def load_file_from_s3(origin_path, final_path, endpoint_url='https://nyc3.digitaloceanspaces.com', bucket='phillygenome-space'):
    """Load a file from S3 if it doesn't exist locally."""
    if not os.path.exists(final_path):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        os.makedirs(models_dir, exist_ok=True)

    ##Get keys
    s3_access_key = os.getenv('DO_ACCESS_KEY')
    s3_secret_key = os.getenv('DO_SECRET_KEY')

    s3_client = boto3.client(
                             service_name = 's3',
                             endpoint_url=endpoint_url,
                             aws_access_key_id=s3_access_key,
                             aws_secret_access_key=s3_secret_key
    )

    s3_client.download_file(bucket, origin_path, final_path)

def cluster_library(bow_corpus, termsim_matrix, papers_library, method='ward', verbose=False):
    """
    Cluster the library and find optimal number of clusters.
    
    Args:
        bow_corpus: Bag-of-words corpus
        termsim_matrix: Term similarity matrix
        papers_library: Library object containing papers to cluster
        method: Clustering method ('ward' or 'complete')
        verbose: Whether to print detailed debugging information
    
    Returns:
        optimal_n: Optimal number of clusters
        silhouette_scores: Silhouette scores for different cluster numbers
    """
    # Get similarity matrix
    similarity_matrix = get_cosine_matrix(termsim_matrix, bow_corpus, bow_corpus)
    if verbose:
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Find optimal clusters
    optimal_n, silhouette_scores = find_optimal_clusters(similarity_matrix, method=method)
    
    # Compute topics and assign to papers
    compute_topics(similarity_matrix, papers_library.papers, method=method, num_topics=optimal_n, verbose=verbose)
    
    return optimal_n, silhouette_scores

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

def compute_topics(cosine_matrix, papers, method = 'average', num_topics=10, verbose=False):
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
    topic_ids = np.unique(cluster_labels)
    matrix_size = (len(topic_ids), cosine_matrix.shape[1])
    topic_sim_matrix = np.zeros(matrix_size)

    for i, topic in enumerate(topic_ids):
        for j in range(cosine_matrix.shape[1]):
            topic_indices = np.where(cluster_labels == topic)[0]
            topic_sim_matrix[i, j] = np.mean(cosine_matrix[topic_indices, j])

    return topic_sim_matrix

def extract_tfidf_keywords(tfidf_model, dictionary, bow_doc, n=5):
    """
    Extracts the top `n` keywords from a BoW document using a trained TF-IDF model.

    Parameters:
    - tfidf_model: Trained Gensim TfidfModel
    - dictionary: Gensim Dictionary mapping words to IDs
    - bow_doc: BoW representation of a document (list of (word_id, count) tuples)
    - n: Number of top keywords to extract

    Returns:
    - List of top `n` keywords for the document
    """
    tfidf_weights = tfidf_model[bow_doc]  # Compute TF-IDF scores for words in the document
    sorted_tfidf = sorted(tfidf_weights, key=lambda x: x[1], reverse=True)  # Sort by score

    top_n_keywords = [dictionary[word_id] for word_id, _ in sorted_tfidf[:n]]  # Convert IDs to words
    return top_n_keywords


def extract_lda_keywords(dictionary, bow_corpus):
    """
    Extracts keywords and their scores from a corpus using LDA.
    
    Args:
        dictionary: Gensim Dictionary
        bow_corpus: List of bag-of-words documents
        
    Returns:
        list: List of tuples containing (keyword, score) pairs
    """
    #Train the LDA model
    lda = models.LdaModel(bow_corpus, num_topics=1, id2word=dictionary, passes=10)
    topic_terms = lda.get_topic_terms(topicid=0, topn=5)  # Get top 5 terms with their scores
    keywords_with_scores = [(dictionary[id], score) for id, score in topic_terms]
    
    return keywords_with_scores
    

def extract_cluster_docs(bow_corpus, papers, cluster_num):
    """
    Extract documents belonging to a specific cluster.
    
    Args:
        bow_corpus: Bag-of-words corpus
        papers: List of Paper objects
        cluster_num: Cluster ID to extract documents for
        
    Returns:
        List of bag-of-words documents for the specified cluster
    """
    cluster_indices = [i for i, paper in enumerate(papers) if paper.cluster_id == cluster_num]
    cluster_corpus = [bow_corpus[i] for i in cluster_indices]
    
    return cluster_corpus

def send_email_with_SMTP(message, to_address):
    # Email account details
    SMTP_SERVER = "smtp.sendgrid.net"
    SMTP_PORT = 587
    SENDGRID_API_KEY = os.getenv('SG_API_KEY')  # Replace with your actual API key
    EMAIL_ADDRESS = os.getenv('SENDER_EMAIL')

    # Recipient details
    TO_EMAIL = to_address
    SUBJECT = "Publications of Interest"
    BODY = message

    # Create the email
    msg = MIMEText(BODY)
    msg["Subject"] = SUBJECT
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL

    # Send the email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login("apikey", SENDGRID_API_KEY)  # Use "apikey" as the username
            server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
            server.quit()
            print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
        
def send_email_with_Web_API(message, to_address):
    # Email configuration
    SENDGRID_API_KEY = os.getenv('SG_API_KEY')
    FROM_EMAIL = "bot@phillygenome.xyz"
    TO_EMAIL = to_address
    SUBJECT = "Publications of Interest"
    BODY = message

    # Create the email message
    msg = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject=SUBJECT,
        plain_text_content=BODY
    )

    try:
        # Create SendGrid client
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        # Send the email
        response = sg.send(msg)
        print(f"Email sent successfully! Status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def get_topic_matching_pubs(pubs, similarity_matrix, topic_num, threshold_similarity = 0.2):
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
    """Load the list of papers that have already been sent in digests."""
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        df = pd.read_csv(os.path.join(data_dir, 'pubs_reviewed.csv'))
        return set(df['paper_id'].tolist())
    except FileNotFoundError:
        return set()

def save_sent_papers(new_pubs):
    """Save the list of papers that have already been sent in digests."""

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a list of dictionaries with paper metadata
    papers_data = []
    for pub in new_pubs:
        paper_data = {
            'paper_id': get_paper_id(pub),
            'title': pub.title,
            #'publication_name': pub.prism_publicationname if hasattr(pub, 'prism_publicationname') else '',
            'date_sent': datetime.now().strftime('%Y-%m-%d'),
            'doi': pub.prism_doi if hasattr(pub, 'prism_doi') else '',
            'link': pub.link if hasattr(pub, 'link') else ''
        }
        papers_data.append(paper_data)
    
    # Convert to DataFrame
    new_df = pd.DataFrame(papers_data)
    
    try:
        # Try to load existing data
        existing_df = pd.read_csv(os.path.join(data_dir, 'pubs_reviewed.csv'))
        # Combine with new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Remove duplicates based on paper_id
        combined_df = combined_df.drop_duplicates(subset=['paper_id'], keep='first')
    except FileNotFoundError:
        # If file doesn't exist, use only new data
        combined_df = new_df
    
    # Save to CSV
    combined_df.to_csv(os.path.join(data_dir, 'pubs_reviewed.csv'), index=False)

def get_paper_id(pub):
    """Get a unique identifier for a paper (DOI or link)."""
    if hasattr(pub, 'prism_doi'):
        return f"doi:{pub.prism_doi}"
    else:
        return f"link:{pub.link}"

def draft_email(pubs, papers, bow_corpus, similarity_matrix, dictionary, included_clusters=None):
    """
    Draft an email with publication recommendations.
    
    Args:
        pubs: List of publications to recommend
        papers: List of Paper objects with cluster IDs
        bow_corpus: Bag of words corpus for new publications
        similarity_matrix: Similarity matrix between papers and publications
        dictionary: Gensim dictionary
        included_clusters: List of cluster IDs to include in recommendations
    """
    # Get unique clusters from papers
    unique_clusters = sorted(set(paper.cluster_id for paper in papers))
    
    # Filter clusters based on preferences
    if included_clusters:
        unique_clusters = [cluster for cluster in unique_clusters if cluster in included_clusters]
    
    # If no new papers, return early
    if not pubs:
        return "No new publications this week.\n"

    print(f"\nTotal papers to process: {len(pubs)}")
    print(f"Number of clusters to check: {len(unique_clusters)}")

    # Create a dictionary to track the best topic for each publication
    pub_best_topic = {}
    pub_best_score = {}

    # First pass: find the best topic for each publication
    for i, cluster in enumerate(unique_clusters):
        matching_pubs = get_topic_matching_pubs(pubs, similarity_matrix, topic_num=cluster)
        print(f"\nCluster {cluster}:")
        print(f"Number of papers with similarity > 0.2: {len(matching_pubs)}")
        
        for pub in matching_pubs:
            pub_index = pubs.index(pub)
            similarity_score = similarity_matrix[cluster - 1, pub_index]
            
            # Debug: Print all scores for this paper
            print(f"\nPaper: {pub.title}")
            print(f"Found at index {pub_index} in pubs list")
            print(f"Current cluster {cluster} score: {similarity_score:.3f}")
            if pub in pub_best_score:
                print(f"Previous best score: {pub_best_score[pub]:.3f} with cluster {pub_best_topic[pub]}")
            
            # If this is the first time we've seen this pub or if this score is better
            if pub not in pub_best_score or similarity_score > pub_best_score[pub]:
                pub_best_score[pub] = similarity_score
                pub_best_topic[pub] = cluster
                print(f"Assigned to cluster {cluster} with score {similarity_score:.3f}")

    # Second pass: organize publications by their best topic
    topic_pubs = {topic: [] for topic in unique_clusters}
    print("\nOrganizing papers into topics:")
    for pub, topic in pub_best_topic.items():
        # Only include papers with a minimum similarity score
        if pub_best_score[pub] >= 0.2:  # Same threshold as get_topic_matching_pubs
            topic_pubs[topic].append(pub)
            print(f"Added to topic {topic}: {pub.title} (score: {pub_best_score[pub]:.3f})")
        else:
            print(f"Skipped due to low score: {pub.title} (score: {pub_best_score[pub]:.3f})")

    # Print summary of papers per topic
    print("\nSummary of papers per topic:")
    for topic in unique_clusters:
        print(f"Topic {topic}: {len(topic_pubs[topic])} papers")

    # Generate the email content
    email_message = "Your literature digest for this week\n"
    email_message += "---------------------------------\n\n"

    section_matches = "Here is a list of recent publications that match your interest.\n"
    section_matches += "---------------------------------\n\n"

    section_no_matches = "Here are your interests without a match this week.\n"
    section_no_matches += "---------------------------------\n\n"

    for cluster in unique_clusters:
        extracted_corpus = extract_cluster_docs(bow_corpus, papers, cluster_num=cluster)
        topic_keys = extract_lda_keywords(dictionary, extracted_corpus)
        matching_pubs = topic_pubs[cluster]  # Get publications that best match this topic
        
        if len(matching_pubs) != 0:
            # Format all topic keywords with their scores
            topic_desc = "*Because you've been reading about a topic with these keywords: ["
            for keyword, score in topic_keys:
                topic_desc += f"{keyword} ({score:.4f}), "
            topic_desc += "]\n"
            section_matches += topic_desc
            
            for pub in matching_pubs:
                if hasattr(pub, 'prism_doi'):
                    link = f"""https://doi.org/{pub.prism_doi}"""
                else:
                    link = pub.link
                # Debug: Print what score we're using
                print(f"\nDisplaying paper: {pub.title}")
                print(f"Using best score: {pub_best_score[pub]:.3f}")
                print(f"Matrix score for current cluster: {similarity_matrix[cluster - 1, pubs.index(pub)]:.3f}")
                
                similarity_score = pub_best_score[pub]
                section_matches += f"""- {pub.title} ({pub.prism_publicationname}) [Similarity: {similarity_score:.3f}] {link}\n"""
            section_matches += "- - - - - - - - - - - - - - - - - - - - - - - -\n\n"
        else:
            # Format all keywords for topics without matches
            topic_desc = "- Your topic of interest based on: ["
            for keyword, score in topic_keys:
                topic_desc += f"{keyword} ({score:.4f}), "
            topic_desc += "]\n"
            section_no_matches += topic_desc + "\n"
        
    email_message += section_matches + section_no_matches
    
    return email_message

def check_feeds(feeds):
    for feed in feeds:
        try:
            response = requests.get(feed)
            if response.status_code == 200:
                print(f"Feed is working: {feed}")
            else:
                print(f"Feed returned status code {response.status_code}: {feed}")
        except Exception as e:
            print(f"Failed to access feed {feed}: {e}")

def analyze_library_evolution(library_files, window_size=100, step_size=25, similarity_model=None):
    """
    Analyze the evolution of a library over time using multiple versions.
    
    Args:
        library_files: List of library file paths in chronological order
        window_size: Size of the rolling window
        step_size: Step size for the rolling window
        similarity_model: Word embedding model for similarity calculations
        
    Returns:
        ClusterTracker and RollingClusterAnalyzer objects
    """
    if similarity_model is None:
        raise ValueError("A similarity model must be provided for analysis")
        
    tracker = ClusterTracker()
    analyzer = RollingClusterAnalyzer(window_size=window_size, step_size=step_size)
    
    for file_path in library_files:
        # Load library
        library = Library()
        library.load(file_path)
        
        # Get timestamp from filename
        timestamp = os.path.basename(file_path)[:10]  # YYYY_MM_DD
        
        # Tokenize and create similarity matrix
        tokenized_library = tokenize_library(library)
        dictionary = corpora.Dictionary(tokenized_library)
        bow_corpus = [dictionary.doc2bow(text) for text in tokenized_library]
        tfidf_model = models.TfidfModel(bow_corpus)
        termsim_index = similarities.WordEmbeddingSimilarityIndex(similarity_model)
        termsim_matrix = similarities.SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf_model)
        similarity_matrix = get_cosine_matrix(termsim_matrix, bow_corpus, bow_corpus)
        
        # Cluster the library
        clusters = compute_topics(similarity_matrix, library.papers, method='complete')
        
        # Update tracker and analyzer
        tracker.update(timestamp, library.titles, clusters, similarity_matrix)
        analyzer.process_new_papers(library.titles, similarity_matrix)
    
    return tracker, analyzer

class ClusterTracker:
    def __init__(self):
        self.cluster_history = {}  # {timestamp: {cluster_id: [paper_ids]}}
        self.paper_cluster_history = {}  # {paper_id: [(timestamp, cluster_id)]}
        self.cluster_metrics = {}  # {timestamp: {cluster_id: metrics}}
        self.cluster_lineage = {}  # {cluster_id: [(timestamp, parent_cluster_id)]}
    
    def update(self, timestamp, papers, clusters, similarity_matrix):
        # Update cluster compositions
        current_clusters = {}
        for cluster_id in np.unique(clusters):
            paper_indices = np.where(clusters == cluster_id)[0]
            current_clusters[cluster_id] = paper_indices
        
        # Track cluster metrics
        metrics = {}
        for cluster_id, paper_indices in current_clusters.items():
            # Calculate cluster cohesion
            cluster_similarities = similarity_matrix[paper_indices][:, paper_indices]
            cohesion = np.mean(cluster_similarities)
            
            # Calculate cluster size
            size = len(paper_indices)
            
            # Calculate cluster stability
            if timestamp in self.cluster_history:
                stability = self._calculate_stability(cluster_id, paper_indices)
            else:
                stability = 1.0
                
            metrics[cluster_id] = {
                'size': size,
                'cohesion': cohesion,
                'stability': stability
            }
        
        # Update histories
        self.cluster_history[timestamp] = current_clusters
        self.cluster_metrics[timestamp] = metrics
        
        # Update paper histories - using paper titles as IDs
        for paper_idx, cluster_id in enumerate(clusters):
            paper_title = papers[paper_idx]  # papers is a list of strings
            if paper_title not in self.paper_cluster_history:
                self.paper_cluster_history[paper_title] = []
            self.paper_cluster_history[paper_title].append((timestamp, cluster_id))
    
    def _calculate_stability(self, cluster_id, current_papers):
        # Compare current cluster composition with previous state
        previous_state = self.cluster_history[max(self.cluster_history.keys())]
        if cluster_id in previous_state:
            previous_papers = set(previous_state[cluster_id])
            current_papers = set(current_papers)
            overlap = len(previous_papers.intersection(current_papers))
            stability = overlap / len(previous_papers)
            return stability
        return 0.0

class RollingClusterAnalyzer:
    def __init__(self, window_size=50, step_size=50):
        self.window_size = window_size
        self.step_size = step_size
        self.windows = []
        self.window_dates = []  # Store dates for each window
        self.sorted_papers = None  # Store sorted papers
    
    def create_sorted_library(self, papers_library, verbose=False):
        """
        Create a sorted library containing only papers with valid creation dates.
        
        Args:
            papers_library: Library object containing papers
            verbose: Whether to print detailed debugging information
            
        Returns:
            Library object containing sorted papers with valid creation dates
        """
        # Create new library object
        sorted_library = Library()
        
        # Filter and sort papers by creation date
        valid_papers = [p for p in papers_library.papers if p.date_created and not pd.isna(p.date_created)]
        sorted_library.papers = sorted(valid_papers, key=lambda x: pd.to_datetime(x.date_created))
        
        if verbose:
            print(f"\nCreated sorted library:")
            print(f"Original papers: {len(papers_library.papers)}")
            print(f"Papers with valid dates: {len(sorted_library.papers)}")
            print(f"Dropped papers: {len(papers_library.papers) - len(sorted_library.papers)}")
        
        return sorted_library
    
    def create_windows(self, papers_library, verbose=False):
        """
        Create windows for analysis based on when papers were added to the library.
        
        Args:
            papers_library: Library object containing papers
            verbose: Whether to print detailed debugging information
        """
        # Create sorted library
        sorted_library = self.create_sorted_library(papers_library, verbose)
        self.sorted_papers = sorted_library.papers
        
        # Create windows based on sorted papers
        self.windows = []
        self.window_dates = []
        
        # Calculate total number of complete windows
        n_complete_windows = (len(self.sorted_papers) - self.window_size) // self.step_size + 1
        
        # Create windows ensuring chronological order
        for i in range(n_complete_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            # Get all dates in this window
            window_dates = [pd.to_datetime(paper.date_created) for paper in self.sorted_papers[start_idx:end_idx]]
            
            # Verify window is chronological
            if all(window_dates[j] <= window_dates[j+1] for j in range(len(window_dates)-1)):
                self.windows.append((start_idx, end_idx))
                window_date = pd.to_datetime(self.sorted_papers[start_idx].date_created)
                self.window_dates.append(window_date)
            elif verbose:
                print(f"Warning: Window {i+1} is not strictly chronological, skipping")
        
        if verbose:
            print(f"\nCreated {len(self.windows)} windows")
            print("\nWindow details:")
            for i, (start_idx, end_idx) in enumerate(self.windows):
                print(f"\nWindow {i+1}:")
                print(f"  Papers {start_idx} to {end_idx-1} (size: {end_idx-start_idx})")
                print(f"  Start date: {self.sorted_papers[start_idx].date_created}")
                print(f"  End date: {self.sorted_papers[end_idx-1].date_created}")
    
    def compute_cluster_frequencies(self, papers, verbose=False):
        """
        Compute the frequency of each cluster's papers across windows.
        Each row (cluster) will sum to 100%, showing how that cluster's papers
        are distributed across time windows.
        
        Args:
            papers: List of Paper objects to analyze
            verbose: Whether to print detailed debugging information
            
        Returns:
            dict: Dictionary containing:
                - 'frequencies': 2D numpy array of shape (n_windows, n_clusters)
                - 'window_labels': List of window labels (dates)
                - 'cluster_ids': List of unique cluster IDs
                - 'window_median_dates': List of median dates for each window
        """
        if self.sorted_papers is None:
            raise ValueError("Must call create_windows before compute_cluster_frequencies")
            
        # Get unique cluster IDs, filtering out None values
        unique_clusters = np.unique([paper.cluster_id for paper in self.sorted_papers if paper.cluster_id is not None])
        if len(unique_clusters) == 0:
            raise ValueError("No papers have cluster IDs assigned")
            
        n_clusters = len(unique_clusters)
        n_windows = len(self.windows)
        
        # Initialize frequency matrix
        frequencies = np.zeros((n_windows, n_clusters))
        
        # First pass: count papers in each window for each cluster
        for i, (start_idx, end_idx) in enumerate(self.windows):
            # Get cluster IDs for this window
            window_papers = self.sorted_papers[start_idx:end_idx]
            
            # Count frequency of each cluster in this window
            for j, cluster_id in enumerate(unique_clusters):
                count = sum(1 for paper in window_papers if paper.cluster_id == cluster_id)
                frequencies[i, j] = count
        
        # Second pass: normalize by cluster size (sum of each column)
        cluster_sizes = np.sum(frequencies, axis=0)
        for j in range(n_clusters):
            if cluster_sizes[j] > 0:  # Avoid division by zero
                frequencies[:, j] = frequencies[:, j] / cluster_sizes[j]
        
        # Create window labels using median dates
        window_median_dates = []
        for i, (start_idx, end_idx) in enumerate(self.windows):
            # Get all dates in this window
            window_dates = [pd.to_datetime(paper.date_created) for paper in self.sorted_papers[start_idx:end_idx]]
            
            # Convert to timestamps for median calculation
            timestamps = [d.timestamp() for d in window_dates]
            median_timestamp = np.median(timestamps)
            median_date = pd.to_datetime(median_timestamp, unit='s')
            window_median_dates.append(median_date.strftime('%Y-%m-%d'))
        
        return {
            'frequencies': frequencies,
            'window_labels': window_median_dates,
            'cluster_ids': unique_clusters,
            'window_median_dates': window_median_dates
        }
    
    def visualize_cluster_frequencies(self, papers, figsize=(15, 10), verbose=False):
        """
        Visualize cluster frequencies across windows as a heatmap.
        
        Args:
            papers: List of Paper objects to analyze
            figsize: Figure size for the plot
            verbose: Whether to print detailed debugging information
        """
        # Compute frequencies
        freq_results = self.compute_cluster_frequencies(papers, verbose)
        frequencies = freq_results['frequencies']
        window_labels = freq_results['window_median_dates']  # Use median dates
        cluster_ids = freq_results['cluster_ids']
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(frequencies.T,  # Transpose to show clusters on y-axis
                    cmap='viridis',
                    xticklabels=window_labels,  # Use median dates as x-labels
                    yticklabels=[f'Cluster {c}' for c in cluster_ids],
                    cbar_kws={'label': 'Proportion of Cluster Papers'},
                    annot=True,  # Show values in cells
                    fmt='.2%',   # Format as percentages with 2 decimal places
                    annot_kws={'size': 8})  # Adjust annotation font size
        
        plt.title('Cluster Distribution Over Time')
        plt.xlabel('Median Date of Papers in Window')
        plt.ylabel('Cluster')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()
        
        if verbose:
            # Print statistics
            print("\nCluster Statistics:")
            print(f"Total number of windows: {len(self.windows)}")
            print(f"Number of unique clusters: {len(cluster_ids)}")
            print(f"Average proportion per cluster: {np.mean(frequencies):.2f}")
            print(f"Maximum proportion in a cluster: {np.max(frequencies):.2f}")
            
            # Print detailed frequencies
            print("\nCluster Frequencies by Window:")
            for i, window in enumerate(window_labels):
                print(f"\nWindow {i+1} (Median Date: {window}):")
                for j, cluster_id in enumerate(cluster_ids):
                    if frequencies[i, j] > 0:  # Only show clusters with papers
                        print(f"  Cluster {cluster_id}: {frequencies[i, j]:.2%}")

    def debug_window_creation(self, papers_library, verbose=False):
        """
        Debug method to verify paper sorting and window creation.
        
        Args:
            papers_library: Library object containing papers
            verbose: Whether to print detailed debugging information
        """
        # Sort papers by their creation date in the library
        sorted_papers = sorted(papers_library.papers, 
                             key=lambda x: pd.to_datetime(x.date_created) if x.date_created else pd.Timestamp.max)
        
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

class ModelCache:
    def __init__(self):
        self.dictionary = None
        self.tfidf_model = None
        self.last_similarity_model = None
        self.dictionary_filter_params = {'no_below': 2, 'no_above': 0.5}
        self.tfidf_params = {'smartirs': 'ntc'}  # Normalized TF-IDF
    
    def get_models(self, tokenized_library, similarity_model):
        """Get or create models, reusing cached versions when possible."""
        # Only rebuild dictionary if it doesn't exist
        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(tokenized_library)
            # Filter dictionary to reduce size
            self.dictionary.filter_extremes(**self.dictionary_filter_params)
        
        # Create bow_corpus for current library
        bow_corpus = [self.dictionary.doc2bow(text) for text in tokenized_library]
        
        # Create TF-IDF model for current library with optimized parameters
        tfidf_model = models.TfidfModel(bow_corpus, **self.tfidf_params)
        
        # Create term similarity index with dictionary filtering
        termsim_index = similarities.WordEmbeddingSimilarityIndex(
            similarity_model,
            dictionary=self.dictionary  # Only use words in dictionary
        )
        
        # Create sparse term similarity matrix
        termsim_matrix = similarities.SparseTermSimilarityMatrix(
            termsim_index, 
            self.dictionary, 
            tfidf_model,
            symmetric=True,  # Enable symmetric computation
            nonzero_limit=100  # Limit number of non-zero elements per row
        )
        
        return self.dictionary, tfidf_model, termsim_matrix, bow_corpus

def find_optimal_clusters(similarity_matrix, min_clusters=2, max_clusters=20, method='average'):
    """
    Find the optimal number of clusters using silhouette scores.
    
    Args:
        similarity_matrix: Similarity matrix (dense)
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        method: Linkage method for hierarchical clustering
        
    Returns:
        Tuple of (optimal number of clusters, dictionary of silhouette scores)
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Ensure diagonal is zero
    np.fill_diagonal(distance_matrix, 0)

    # Convert to condensed form (upper triangular part as a vector)
    condensed_distance = squareform(distance_matrix, checks=False)
    
    # Try different numbers of clusters
    silhouette_scores = {}
    best_score = -1
    best_n = min_clusters
    
    for n in range(min_clusters, max_clusters + 1):
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distance, method=method, optimal_ordering=True)
        clusters = fcluster(linkage_matrix, n, criterion='maxclust')
        
        # Compute silhouette score
        score = silhouette_score(distance_matrix, clusters, metric='precomputed')
        silhouette_scores[n] = score
        
        # Update best score
        if score > best_score:
            best_score = score
            best_n = n
    
    print(f"\nOptimal number of clusters: {best_n} (score: {best_score:.3f})")
    
    return best_n, silhouette_scores

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

def load_word_model(model_path):
    """
    Load the word embedding model from the specified path.
    """
    model_file = os.path.join(model_path, 'pubmed_wordmodel.bin')
    if not os.path.exists(model_file):
        print("Downloading the Pubmed model...")
        os.makedirs(model_path, exist_ok=True)
        load_file_from_s3(origin_path='lang_processing/gensim-data/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin',
                           final_path=model_file)
    
    pubmed_wordmodel = models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    return pubmed_wordmodel

def load_or_create_models(papers_library, pubmed_wordmodel, model_path, overwrite=False):
    """
    Load existing models if they exist and overwrite is False, otherwise create new models.
    """
    # Define model file paths
    dictionary_path = os.path.join(model_path, 'dictionary.gensim')
    tfidf_path = os.path.join(model_path, 'model_tfidf.gensim')
    termsim_path = os.path.join(model_path, 'termsim_matrix.gensim')
    
    # Check if all models exist
    models_exist = all(os.path.exists(p) for p in [dictionary_path, tfidf_path, termsim_path])
    
    if models_exist and not overwrite:
        print("Loading existing models...")
        dictionary = corpora.Dictionary.load(dictionary_path)
        model_tfidf = models.TfidfModel.load(tfidf_path)
        termsim_matrix = similarities.SparseTermSimilarityMatrix.load(termsim_path)
        
        # Create bow_corpus for consistency
        tokenized_library = tokenize_library(papers_library)
        bow_corpus = [dictionary.doc2bow(text) for text in tokenized_library]
        
        return dictionary, bow_corpus, model_tfidf, termsim_matrix
    
    print("Creating new models...")
    # Tokenize the library
    tokenized_library = tokenize_library(papers_library)
    
    # Create dictionary
    dictionary = corpora.Dictionary(tokenized_library)
    dictionary.filter_tokens(bad_ids=[tokenid for tokenid, term in dictionary.items() if term not in pubmed_wordmodel])
    
    # Create bag-of-words corpus
    bow_corpus = [dictionary.doc2bow(text) for text in tokenized_library]
    
    # Create TF-IDF model
    model_tfidf = models.TfidfModel(bow_corpus)
    
    # Create term similarity matrix
    termsim_index = similarities.WordEmbeddingSimilarityIndex(pubmed_wordmodel)
    termsim_matrix = similarities.SparseTermSimilarityMatrix(source=termsim_index, dictionary=dictionary, tfidf=model_tfidf)
    
    return dictionary, bow_corpus, model_tfidf, termsim_matrix

def save_models(dictionary, model_tfidf, termsim_matrix, model_path, papers_library):
    """
    Save the models and analysis results to the specified directory.
    
    Args:
        dictionary: Gensim dictionary
        model_tfidf: TF-IDF model
        termsim_matrix: Term similarity matrix
        model_path: Directory to save models
        papers_library: Library object containing papers with cluster assignments
    """
    # Create output directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save models
    dictionary.save(os.path.join(model_path, 'dictionary.gensim'))
    model_tfidf.save(os.path.join(model_path, 'model_tfidf.gensim'))
    termsim_matrix.save(os.path.join(model_path, 'termsim_matrix.gensim'))
    
    # Save papers library with cluster assignments
    papers_data = []
    for paper in papers_library.papers:
        # Helper function to safely convert values
        def safe_convert(value):
            if pd.isna(value) or value is None:
                return None
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        
        paper_dict = {
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': paper.authors,
            'date_created': paper.date_created,
            'date_updated': paper.date_updated,
            'read': paper.read,
            'notes': paper.notes,
            'ratings': paper.ratings,
            'doi': paper.doi,
            'pmid': paper.pmid,
            'pmcid': paper.pmcid,
            'year': safe_convert(paper.year),
            'journal': paper.journal,
            'date': paper.date,
            'pages': paper.pages,
            'issue': paper.issue,
            'volume': paper.volume,
            'cluster_id': safe_convert(paper.cluster_id),
            'silhouette_score': float(paper.silhouette_score) if paper.silhouette_score is not None else None
        }
        papers_data.append(paper_dict)
    
    # Save as CSV
    df = pd.DataFrame(papers_data)
    df.to_csv(os.path.join(model_path, 'papers_library_with_clusters.csv'), index=False)
    
    # Save as JSON
    import json
    with open(os.path.join(model_path, 'papers_library_with_clusters.json'), 'w') as f:
        json.dump(papers_data, f, indent=2)
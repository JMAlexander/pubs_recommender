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

class Library:
    def __init__(self):
        self.titles = None
        self.abstract = None
        self.authors = None

    def load(self, file):
        lib = pd.read_csv(file)
        self.titles = list(lib['title'])
        self.abstracts = list(lib['abstract'])
        #self.authors = list(lib['authors'])

def get_stop(stop_method='nltk'):
    if (stop_method == 'nltk'):
        stop_words = set(nltk.corpus.stopwords.words('english'))

    return stop_words
        

def tokenize_library(library, stop_method='nltk', type='abstracts'):
    ##Get stopword corpus
    stop_words = get_stop(stop_method=stop_method)
        
    tokenized_texts = []
    if (type == 'abstracts'):
        for text in library.abstracts:
            if not(isinstance(text, float) and np.isnan(text)): 
                tokenized_texts.append(text.lower().split())
    elif (type == 'title'):
        for text in library.titles:
            if not(isinstance(text, float) and np.isnan(text)): 
                tokenized_texts.append(text.lower().split())
    tokenized_texts =[[word for word in doc if word not in stop_words] for doc in tokenized_texts]
    
    return tokenized_texts

def tokenize_feeds(feeds, stop_method='nltk', type='title'):
    ##Get stopword corpus
    stop_words = get_stop(stop_method=stop_method)
    
    pubs = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            #if (type == 'title'):
            #    pubs.append(entry.title)
            #elif (type == 'abstract'):
            #    pubs.append(entry.abstract)
            pubs.append(entry)

    if (type == 'title'):
        tokenized_texts = [text.lower().split() for text in pubs.title]
    elif (type == 'abstract'):
        tokenized_text = [text.lower().split() for text in pubs.abstract]
    tokenized_texts =[[word for word in doc if word not in stop_words] for doc in tokenized_texts]
    
    return pubs, tokenized_texts

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

def cluster_library(library, similarity_model, num_clusters=10):
    tokenized_library = tokenize_library(library)
    dictionary = corpora.Dictionary(tokenized_library)

    # Generate a BoWs
    bow_corpus =  [dictionary.doc2bow(text) for text in tokenized_library]
    tfidf_model = models.Tfidf(bow_corpus)
    termsim_index = similarities.WordEmbeddingSimilarityIndex(similarity_model)
    termsim_matrix = similarities.SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf_model)


def get_cosine_matrix(termsim_matrix, bow_i, bow_j, normalized_value=(True, True)):
    # Initialize matrix
    matrix_size = (len(bow_i), len(bow_j))
    cosine_similarities = np.zeros(matrix_size)

    # Compute cosine similarities for each value
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            cosine_similarities[i,j] = termsim_matrix.inner_product(bow_i[i], bow_j[j], normalized=normalized_value)

    return cosine_similarities

def compute_topics(cosine_matrix, num_topics):
    # Cluster papers based on soft cosine distance
    distance_matrix = 1 - cosine_matrix
    linkage_matrix = linkage(distance_matrix, method = 'average', optimal_ordering=True)
    topic_labels = fcluster(linkage_matrix, num_topics, criterion='maxclust')

    return topic_labels

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


    """
    #Train the LDA model
    lda = models.LdaModel(bow_corpus, num_topics=1, id2word=dictionary, passes=10)
    #keywords = [lda.get_topic_terms(id, topn=10)[0] for id in topic_ids]
    topic_terms = lda.get_topic_terms(topicid=0, topn=2)
    keyword_ids = list(x[0] for x in topic_terms) ##Gaves list of keyword ids
    print(keyword_ids)
    keywords = [dictionary[id] for id in keyword_ids]

    return keywords
    

def send_email(message, to_address):
    # Email account details
    SMTP_SERVER = "smtp.sendgrid.net"
    SMTP_PORT = 587
    SENDGRID_API_KEY = os.getenv('SG_API_KEY')  # Replace with your actual API key
    EMAIL_ADDRESS = "bot@phillygenome.xyz"

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
           
def extract_cluster_docs(bow_corpus, topic_labels, cluster_num):
    cluster_indices = np.where(topic_labels == cluster_num)
    cluster_corpus = bow_corpus[cluster_indices]
    
    return cluster_corpus


def get_topic_matching_pubs(similarity_matrix, topic_num, threshold_similarity = 0.2):
    topic_pub_indices = np.where(similarity_matrix[topic_num - 1] > threshold_similarity)[0]
    topic_pub_indices = np.sort(topic_pub_indices)[::-1]
    matching_pubs = [pubs[idx] for idx in topic_pub_indices]

    return matching_pubs
    
import pandas as pd
import os
import boto3
import json
from gensim import corpora, models, similarities
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from dotenv import load_dotenv

load_dotenv('/home/jeff/data_server_home/DS_projects/literature_analysis/.env')

class Paper:
    """
    Represents a single paper in the library.
    """
    def __init__(self, title=None, abstract=None, authors=None, date_created=None, date_updated=None, read=False, notes=None, ratings=None, doi=None, pmid=None, pmcid=None, year=None, journal=None, date=None, pages=None, issue=None, volume=None, cluster_id=None, silhouette_score=None):
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.date_created = date_created
        self.date_updated = date_updated
        self.read = read
        self.notes = notes
        self.ratings = ratings
        self.doi = doi
        self.pmid = pmid
        self.pmcid = pmcid
        self.year = year
        self.journal = journal
        self.date = date
        self.pages = pages
        self.issue = issue
        self.volume = volume
        self.cluster_id = cluster_id
        self.silhouette_score = silhouette_score

    def to_dict(self):
        return self.__dict__

    def load_from_row(self, row):
        """Load paper data from a pandas row."""
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
    """
    Represents a collection of papers.
    """
    def __init__(self):
        self.papers = []

    def add_paper(self, paper):
        self.papers.append(paper)

    def to_dataframe(self):
        return pd.DataFrame([p.to_dict() for p in self.papers])

    def load(self, file):
        """Load library from CSV file."""
        lib = pd.read_csv(file)
        for _, row in lib.iterrows():
            paper = Paper()
            paper.load_from_row(row)
            self.papers.append(paper)

class ModelCache:
    """
    Cache for models to avoid rebuilding them unnecessarily.
    """
    def __init__(self):
        self.dictionary = None
        self.tfidf_model = None
        self.last_similarity_model = None
        self.dictionary_filter_params = {'no_below': 2, 'no_above': 0.5}
        self.tfidf_params = {'smartirs': 'ntc'}  # Normalized TF-IDF
    
    def get_models(self, tokenized_library, similarity_model):
        """
        Get or create models, reusing cached versions when possible.
        
        Args:
            tokenized_library: List of tokenized documents
            similarity_model: Word embedding model
            
        Returns:
            tuple: (dictionary, tfidf_model, termsim_matrix, bow_corpus)
        """
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

def load_word_model(model_path):
    """
    Load the word embedding model from the specified path.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Word embedding model
    """
    model_file = os.path.join(model_path, 'pubmed_wordmodel.bin')
    if not os.path.exists(model_file):
        print("Downloading the Pubmed model...")
        os.makedirs(model_path, exist_ok=True)
        load_file_from_s3(origin_path='lang_processing/gensim-data/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin',
                           final_path=model_file)
    
    pubmed_wordmodel = models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    return pubmed_wordmodel

def load_file_from_s3(origin_path, final_path, endpoint_url='https://nyc3.digitaloceanspaces.com', bucket='phillygenome-space'):
    """
    Load a file from S3 if it doesn't exist locally.
    
    Args:
        origin_path: S3 path to the file
        final_path: Local path to save the file
        endpoint_url: S3 endpoint URL
        bucket: S3 bucket name
    """
    if not os.path.exists(final_path):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        os.makedirs(models_dir, exist_ok=True)

    # Get keys
    s3_access_key = os.getenv('DO_ACCESS_KEY')
    s3_secret_key = os.getenv('DO_SECRET_KEY')

    s3_client = boto3.client(
                             service_name = 's3',
                             endpoint_url=endpoint_url,
                             aws_access_key_id=s3_access_key,
                             aws_secret_access_key=s3_secret_key
    )

    s3_client.download_file(bucket, origin_path, final_path)

def load_existing_models_for_feeds(model_dir):
    """
    Load pre-existing models for feed analysis (used by feed_analysis.py).
    
    Args:
        model_dir: Directory containing the models
        
    Returns:
        tuple: (dictionary, termsim_index, pubmed_wordmodel)
    """
    dictionary = corpora.Dictionary.load(os.path.join(model_dir, 'dictionary.gensim'))

    # Load the Pubmed model using the helper function
    print("Loading the Pubmed model and building the similarity index...")
    pubmed_wordmodel = load_word_model(model_dir)
    termsim_index = WordEmbeddingSimilarityIndex(pubmed_wordmodel)

    return dictionary, termsim_index, pubmed_wordmodel

def create_tfidf_and_similarity_matrix(dictionary, combined_corpus, termsim_index):
    """
    Create TF-IDF model and term similarity matrix from combined corpus (used by feed_analysis.py).
    
    Args:
        dictionary: Gensim dictionary
        combined_corpus: BOW corpus combining library and feed papers
        termsim_index: Term similarity index
        
    Returns:
        tuple: (model_tfidf, termsim_matrix)
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

def load_or_create_models_for_clustering(papers_library, pubmed_wordmodel, model_path, overwrite=False):
    """
    Load existing models if they exist and overwrite is False, otherwise create new models for clustering analysis.
    
    Args:
        papers_library: Library object
        pubmed_wordmodel: Word embedding model
        model_path: Path to model directory
        overwrite: Whether to overwrite existing models
        
    Returns:
        tuple: (dictionary, bow_corpus, model_tfidf, termsim_matrix)
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
        from .utils import tokenize_library
        tokenized_library = tokenize_library(papers_library)
        bow_corpus = [dictionary.doc2bow(text) for text in tokenized_library]
        
        return dictionary, bow_corpus, model_tfidf, termsim_matrix
    
    print("Creating new models...")
    # Tokenize the library
    from .utils import tokenize_library
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

def save_models(dictionary, model_tfidf, termsim_matrix, model_path):
    """
    Save the models to the specified directory.
    
    Args:
        dictionary: Gensim dictionary
        model_tfidf: TF-IDF model
        termsim_matrix: Term similarity matrix
        model_path: Directory to save models
    """
    # Create output directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save models
    dictionary.save(os.path.join(model_path, 'dictionary.gensim'))
    model_tfidf.save(os.path.join(model_path, 'model_tfidf.gensim'))
    termsim_matrix.save(os.path.join(model_path, 'termsim_matrix.gensim'))
    
    print(f"Models saved to {model_path}")

def save_papers_library(papers_library, output_dir):
    """
    Save the papers library with cluster assignments to the specified output directory.
    
    Args:
        papers_library: Library object containing papers with cluster assignments
        output_dir: Directory to save the papers library
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    df.to_csv(os.path.join(output_dir, 'papers_library_with_clusters.csv'), index=False)
    
    # Save as JSON
    with open(os.path.join(output_dir, 'papers_library_with_clusters.json'), 'w') as f:
        json.dump(papers_data, f, indent=2)
    
    print(f"Papers library saved to {output_dir}")

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
    library = Library()
    
    # Load JSON data
    with open(library_path, 'r') as f:
        papers_data = json.load(f)
    
    # Create Paper objects from JSON data
    for paper_dict in papers_data:
        paper = Paper()
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
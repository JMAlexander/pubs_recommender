import re
import pandas as pd
from datetime import datetime
import os
import glob
import numpy as np
import nltk
import html
import string

def clean_text(text):
    """
    Basic text cleaning: lowercasing, removing extra whitespace, etc.
    """
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_date(date_str, fmt='%Y-%m-%d'):
    """
    Parse a date string to a datetime object. Returns None if parsing fails.
    """
    try:
        return datetime.strptime(date_str, fmt)
    except Exception:
        return None

def safe_int(value):
    """
    Convert value to int if possible, else return None.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_float(value):
    """
    Convert value to float if possible, else return None.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def get_stop(stop_method='nltk'):
    """
    Get stopwords for text processing.
    
    Args:
        stop_method: Method for stopword retrieval ('nltk')
        
    Returns:
        Set of stopwords
    """
    if (stop_method == 'nltk'):
        stop_words = set(nltk.corpus.stopwords.words('english'))

    return stop_words

def filter_library(library, field='title'):
    """
    Filter library to only include papers with valid titles or abstracts.
    
    Args:
        library: Library object containing papers
        field: Field to check ('title' or 'abstract')
        
    Returns:
        Library object with only valid papers
    """
    filtered_library = type('Library', (), {'papers': []})()
    for paper in library.papers:
        if field == 'title':
            if not (isinstance(paper.title, float) and np.isnan(paper.title)):
                filtered_library.papers.append(paper)
        elif field == 'abstract':
            if not (isinstance(paper.abstract, float) and np.isnan(paper.abstract)):
                filtered_library.papers.append(paper)
    return filtered_library

def tokenize_library(library, stop_method='nltk', field='title', wordmodel=None):
    """
    Tokenize library papers.
    
    Args:
        library: Library object
        stop_method: Method for stopword removal
        field: Field to tokenize ('title' or 'abstract')
        wordmodel: Word embedding model for cleaning and splitting terms
        
    Returns:
        List of tokenized documents
    """
    stop_words = get_stop(stop_method=stop_method)
        
    tokenized_texts = []
    if (field == 'abstract'):
        for paper in library.papers:
            if not(isinstance(paper.abstract, float) and np.isnan(paper.abstract)):
                words = paper.abstract.lower().split()
                if wordmodel is not None:
                    cleaned_words = []
                    for word in words:
                        cleaned_words.extend(clean_and_split_term(word, wordmodel))
                    words = cleaned_words
                tokenized_texts.append([word for word in words if word not in stop_words])
    elif (field == 'title'):
        for paper in library.papers:
            if not(isinstance(paper.title, float) and np.isnan(paper.title)):
                words = paper.title.lower().split()
                if wordmodel is not None:
                    cleaned_words = []
                    for word in words:
                        cleaned_words.extend(clean_and_split_term(word, wordmodel))
                    words = cleaned_words
                tokenized_texts.append([word for word in words if word not in stop_words])
    return tokenized_texts

def get_recent_library(directory):
    """
    Find the most recent library file in the specified directory.
    
    Args:
        directory: Directory to search for library files
        
    Returns:
        Path to the most recent library file, or None if not found
    """
    today = datetime.today().date()
    library_paths = glob.glob(f"{directory}/????_??_??_papers_library.csv")
    
    if not library_paths:
        print(f"No library files found in {directory}")
        return None

    # Initialize variables for selecting most recent archive
    closest_archive = None
    min_diff = float("inf")
    for path in library_paths:
        try:
            archive = os.path.basename(path)
            date_str = archive[:10]  # First 10 characters (YYYY_MM_DD)
            archive_date = datetime.strptime(date_str, "%Y_%m_%d").date()
            # Compute days since the archive
            diff = abs((archive_date - today).days)

            if diff < min_diff:
                min_diff = diff
                closest_archive = path
        except ValueError:
            print(f"Warning: Skipping file with invalid date format: {path}")
            continue

    return closest_archive

def load_library(library_path):
    """
    Load a library from a CSV file.
    
    Args:
        library_path: Path to the library CSV file
        
    Returns:
        Library object, or None if loading fails
    """
    if not os.path.exists(library_path):
        print(f"Error: Library file not found: {library_path}")
        return None
    
    try:
        lib = pd.read_csv(library_path)
        from .models import Library, Paper
        library = Library()
        for _, row in lib.iterrows():
            paper = Paper()
            paper.load_from_row(row)
            library.papers.append(paper)
        return library
    except Exception as e:
        print(f"Error loading library: {str(e)}")
        return None

def clean_and_split_term(term, wordmodel):
    """
    Clean and split a term using a word embedding model.
    
    Args:
        term: Term to clean and split
        wordmodel: Word embedding model
        
    Returns:
        List of cleaned and split terms
    """
    if not isinstance(term, str):
        return []

    # Step 1: Shortcut — return original if already in model
    if term in wordmodel:
        return [term]

    # Step 2: Unescape HTML entities and normalize known punctuation
    term = html.unescape(term)
    replacements = {
        "'": "'", "'": "'", '"': '"', '"': '"',
        '–': '-', '—': '-', "'": "'", '•': '*',
        '\xa0': ' ',
    }
    for src, tgt in replacements.items():
        term = term.replace(src, tgt)

    # Remove HTML tags
    term = re.sub(r'<[^>]+>', '', term)

    # Lowercase and trim
    term = term.lower().strip(string.punctuation + string.whitespace + "\"'")

    # Step 3: Try stripping possessive 's and rechecking
    if term.endswith("'s") and term[:-2] in wordmodel:
        return [term[:-2]]
    elif term.endswith("'s") and term[:-2] in wordmodel:
        return [term[:-2]]

    # Step 4: If slash or hyphen present, try splitting
    if '/' in term and term not in wordmodel:
        parts = term.split('/')
    elif '-' in term and term not in wordmodel:
        parts = term.split('-')
    else:
        parts = term.split()

    # Step 5: Validate and return any model-covered subparts
    cleaned_parts = []
    for part in parts:
        part = part.strip(string.punctuation + string.whitespace + "\"'")
        if part and re.match(r"^[a-z0-9']+$", part):
            if part in wordmodel:
                cleaned_parts.append(part)

    return cleaned_parts

def load_cluster_preferences(config_dir):
    """
    Load cluster preferences from CSV file and return list of included cluster IDs.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        List of cluster IDs to include in search
    """
    preferences_file = os.path.join(config_dir, 'cluster_search_preferences.csv')
    if not os.path.exists(preferences_file):
        return []  # Return empty list if file doesn't exist
    preferences_df = pd.read_csv(preferences_file)
    return preferences_df[preferences_df['include_in_search'] == 'YES']['cluster_id'].tolist() 
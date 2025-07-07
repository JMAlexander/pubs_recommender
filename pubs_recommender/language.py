from gensim import models

def extract_tfidf_keywords(tfidf_model, dictionary, bow_doc, n=5):
    """
    Extracts the top `n` keywords from a BoW document using a trained TF-IDF model.

    Args:
        tfidf_model: Trained Gensim TfidfModel
        dictionary: Gensim Dictionary mapping words to IDs
        bow_doc: BoW representation of a document (list of (word_id, count) tuples)
        n: Number of top keywords to extract

    Returns:
        List of top `n` keywords for the document
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
    # Train the LDA model
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
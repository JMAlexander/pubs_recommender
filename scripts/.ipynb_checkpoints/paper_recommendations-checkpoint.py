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



##RSS Feeds
feeds = [
            'https://www.nature.com/nrd.rss',
            'https://www.nature.com/nbt.rss',
            'https://www.nature.com/nature.rss',
            'https://www.nature.com/ng.rss',
            'https://www.nature.com/nm.rss',
            'https://www.nature.com/ncb.rss',
            'https://www.nature.com/ncomms.rss',
            'https://www.nature.com/gt.rss',
            'https://www.nature.com/jhg.rss',
            'https://www.nature.com/nsmb.rss',
            'https://www.nature.com/nrd.rss',
            'https://www.cell.com/cell/inpress.rss',
            'https://www.cell.com/molecular-cell/inpress.rss',
            'https://www.science.org/action/showFeed?type=axatoc&feed=rss&jc=science',
            'https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=stm',
            'https://www.nejm.org/action/showFeed?jc=nejm&type=etoc&feed=rss',
            'https://eurjmedres.biomedcentral.com/articles/most-recent/rss.xml'
        ]

archive_directory = '/mnt/volume_nyc1_01/users/jeff/DS_projects/literature_analysis/data/'
papers_library = lang_helper.load_recent_library(archive_directory)


# Load the Pubmed model or download it if it doesn't exist locally
expected_model_location = '/mnt/volume_nyc1_01/users/jeff/DS_projects/literature_analysis/models/pubmed2018_w2v_200D.bin'
if (os.path.exists(expected_model_location)):
    print('Found it')
else:
    load_file_from_s3(origin_path='lang_processing/gensim-data/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin',
                       final_path=expected_model_location)
#with open("topic_indices.pkl", "rb") as file:
#        topic_indices = pickle.load(file)

tokenized_library = lang_helper.tokenize_library(papers_library)
pubs, tokenized_pubs = lang_helper.tokenize_feeds(feeds, type='title')

#Generate the dictionary to use for comparison
dictionary = corpora.Dictionary(tokenized_library)
dictionary.add_documents(tokenized_pubs)
print(len(dictionary))
dictionary.filter_tokens(bad_ids=[tokenid for tokenid, term in dictionary.items() if term not in pubmed_wordmodel]) # Filter the dictionary in place to include only terms present in pubmed_wordmodel
print(len(dictionary))


# Transform each document into a bag-of-words (BoW) format using the dictionary
bow_corpus_paperslibrary = [dictionary.doc2bow(text) for text in tokenized_library]
bow_corpus_rssfeed = [dictionary.doc2bow(text) for text in tokenized_pubs]
combined_corpus = bow_corpus_paperslibrary + bow_corpus_rssfeed

model_tfidf_paperslibrary = models.TfidfModel(bow_corpus_paperslibrary)
model_tfidf_rssfeed = models.TfidfModel(bow_corpus_rssfeed)
model_tfidf_combined = models.TfidfModel(combined_corpus)

termsim_index = WordEmbeddingSimilarityIndex(pubmed_wordmodel)
termsim_matrix = SparseTermSimilarityMatrix(source=termsim_index, dictionary=dictionary, tfidf=model_tfidf_combined)

library_pubs_similarity_matrix = get_cosine_matrix(bow_i = bow_corpus_paperslibrary, bow_j = bow_corpus_rssfeed)

if True: ##Update with condition
    library_self_similarity_matrix = get_cosine_matrix(termsim_matrix, bow_i = bow_corpus_paperslibrary, bow_j = bow_corpus_paperslibrary)
    topic_labels = compute_topics(library_self_similarity_matrix, num_topics=15)

topic_pubs_similarity_matrix = get_topic_similarity(library_pubs_similary_matrix, topic_labels)


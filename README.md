# Publication Recommender

A tool for analyzing scientific literature, tracking research trends, and providing personalized paper recommendations based on semantic similarity and topic clustering. The system analyzes your existing library of papers, identifies research clusters, and recommends new publications from RSS feeds that match your interests.

## Features

- **Semantic Analysis**: Uses PubMed word embeddings to understand paper content and relationships
- **Topic Clustering**: Automatically identifies research themes in your library
- **Trend Analysis**: Tracks the evolution of research topics over time
- **Smart Recommendations**: Filters RSS feeds to find papers matching your interests
- **Automated Reports**: Generates PDF reports with cluster analysis and visualizations
- **Email Notifications**: Sends personalized paper recommendations via email

## Project Structure

- `config/` - Configuration files including RSS feed URLs and cluster preferences
- `data/` - Storage for library data and analysis results
- `logs/` - Log files from analysis runs
- `models/` - Model files, embeddings, and cached analysis results
- `output/` - Generated reports and analysis output
- `scripts/` - Python source code

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   ```
   TARGET_EMAIL=your.email@example.com
   SENDGRID_API_KEY=your_sendgrid_api_key
   ```
4. Download required model files (see below)

## Required Model Files

The following model files are required and should be placed in the `models/` directory:

### Word Embeddings
- `pubmed2018_w2v_200D.bin` (2.0GB) - PubMed word embeddings
  - Source: [BioWordVec](https://github.com/ncbi-nlp/BioWordVec)
  - Direct download: [pubmed2018_w2v_200D.bin](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioWordVec/pubmed2018_w2v_200D.bin)

### Generated Model Files
The following files are automatically generated during analysis:

- `dictionary.gensim` - Gensim dictionary mapping words to IDs
- `model_tfidf.gensim` - TF-IDF model
- `termsim_matrix.gensim` - Term similarity matrix
- `papers_library_with_clusters.json` - Library data with cluster assignments

## Usage

### Library Analysis
Analyze your paper library and generate cluster reports:

```bash
python scripts/library_analysis.py [options]
```

Options:
- `--clustering_method`: Clustering method ('ward' or 'complete', default: 'ward')
- `--window_size`: Papers per analysis window (default: 50)
- `--step_size`: Window step size (default: 50)
- `--verbose`: Print detailed debugging information
- `--overwrite`: Overwrite existing models

### Feed Analysis
Process RSS feeds and get paper recommendations:

```bash
python scripts/feed_analysis.py [options]
```

Options:
- `--include-read`: Include already read items in recommendations
- `--target-email`: Email address for recommendations (overrides .env)

## Configuration

### RSS Feeds
Add RSS feed URLs to `config/feeds.txt`:
```
# Comments start with #
https://example.com/feed1.xml
https://example.com/feed2.xml
```

### Cluster Preferences
Manage cluster preferences in `config/cluster_search_preferences.csv`:
```csv
cluster_id,include_in_search
1,YES
2,NO
```

## Dependencies

- Python 3.8+
- gensim: For text analysis and topic modeling
- nltk: For text processing
- pandas: For data manipulation
- scipy: For clustering and similarity calculations
- reportlab: For PDF report generation
- matplotlib & seaborn: For visualizations
- sendgrid: For email notifications
- feedparser: For RSS feed processing
- biopython: For PubMed integration

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
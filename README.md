# Business News Processor

## Overview

This Streamlit app fetches and processes business news from multiple RSS feeds. It identifies relevant companies, tags alerts, analyzes sentiment, and extracts keywords from news articles. The app provides a user-friendly interface to view processed news items.

## Features

- Fetches news from 19 different business and technology RSS feeds
- Processes news to identify mentioned companies from a predefined list
- Tags news with relevant alerts from a comprehensive set of business and tech categories
- Performs sentiment analysis on news articles
- Extracts key words from each news item
- Displays processed news in an easy-to-read format

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Mountain311/business-news-processor.git
   cd business-news-processor
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the spaCy English model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run news_proc.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Click the "Fetch and Process News" button to start processing the latest news.

4. View the processed news items, including titles, companies mentioned, alerts, sentiment, and keywords.

## Customization

- To modify the list of tracked companies or alerts, edit the `companies` and `alerts` lists in the `news_proc.py` file.
- To add or remove RSS feeds, update the `rss_feed_urls` list in the `run_news_processor` function.

## Dependencies

- streamlit
- requests
- beautifulsoup4
- spacy
- nltk
- scikit-learn
- textblob
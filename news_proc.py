import streamlit as st
import requests
from bs4 import BeautifulSoup
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import time
from textblob import TextBlob
from typing import List, Dict, Any
import os
import nltk

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Initialize NLTK components
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Expanded list of companies to track
companies = [
    "Apple Inc.", "LinkedIn", "Tesla", "Microsoft", "Google", "Amazon", "Facebook",
    "IBM", "Intel", "Oracle", "Nvidia", "Adobe", "Salesforce", "Netflix", "Uber",
    "Airbnb", "PayPal", "Twitter", "Snapchat", "Spotify", "Zoom", "Slack",
    "Dropbox", "Square", "Shopify", "Twilio", "Atlassian", "Workday", "ServiceNow",
    "DocuSign", "Okta", "Palantir", "Snowflake", "Splunk", "Crowdstrike", "Cloudflare",
    "Datadog", "MongoDB", "Elastic", "Zendesk", "HubSpot", "Twilio", "Box",
    "Coupa", "Fastly", "Ping Identity", "Dynatrace", "New Relic", "PagerDuty",
    "Zuora", "Alteryx", "Anaplan", "Asana", "Bill.com", "Domo", "Smartsheet",
    "SolarWinds", "Sprout Social", "Sumo Logic", "Tufin", "Yext", "ZScaler"
]

# Expanded list of alerts
alerts = [
    "CXO News", "Cybersecurity", "Artificial Intelligence", "Finance", "Mergers and Acquisitions",
    "Earnings Report", "Product Launch", "Market Expansion", "Regulatory Changes", "Innovation",
    "Sustainability", "Talent Acquisition", "Digital Transformation", "Supply Chain",
    "Customer Experience", "Data Privacy", "Cloud Computing", "Blockchain", "Internet of Things",
    "5G Technology", "Renewable Energy", "E-commerce", "Remote Work", "Fintech",
    "Quantum Computing", "Augmented Reality", "Virtual Reality", "Robotics", "Autonomous Vehicles",
    "Space Technology", "Biotechnology", "Nanotechnology", "Cryptocurrency", "Machine Learning",
    "Edge Computing", "Big Data", "DevOps", "User Experience", "Mobile Technology",
    "Wearable Technology", "Smart Cities", "Green Technology", "3D Printing", "Drones",
    "Voice Technology", "Chatbots", "Natural Language Processing", "Computer Vision",
    "Predictive Analytics", "Personalization", "Biometrics", "Health Tech", "EdTech",
    "AgTech", "LegalTech", "PropTech", "InsurTech", "RegTech", "CleanTech",
    "Social Media Trends", "Influencer Marketing", "Content Marketing", "Growth Hacking"
]

# TF-IDF vectorizer for alert tagging
tfidf_vectorizer = TfidfVectorizer()
alert_vectors = tfidf_vectorizer.fit_transform(
    [" ".join(alert.lower().split()) for alert in alerts]
)

class NewsProcessor:
    def fetch_news(self, url: str) -> List[Dict[str, Any]]:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features="xml")
        news_items = soup.findAll("item")
        return [
            {
                "title": item.find("title").text if item.find("title") else "",
                "description": (
                    item.find("description").text if item.find("description") else ""
                ),
                "pub_date": item.find("pubDate").text if item.find("pubDate") else "",
                "link": item.find("link").text if item.find("link") else "",
            }
            for item in news_items
        ]

    def preprocess_text(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words and token.isalnum()
        ]
        return " ".join(tokens)

    def is_business_news(self, text: str) -> bool:
        doc = nlp(text)
        business_entities = [
            ent.text
            for ent in doc.ents
            if ent.label_ in ["ORG", "PRODUCT", "MONEY", "PERCENT"]
        ]
        business_keywords = [
            "revenue", "profit", "merger", "acquisition", "stock", "market", "industry",
            "economy", "startup", "investment", "venture capital", "IPO", "earnings",
            "financial", "CEO", "executives", "board", "shareholders", "dividend",
            "forecast", "growth", "downturn", "recession", "expansion", "quarterly",
            "fiscal", "technology", "innovation", "disruption", "partnership",
            "collaboration", "competition", "market share", "strategy", "valuation",
            "funding", "Series A", "Series B", "angel investor", "private equity",
            "hedge fund", "blockchain", "cryptocurrency", "artificial intelligence",
            "machine learning", "cloud computing", "SaaS", "e-commerce", "fintech",
            "biotech", "cleantech", "cybersecurity", "data analytics", "IoT",
            "augmented reality", "virtual reality", "5G", "quantum computing",
            "robotics", "autonomous vehicles", "space technology", "green energy"
        ]
        return len(business_entities) > 0 or any(
            keyword in text.lower() for keyword in business_keywords
        )

    def identify_companies(self, text: str) -> List[str]:
        doc = nlp(text)
        mentioned_companies = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                for company in companies:
                    if (
                        ent.text.lower() in company.lower()
                        or company.lower() in ent.text.lower()
                    ):
                        mentioned_companies.append(company)
        return list(set(mentioned_companies))

    def tag_alerts(self, text: str) -> List[str]:
        preprocessed_text = self.preprocess_text(text)
        text_vector = tfidf_vectorizer.transform([preprocessed_text])
        similarities = cosine_similarity(text_vector, alert_vectors)
        tagged_alerts = [
            alerts[i]
            for i in similarities.argsort()[0][::-1][:5]
            if similarities[0][i] > 0.1
        ]
        return tagged_alerts

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }

    def extract_keywords(self, text: str) -> List[str]:
        doc = nlp(text)
        keywords = [
            token.text
            for token in doc
            if not token.is_stop
            and token.is_alpha
            and token.pos_ in ["NOUN", "PROPN", "ADJ"]
        ]
        return list(set(keywords))[:20]  # Return top 20 unique keywords

    def process_news(self, news_item: Dict[str, str]) -> Dict[str, Any]:
        full_text = f"{news_item['title']} {news_item['description']}"

        if not self.is_business_news(full_text):
            return None

        relevant_companies = self.identify_companies(full_text)
        if not relevant_companies:
            return None

        tagged_alerts = self.tag_alerts(full_text)
        sentiment = self.analyze_sentiment(full_text)
        keywords = self.extract_keywords(full_text)

        return {
            "title": news_item["title"],
            "description": news_item["description"],
            "pub_date": news_item["pub_date"],
            "link": news_item["link"],
            "companies": relevant_companies,
            "alerts": tagged_alerts,
            "sentiment": sentiment,
            "keywords": keywords,
        }

def run_news_processor():
    processor = NewsProcessor()
    rss_feed_urls = [
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://techcrunch.com/feed/",
        "http://feeds.bbci.co.uk/news/business/rss.xml",
        "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
        "https://fortune.com/feed",
        "https://www.forbes.com/business/feed/",
        "https://www.entrepreneur.com/latest.rss",
        "https://feeds.feedburner.com/venturebeat/SZYF",
        "https://www.wired.com/feed/category/business/latest/rss",
        "https://www.businessinsider.com/rss",
        "https://www.fastcompany.com/rss",
        "https://hbr.org/rss/current",
        "https://www.inc.com/rss/",
        "https://www.eweek.com/feed/",
        "https://www.zdnet.com/news/rss.xml",
        "https://www.computerworld.com/index.rss",
        "https://www.cio.com/index.rss",
        "https://www.infoworld.com/index.rss"
    ]

    st.write(f"Fetching news at {datetime.now()}")
    news_items = []
    for url in rss_feed_urls:
        news_items.extend(processor.fetch_news(url))

    processed_news = []
    for item in news_items:
        processed_item = processor.process_news(item)
        if processed_item:
            processed_news.append(processed_item)

    return processed_news

def main():
    st.title("Business News Processor")
    
    if st.button("Fetch and Process News"):
        with st.spinner("Fetching and processing news..."):
            processed_news = run_news_processor()
        
        st.success(f"Processed {len(processed_news)} news items")
        
        for item in processed_news:
            st.subheader(item['title'])
            st.write(f"**Date:** {item['pub_date']}")
            st.write(f"**Link:** {item['link']}")
            st.write(f"**Companies:** {', '.join(item['companies'])}")
            st.write(f"**Alerts:** {', '.join(item['alerts'])}")
            st.write(f"**Sentiment:** Polarity {item['sentiment']['polarity']:.2f}, Subjectivity {item['sentiment']['subjectivity']:.2f}")
            st.write(f"**Keywords:** {', '.join(item['keywords'])}")
            st.write(f"**Description:** {item['description']}")
            st.markdown("---")

if __name__ == "__main__":
    main()

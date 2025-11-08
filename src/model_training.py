import pandas as pd
import numpy as np
import re
import string
import joblib
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# --- 1. Setup & Pre-download NLTK resources ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# --- 2. Preprocessing Functions (Identical to Notebook) ---

# Note: The original notebook included translation, which is skipped here for simplicity
# and dependency management, assuming the data is already in English after initial notebook run.
# The `preprocess_text` function is adjusted to run on new input.
def preprocess_text(text):
    """Cleans and tokenizes text for vectorization."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)

    # Remove @mentions and #hashtags
    text = re.sub(r'[@#]\w+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def get_sentiment_from_rating(rating):
    """Maps numerical rating to sentiment label."""
    if rating in [4, 5]:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif rating in [1, 2]:
        return "Negative"
    else:
        return "Unknown"

# --- 3. Data Loading and Preparation ---
try:
    # Assuming the Excel file is in the current directory.
    df = pd.read_excel('P597 DATASET.xlsx')
except FileNotFoundError:
    print("Error: 'P597 DATASET.xlsx' not found. Please ensure the file is in the current directory.")
    exit()

# Apply sentiment labeling and initial preprocessing
df['sentiment'] = df['rating'].apply(get_sentiment_from_rating)
df['title'] = df['title'].apply(preprocess_text)
df['body'] = df['body'].apply(preprocess_text)

# Define features and target
X = df[['title', 'body']]
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Shared Vocabulary Creation (Crucial for consistent vectorization) ---
# Combine text columns for building a single, shared vocabulary
X_train_combined = X_train['title'].fillna('') + " " + X_train['body'].fillna('')

# Fit the vectorizer on the combined training text to build the vocabulary
# This vectorizer is only used to extract the vocabulary, not for the final features
vocab_vectorizer = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    lowercase=True,
    ngram_range=(1, 2)
)
vocab_vectorizer.fit(X_train_combined)
shared_vocab = vocab_vectorizer.vocabulary_
print(f"Vocabulary size created: {len(shared_vocab)}")

# --- 5. Define Preprocessing Pipeline Steps (ColumnTransformer) ---
# Create the vectorizers using the SHAIRED vocabulary
preprocessor = ColumnTransformer(
    transformers=[
        # Title uses CountVectorizer (BoW)
        ('title_bow', CountVectorizer(vocabulary=shared_vocab), 'title'),
        # Body uses TfidfVectorizer (TF-IDF)
        ('body_tfidf', TfidfVectorizer(vocabulary=shared_vocab), 'body')
    ],
    remainder='passthrough'
)

# --- 6. Define and Train the Final Pipeline (Logistic Regression) ---
# Logistic Regression showed good performance in the notebook (0.7847)
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000))
])

print("Starting to train the final sentiment prediction pipeline...")
final_pipeline.fit(X_train, y_train)
print("Pipeline training complete.")

# --- 7. Save the Pipeline ---
model_filename = 'sentiment.joblib'
joblib.dump(final_pipeline, model_filename)
print(f"Trained model saved successfully as {model_filename}. Ready for Streamlit app.")
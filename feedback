pip install pandas scikit-learn nltk  


!pip install nltk
import nltk
nltk.download('punkt_tab')


pip install textblob



# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from textblob import TextBlob

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
file_path = "/content/product_review.csv"  # Update with your file path
df = pd.read_csv(file_path, encoding='latin-1')

# Data Preprocessing Function
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back to a string
    return ' '.join(tokens)

# Perform sentiment analysis using TextBlob
sentiments = []
for review in df['Review']:
    blob = TextBlob(review)
    sentiment = blob.sentiment.polarity  # Polarity: -1 (negative) to 1 (positive)
    if sentiment > 0:
        sentiments.append('Positive')
    elif sentiment < 0:
        sentiments.append('Negative')
    else:
        sentiments.append('Neutral')

df['TextBlob_Sentiment'] = sentiments
print(df[['Review', 'TextBlob_Sentiment']])
df['Cleaned_Review'] = df['Review'].apply(preprocess_text)

# Splitting the data
X = df['Cleaned_Review']  # Features (processed reviews)
y = df['TextBlob_Sentiment']  # Labels (positive, negative, neutral)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Model Training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test_tfidf)

# Step 7: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict overall sentiment for each product
def predict_product_sentiment():
    # Get all unique products
    products = df['Product'].unique()

    for product in products:
        product_reviews = df[df['Product'] == product]
        cleaned_reviews = product_reviews['Cleaned_Review']

        if cleaned_reviews.empty:
            print(f"No reviews found for {product}")
            continue

        # Transform the reviews using the trained TF-IDF vectorizer
        tfidf_reviews = vectorizer.transform(cleaned_reviews)

        # Predict sentiments for all reviews of the product
        sentiments = model.predict(tfidf_reviews)

        # Determine overall sentiment based on majority
        overall_sentiment = pd.Series(sentiments).value_counts().idxmax()
        print(f"Overall sentiment for {product}: {overall_sentiment}")
        print("Individual Sentiments:", list(sentiments))

# Example: Predict overall sentiment for all products
predict_product_sentiment()

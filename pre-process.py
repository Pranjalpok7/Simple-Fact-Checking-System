import requests
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data (if not already downloaded)
import nltk


# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Wikimedia API endpoint
WIKIMEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

def fetch_multiple_pages(query, limit=5):
    """
    Fetch content for multiple pages based on a search query using the Wikimedia API.
    """
    # Parameters for the search request
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,  # Search query
        "srlimit": limit,   # Limit the number of results
    }

    # Make the API request
    response = requests.get(WIKIMEDIA_API_URL, params=params)
    data = response.json()

    # Extract search results
    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return None

    # Fetch content for each page
    pages = []
    for result in search_results:
        page_title = result["title"]
        page_content = fetch_wikipedia_page(page_title)  # Fetch page content
        if page_content:
            # Preprocess the content
            cleaned_content = preprocess_text(page_content)
            pages.append({
                "title": page_title,
                "content": cleaned_content
            })
    return pages

def fetch_wikipedia_page(page_title):
    """
    Fetch the content of a Wikipedia page by title.
    """
    # Parameters for the page content request
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "extracts",  # Fetch page content
        "explaintext": True,  # Return plain text
    }

    # Make the API request
    response = requests.get(WIKIMEDIA_API_URL, params=params)
    data = response.json()

    # Extract page content
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_info in pages.items():
        return page_info.get("extract", "")  # Return the page content

def preprocess_text(text):
    """
    Preprocess the text by cleaning, tokenizing, removing stopwords, and lemmatizing.
    """
    # Step 1: Clean the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase

    # Step 2: Tokenize into sentences
    sentences = sent_tokenize(text)

    # Step 3: Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        # Lemmatize each word and remove stopwords
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        cleaned_sentences.append(" ".join(lemmatized_words))

    # Step 4: Join sentences into chunks (e.g., paragraphs)
    cleaned_text = " ".join(cleaned_sentences)
    return cleaned_text

# Example usage
query = "Machine Learning"
pages = fetch_multiple_pages(query)
if pages:
    for page in pages:
        print(f"Title: {page['title']}")
        print(f"Cleaned and lemmatized content snippet: {page['content'][:200]}...")  # Print first 200 characters
        print("-" * 50)
else:
    print("No pages found.")


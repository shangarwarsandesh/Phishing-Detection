import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
import tldextract
import re

# Load dataset
df = pd.read_csv("dataset_phishing.csv")

# Selecting features and target
FEATURE_COLUMNS = [
    "length_url", "length_hostname", "nb_dots", "nb_hyphens", "nb_at", "nb_qm",
    "nb_and", "nb_or", "nb_eq", "nb_underscore", "nb_tilde", "nb_percent",
    "nb_slash", "nb_star", "nb_colon", "nb_comma", "nb_semicolumn",
    "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash", "http_in_path",
    "https_token", "ratio_digits_url", "ratio_digits_host", "prefix_suffix",
    "random_domain", "shortening_service", "nb_subdomains", "phish_hints"
]
X = df[FEATURE_COLUMNS]
y = df["status"]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "phishing_model.pkl")

def extract_features(url):
    """Extracts features from a given URL."""
    parsed_url = urlparse(url)
    extracted_domain = tldextract.extract(url)
    
    features = {
        "length_url": len(url),
        "length_hostname": len(parsed_url.netloc),
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": url.count("@"),
        "nb_qm": url.count("?"),
        "nb_and": url.count("&"),
        "nb_or": url.count("|"),
        "nb_eq": url.count("="),
        "nb_underscore": url.count("_"),
        "nb_tilde": url.count("~"),
        "nb_percent": url.count("%"),
        "nb_slash": url.count("/"),
        "nb_star": url.count("*"),
        "nb_colon": url.count(":"),
        "nb_comma": url.count(","),
        "nb_semicolumn": url.count(";"),
        "nb_dollar": url.count("$"),
        "nb_space": url.count(" "),
        "nb_www": url.lower().count("www"),
        "nb_com": url.lower().count(".com"),
        "nb_dslash": url.count("//"),
        "http_in_path": 1 if "http" in parsed_url.path else 0,
        "https_token": 1 if "https" in extracted_domain.domain else 0,
        "ratio_digits_url": sum(c.isdigit() for c in url) / len(url),
        "ratio_digits_host": sum(c.isdigit() for c in parsed_url.netloc) / len(parsed_url.netloc),
        "prefix_suffix": 1 if "-" in extracted_domain.domain else 0,
        "random_domain": 1 if re.search(r"[0-9]{4,}", extracted_domain.domain) else 0,
        "shortening_service": 1 if re.search(r"bit\.ly|goo\.gl|tinyurl|ow\.ly", url) else 0,
        "nb_subdomains": extracted_domain.subdomain.count(".") + 1,
        "phish_hints": 1 if re.search(r"login|secure|account|banking", url, re.IGNORECASE) else 0,
    }
    
    return np.array([features[col] for col in FEATURE_COLUMNS])



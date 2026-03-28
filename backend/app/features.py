import re
import math
from collections import Counter

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account",
    "update", "confirm", "banking", "password", "credential",
    "wallet", "paypal", "apple", "microsoft", "google",
    "facebook", "instagram", "netflix", "amazon", "ebay"
]

def calculate_entropy(text):
    counter = Counter(text)
    length = len(text)
    entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())
    return entropy

def extract_features(url):
    features = {}
    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    
    ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    features["has_ip"] = 1 if re.search(ip_pattern, url) else 0
    features["has_https"] = 1 if url.startswith("https") else 0
    
    url_lower = url.lower()
    features["suspicious_words"] = sum(1 for word in SUSPICIOUS_WORDS if word in url_lower)
    features["url_entropy"] = calculate_entropy(url)
    
    return features

def url_to_vector(url):
    f = extract_features(url)
    return [
        f["url_length"],
        f["num_dots"],
        f["num_hyphens"],
        f["has_ip"],
        f["has_https"],
        f["suspicious_words"],
        f["url_entropy"]
    ]

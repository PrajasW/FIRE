"""
build_demo_indexes.py
---------------------
Builds three index variants (Boolean, TF, TF-IDF) on a small set of
embedded sample articles and exports them as JSON files that the
index viewer website can load.

Run:
    python build_demo_indexes.py
"""

import json, math, re, os, sys
from collections import defaultdict, Counter

# Add src to python path to allow importing preprocess_data
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing.preprocess_data import load_news_data, load_wikipedia_data, preprocess_text


# ── Embedded sample corpus (no external data needed) ────────────────────────
SAMPLE_ARTICLES = [
    {
        "title": "SpaceX Launches New Satellite Constellation",
        "text": "SpaceX successfully launched a new batch of satellites into orbit today. "
                "The Falcon 9 rocket lifted off from Cape Canaveral carrying 60 Starlink "
                "satellites. This mission marks the company's 15th launch this year. Elon Musk "
                "confirmed the deployment was successful. The satellites will provide broadband "
                "internet coverage to rural areas across the globe.",
        "published": "2024-03-15"
    },
    {
        "title": "Global Climate Summit Reaches Historic Agreement",
        "text": "World leaders gathered at the Global Climate Summit have reached a historic "
                "agreement to reduce carbon emissions by 50 percent by 2035. The agreement, "
                "signed by 195 countries, includes binding commitments for renewable energy "
                "investment and phasing out coal power plants. Environmental groups praised "
                "the deal as a turning point in the fight against climate change.",
        "published": "2024-06-22"
    },
    {
        "title": "Apple Unveils Revolutionary AI Chip",
        "text": "Apple has announced its latest M4 chip featuring advanced artificial "
                "intelligence capabilities. The new processor delivers 38 trillion operations "
                "per second for machine learning tasks. Apple CEO Tim Cook called it a "
                "breakthrough in personal computing. The chip will power the next generation "
                "of MacBook Pro and iPad Pro devices, enabling on-device AI processing.",
        "published": "2024-09-10"
    },
    {
        "title": "Breakthrough in Cancer Research Announced",
        "text": "Scientists at MIT have announced a major breakthrough in cancer treatment. "
                "A new immunotherapy approach has shown a 90 percent success rate in clinical "
                "trials for patients with advanced pancreatic cancer. The treatment uses "
                "engineered T-cells to target and destroy cancer cells while leaving healthy "
                "tissue intact. The research was published in the journal Nature Medicine.",
        "published": "2024-04-18"
    },
    {
        "title": "Stock Market Reaches Record Highs Amid Tech Rally",
        "text": "The stock market surged to record highs today driven by a massive rally "
                "in technology stocks. The S&P 500 gained 2.3 percent while the Nasdaq "
                "composite rose 3.1 percent. Major tech companies including Apple, Google, "
                "and Microsoft all posted significant gains. Analysts attribute the rally to "
                "strong earnings reports and optimism about artificial intelligence.",
        "published": "2024-07-08"
    },
    {
        "title": "New York City Announces Green Infrastructure Plan",
        "text": "New York City has unveiled a 10 billion dollar green infrastructure plan "
                "to combat flooding and improve air quality. The plan includes expanding "
                "parks, installing green roofs on public buildings, and creating new bike "
                "lanes throughout the city. Mayor Adams said the project will create 50,000 "
                "jobs and make New York the greenest city in America.",
        "published": "2024-02-28"
    },
    {
        "title": "Olympic Games: Athletes Break Multiple Records",
        "text": "The 2024 Olympic Games in Paris saw athletes shatter multiple world records "
                "across various sports. In swimming, a new world record was set in the 100m "
                "freestyle. Track and field events also produced remarkable performances with "
                "records falling in the 200m sprint and high jump. The games attracted a "
                "global audience of over 3 billion viewers.",
        "published": "2024-08-11"
    },
    {
        "title": "Electric Vehicle Sales Surge Globally",
        "text": "Global electric vehicle sales have surged by 45 percent compared to last year. "
                "Tesla remains the market leader, but Chinese manufacturers are rapidly gaining "
                "ground. Battery technology improvements have extended range to over 500 miles "
                "for premium models. Governments worldwide are offering subsidies and tax "
                "credits to encourage adoption of clean transportation.",
        "published": "2024-05-03"
    },
    {
        "title": "Artificial Intelligence Transforms Healthcare",
        "text": "Hospitals around the world are adopting artificial intelligence to improve "
                "patient outcomes. AI-powered diagnostic tools can now detect diseases from "
                "medical imaging with greater accuracy than human doctors. Machine learning "
                "algorithms analyze patient data to predict health risks and recommend "
                "personalized treatment plans. The technology is expected to save billions "
                "in healthcare costs.",
        "published": "2024-01-20"
    },
    {
        "title": "Major Earthquake Strikes Pacific Region",
        "text": "A major earthquake measuring 7.8 on the Richter scale struck the Pacific "
                "region early this morning. Tsunami warnings were issued for coastal areas "
                "across several countries. Emergency response teams have been deployed to "
                "affected areas. Initial reports indicate significant damage to infrastructure "
                "but fortunately no casualties have been reported so far.",
        "published": "2024-11-05"
    },
    {
        "title": "World Cup Qualification Heats Up",
        "text": "FIFA World Cup qualification matches produced dramatic results across all "
                "confederations. European teams dominated with Germany and Spain securing "
                "comfortable victories. South American qualifiers saw Brazil edge past "
                "Argentina in a thrilling encounter. Asian qualification saw Japan and "
                "South Korea confirm their places in the tournament.",
        "published": "2024-10-14"
    },
    {
        "title": "Quantum Computing Achieves New Milestone",
        "text": "Google's quantum computing team has achieved a significant milestone by "
                "demonstrating quantum error correction at scale. The new Willow processor "
                "can maintain quantum coherence for over 10 minutes, a dramatic improvement "
                "over previous systems. This breakthrough brings practical quantum computing "
                "closer to reality and could revolutionize fields from cryptography to drug "
                "discovery.",
        "published": "2024-12-09"
    },
]

# ── Minimal NLP preprocessing ──────────────────────────────────────────────
# We will use the preprocess_text function from src.preprocessing.preprocess_data
# which handles tokenization, stopwords, stemming, etc.

def tokenize(text):
    """Tokenize the pre-processed text by splitting on spaces."""
    return text.split()

# ── Index Builders ──────────────────────────────────────────────────────────

def build_boolean_index(articles):
    """x=1: Boolean index with doc IDs and position IDs."""
    inverted = defaultdict(dict)
    docs = {}
    for doc_id, art in enumerate(articles):
        docs[doc_id] = {"title": art["title"], "date": art["published"]}
        tokens = tokenize(art["text"])
        positions = defaultdict(list)
        for pos, tok in enumerate(tokens):
            positions[tok].append(pos)
        for term, pos_list in positions.items():
            inverted[term][doc_id] = pos_list
    return dict(inverted), docs

def build_tf_index(articles):
    """x=2: Ranked index with word counts (term frequency)."""
    inverted = defaultdict(dict)
    docs = {}
    for doc_id, art in enumerate(articles):
        docs[doc_id] = {"title": art["title"], "date": art["published"]}
        tokens = tokenize(art["text"])
        counts = Counter(tokens)
        for term, count in counts.items():
            inverted[term][doc_id] = count
    return dict(inverted), docs

def build_tfidf_index(articles):
    """x=3: TF-IDF scored index."""
    N = len(articles)
    inverted = defaultdict(dict)
    docs = {}
    doc_freq = Counter()

    for doc_id, art in enumerate(articles):
        docs[doc_id] = {"title": art["title"], "date": art["published"]}
        tokens = tokenize(art["text"])
        counts = Counter(tokens)
        for term in counts:
            doc_freq[term] += 1
        for term, count in counts.items():
            inverted[term][doc_id] = count  # raw TF for now

    # Second pass: convert raw TF to TF-IDF
    tfidf_index = defaultdict(dict)
    for term, postings in inverted.items():
        idf = math.log(N / doc_freq[term]) if doc_freq[term] > 0 else 0
        for doc_id, raw_tf in postings.items():
            tf_w = 1 + math.log(raw_tf) if raw_tf > 0 else 0
            tfidf_index[term][doc_id] = round(tf_w * idf, 4)
    return dict(tfidf_index), docs

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "index_viewer")
    os.makedirs(out_dir, exist_ok=True)

    print("Streaming Wikipedia data from HuggingFace...")
    from datasets import load_dataset
    try:
        # streaming=True means it only downloads enough to yield the first 200 items,
        # avoiding the huge 40+ GB download of the entire dataset.
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        # Take the first 200 articles
        articles = []
        for i, item in enumerate(ds):
            if i >= 200:
                break
            articles.append({
                "title": item.get("url", f"Wiki Document {i}"),
                "text": item.get("text", ""),
                "published": "2023-11-01"
            })
    except Exception as e:
        print(f"Error loading Wikipedia: {e}")
        articles = []
    
    print(f"Loaded {len(articles)} articles. Preprocessing...")
    
    processed_articles = []
    # Using tqdm if available, otherwise just loop
    try:
        from tqdm import tqdm
        iterator = tqdm(articles)
    except ImportError:
        iterator = articles
        
    for art in iterator:
        # Preprocess text and store it so tokenize() just needs to split it
        text = art.get('text', '')
        if text:
            # We store the output of NLTK preprocessing in the 'text' key
            # so the index builders can use it
            processed_articles.append({
                "title": art.get('title', 'Untitled'),
                "text": preprocess_text(text),
                "published": art.get('published', 'N/A')
            })

    print(f"Building indexes on {len(processed_articles)} preprocessed articles...")

    indexes = {}

    # Boolean (v1.1110T)
    inv, docs = build_boolean_index(processed_articles)
    indexes["v1_1110T"] = {
        "name": "SelfIndex-v1.1110T",
        "description": "Boolean index with positional postings (x=1, y=1, z=1, i=0, q=T)",
        "config": {"x": "1 -- Boolean (doc IDs + positions)",
                   "y": "1 -- Local custom (pickle/JSON)",
                   "z": "1 -- Simple (no compression)",
                   "i": "0 -- No skip pointers",
                   "q": "T -- Term-at-a-time"},
        "num_terms": len(inv),
        "num_docs": len(docs),
        "documents": {str(k): v for k, v in docs.items()},
        "inverted_index": {term: {str(did): val for did, val in postings.items()}
                          for term, postings in inv.items()},
    }
    print(f"  [OK] v1.1110T -- {len(inv)} terms across {len(docs)} docs")

    # TF (v1.2110D)
    inv, docs = build_tf_index(processed_articles)
    indexes["v1_2110D"] = {
        "name": "SelfIndex-v1.2110D",
        "description": "Ranked retrieval with raw term frequency (x=2, y=1, z=1, i=0, q=D)",
        "config": {"x": "2 -- Term frequency (word counts)",
                   "y": "1 -- Local custom (pickle/JSON)",
                   "z": "1 -- Simple (no compression)",
                   "i": "0 -- No skip pointers",
                   "q": "D -- Document-at-a-time"},
        "num_terms": len(inv),
        "num_docs": len(docs),
        "documents": {str(k): v for k, v in docs.items()},
        "inverted_index": {term: {str(did): val for did, val in postings.items()}
                          for term, postings in inv.items()},
    }
    print(f"  [OK] v1.2110D -- {len(inv)} terms across {len(docs)} docs")

    # TF-IDF (v1.3110D)
    inv, docs = build_tfidf_index(processed_articles)
    indexes["v1_3110D"] = {
        "name": "SelfIndex-v1.3110D",
        "description": "TF-IDF scored index with cosine similarity (x=3, y=1, z=1, i=0, q=D)",
        "config": {"x": "3 -- TF-IDF scores",
                   "y": "1 -- Local custom (pickle/JSON)",
                   "z": "1 -- Simple (no compression)",
                   "i": "0 -- No skip pointers",
                   "q": "D -- Document-at-a-time"},
        "num_terms": len(inv),
        "num_docs": len(docs),
        "documents": {str(k): v for k, v in docs.items()},
        "inverted_index": {term: {str(did): val for did, val in postings.items()}
                          for term, postings in inv.items()},
    }
    print(f"  [OK] v1.3110D -- {len(inv)} terms across {len(docs)} docs")

    # Write combined JSON
    out_path = os.path.join(out_dir, "index_data.js")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by build_demo_indexes.py\n")
        f.write("const INDEX_DATA = ")
        json.dump(indexes, f, indent=2)
        f.write(";\n")
    print(f"\n[DONE] Index data written to {out_path}")
    print(f"   Open index_viewer/index.html in your browser to explore!")

if __name__ == "__main__":
    main()

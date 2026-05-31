try:
    import streamlit as st  # type: ignore[import]
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("streamlit is required to run this app. Install it with: pip install streamlit") from e
try:
    import pandas as pd  # type: ignore[import]
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("pandas is required to run this app. Install it with: pip install pandas") from e
import re
import nltk
from nltk.corpus import stopwords, wordnet
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("scikit-learn is required to run this app. Install it with: pip install scikit-learn") from e
import os

# Download NLTK resources saat pertama kali run
@st.cache_resource
def setup_nltk():
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet'
    }
    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
setup_nltk()

class NetflixSearchEngine:
    def __init__(self, df):
        self.df = df
        self.df['title_clean'] = self.df['title'].fillna('').apply(self._clean_text)
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            analyzer='word',
            ngram_range=(1, 2)
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.df['title_clean'])
        st.session_state["engine_ready"] = True

    def _clean_text(self, text):
        text = str(text).lower()
        return re.sub(r'[^\w\s]', '', text)

    def expand_query(self, query):
        expanded = []
        for token in self._clean_text(query).split():
            expanded.append(token)
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    expanded.append(lemma.name().lower().replace('_', ' '))
        return ' '.join(set(expanded))

    def search(self, query, top_k=10):
        if not query.strip():
            return pd.DataFrame()
            
        expanded_query = self.expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        sorted_indices = similarities.argsort()[::-1]
        valid_mask = similarities[sorted_indices] > 0.0
        top_indices = sorted_indices[valid_mask][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.df.iloc[idx]
            results.append({
                'Peringkat': rank,
                'Relevansi': round(float(similarities[idx]), 4),
                'Jenis': row.get('type', 'N/A'),
                'Judul': row.get('title', 'N/A'),
                'Tahun Rilis': str(row.get('release_year', 'N/A')),
                'Durasi': str(row.get('duration', 'N/A'))
            })
        return pd.DataFrame(results)

@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file, encoding='utf-8', encoding_errors='ignore')
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="Netflix Search Engine", page_icon="🎬", layout="wide")
    st.title("🎬 Netflix Search Engine")
    st.markdown("Cari film/acara TV Netflix berdasarkan judul. **Tidak perlu ketik lengkap!**")

    # Upload File atau Gunakan Path Lokal
    uploaded_file = st.file_uploader("Upload file CSV dataset Anda:", type=["csv"])
    
    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    elif os.path.exists("Netflix_movies_and_tv_shows.csv"):
        df = load_data("Netflix_movies_and_tv_shows.csv")
    else:
        st.warning("⚠️ Silakan upload file `Netflix_movies_and_tv_shows.csv` atau letakkan di folder yang sama dengan `app.py`.")

    if df is not None and not df.empty:
        if "engine_ready" not in st.session_state:
            with st.spinner("⚙️ Membangun inverted index & model pencarian..."):
                engine = NetflixSearchEngine(df)
        else:
            engine = NetflixSearchEngine(df)

        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("🔍 Masukkan judul film/TV show:", placeholder="Contoh: stranger things, black mirror, the crown...")
        with col2:
            top_k = st.number_input("Jumlah hasil", min_value=1, max_value=50, value=10)

        if query.strip():
            results = engine.search(query, top_k=top_k)
            if not results.empty:
                st.success(f"✅ Ditemukan **{len(results)}** judul!")
                st.dataframe(
                    results,
                    column_config={
                        "Relevansi": st.column_config.ProgressColumn(
                            label="Skor Relevansi", min_value=0, max_value=1, format="%.3f"
                        ),
                        "Tahun Rilis": st.column_config.TextColumn(),
                        "Durasi": st.column_config.TextColumn(),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=450
                )
            else:
                st.warning("⚠️ Tidak ada hasil yang relevan. Coba kata kunci lain.")
        else:
            st.info("💡 Ketik judul di atas untuk memulai pencarian.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import re
import os
import nltk

# Explicitly ensure WordNet is downloaded in cloud environments
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NetflixSearchEngine:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['title_clean'] = self.df['title'].fillna('').apply(self._clean_text)
        
        # TfidfVectorizer internally builds the Inverted Index & TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            analyzer='word',
            ngram_range=(1, 2)
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.df['title_clean'])
        self.feature_names = self.vectorizer.get_feature_names_out()

    def _clean_text(self, text):
        text = str(text).lower()
        return re.sub(r'[^\w\s]', '', text)

    def _expand_query(self, query):
        """Safely expands query using WordNet with a fallback if corpus fails"""
        expanded_terms = []
        tokens = self._clean_text(query).split()
        
        for token in tokens:
            expanded_terms.append(token)
            try:
                # Safely access WordNet synonyms for nouns and adjectives
                for syn in wordnet.synsets(token, pos=(wordnet.NOUN, wordnet.ADJ)):
                    for lemma in syn.lemmas()[:2]:
                        expanded_terms.append(lemma.name().lower().replace('_', ' '))
            except Exception:
                # Graceful fallback if WordNet fails to load in cloud environment
                pass
                
        return ' '.join(list(set(expanded_terms)))

    def search(self, query, min_score=0.05, top_k=10):
        """Search with safe expansion and threshold filtering"""
        expanded_query = self._expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Filter & Rank
        mask = similarities >= min_score
        filtered_scores = similarities[mask]
        filtered_indices = np.where(mask)[0]
        
        if len(filtered_indices) == 0:
            return pd.DataFrame(columns=['Peringkat', 'Skor Kemiripan', 'Jenis', 'Judul', 'Tahun Rilis', 'Durasi'])
            
        # Sort descending
        sorted_indices = filtered_indices[filtered_scores.argsort()[::-1]][:top_k]
        
        results = []
        for i, idx in enumerate(sorted_indices, start=1):
            row = self.df.iloc[idx]
            results.append({
                'Peringkat': i,
                'Skor Kemiripan': round(float(similarities[idx]), 4),
                'Jenis': row['type'],
                'Judul': row['title'],
                'Tahun Rilis': str(row.get('release_year', 'N/A')),
                'Durasi': str(row.get('duration', 'N/A'))
            })
        return pd.DataFrame(results)

# Initialize once in cloud
@st.cache_resource
def init_engine(csv_path):
    return NetflixSearchEngine(csv_path)

def main():
    st.set_page_config(page_title="Netflix Search Pro", page_icon="🎬", layout="wide")
    st.title("🎬 Netflix Search Engine Pro")
    st.caption("Vector Space Search dengan Inverted Index & Query Expansion")

    csv_path = st.text_input("📂 Path File CSV:", value="Netflix_movies_and_tv_shows.csv")
    if not os.path.exists(csv_path):
        st.error("❌ File tidak ditemukan.")
        return

    with st.spinner("⏳ Memuat indeks..."):
        engine = init_engine(csv_path)

    st.sidebar.header("⚙️ Pengaturan")
    min_score = st.sidebar.slider("📊 Minimum Skor", 0.0, 1.0, 0.08, 0.01)
    top_k = st.sidebar.slider("📦 Jumlah Hasil", 1, 50, 10)

    st.divider()
    query = st.text_input("🔎 Masukkan kata kunci:", placeholder="Contoh: stranger things, black mirror...")
    
    if st.button("🚀 Cari Sekarang", type="primary", use_container_width=True) and query.strip():
        with st.spinner("🔎 Mencari..."):
            results = engine.search(query, min_score=min_score, top_k=top_k)
            
        if not results.empty:
            st.success(f"✅ Ditemukan **{len(results)}** judul!")
            st.dataframe(
                results,
                column_config={"Skor Kemiripan": st.column_config.NumberColumn(format="%.3f")},
                hide_index=True,
                use_container_width=True,
                height=450
            )
        else:
            st.warning("⚠️ Tidak ada hasil di atas batas skor. Coba turunkan slider atau ubah kata kunci.")
    elif not query.strip():
        st.info("💡 Masukkan kata kunci untuk memulai.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Download data NLTK yang diperlukan saat pertama kali jalan
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class NetflixSearchEngine:
    def __init__(self, df):
        self.df = df
        # Bersihkan kolom judul
        self.df['title_clean'] = self.df['title'].fillna('').apply(self._clean_text)
        
        # Inisialisasi Vectorizer (Membangun Inverted Index & VSM di balik layar)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            analyzer='word',
            ngram_range=(1, 2)
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.df['title_clean'])

    def _clean_text(self, text):
        text = str(text).lower()
        return re.sub(r'[^\w\s]', '', text)

    def _expand_query(self, query):
        """Query Expansion menggunakan WordNet"""
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        tokens = clean_query.split()
        expanded = []
        for token in tokens:
            expanded.append(token)
            try:
                # Ambil sinonim untuk Noun & Adjective
                for syn in wordnet.synsets(token, pos=(wordnet.NOUN, wordnet.ADJ)):
                    for lemma in syn.lemmas()[:2]:
                        expanded.append(lemma.name().lower())
            except Exception:
                pass
        return ' '.join(list(set(expanded)))

    def search(self, query, min_score=0.05, top_k=10):
        if not query.strip():
            return pd.DataFrame()
            
        expanded_query = self._expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Filter berdasarkan threshold
        mask = similarities >= min_score
        filtered_scores = similarities[mask]
        filtered_indices = np.where(mask)[0]
        
        if len(filtered_indices) == 0:
            return pd.DataFrame(columns=['Peringkat', 'Skor Similarity', 'Jenis', 'Judul', 'Tahun Rilis', 'Durasi'])
            
        # Urutkan dari skor tertinggi
        sorted_indices = filtered_indices[filtered_scores.argsort()[::-1]][:top_k]
        
        results = []
        for i, idx in enumerate(sorted_indices, start=1):
            row = self.df.iloc[idx]
            results.append({
                'Peringkat': i,
                'Skor Similarity': round(float(similarities[idx]), 4),
                'Jenis': row['type'],
                'Judul': row['title'],
                'Tahun Rilis': row['release_year'],
                'Durasi': row['duration']
            })
        return pd.DataFrame(results)

def main():
    st.set_page_config(page_title="Netflix Search Engine", page_icon="🎬", layout="wide")
    st.title("Netflix Search Engine Prototype")
    st.markdown("Temukan film atau acara TV Netflix dengan cepat. **Tidak perlu mengetik judul lengkap!**")
    
    # Opsi Upload File
    uploaded_file = st.file_uploader("Upload file CSV dataset Anda:", type=["csv"])
    
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8', encoding_errors='ignore')
    elif os.path.exists("Netflix_movies_and_tv_shows.csv"):
        df = pd.read_csv("Netflix_movies_and_tv_shows.csv", encoding='utf-8', encoding_errors='ignore')
    else:
        st.warning("⚠️ Silakan upload file `Netflix_movies_and_tv_shows.csv`")
        return

    if df is not None and not df.empty:
        # Inisialisasi Engine
        engine = NetflixSearchEngine(df)
        
        st.sidebar.header("⚙️ Pengaturan Pencarian")
        min_score = st.sidebar.slider("Minimum Skor Kemiripan", 0.0, 1.0, 0.08, 0.01)
        top_k = st.sidebar.slider("Jumlah Hasil Maksimal", 1, 50, 10)
        
        st.divider()
        query = st.text_input("Ketik judul film/TV show:", placeholder="Contoh: stranger things, black mirror...")
        
        if st.button("Cari Sekarang", type="primary") and query.strip():
            results = engine.search(query, min_score=min_score, top_k=top_k)
            
            if not results.empty:
                st.success(f"✅ Ditemukan **{len(results)}** judul yang relevan!")
                st.dataframe(
                    results,
                    column_config={
                        "Skor Similarity": st.column_config.NumberColumn(format="%.3f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=450
                )
            else:
                st.warning("⚠️ Tidak ada hasil dengan skor di atas batas minimum. Coba turunkan slider atau ubah kata kunci.")
        elif not query.strip():
            st.info("💡 Ketik judul di atas untuk memulai pencarian.")

if __name__ == "__main__":
    main()

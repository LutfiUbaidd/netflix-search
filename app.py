import streamlit as st
import pandas as pd
import re
import numpy as np
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NetflixSearchEngine:
    def __init__(self, df):
        self.df = df
        self.df['title_clean'] = self.df['title'].fillna('').apply(self._clean_text)
        
        # TfidfVectorizer secara internal membangun Inverted Index & VSM
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
        """Query Expansion terkontrol untuk mengurangi noise"""
        # Gunakan re.sub() untuk regex pada string biasa
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        tokens = clean_query.split()
        
        expanded = []
        for token in tokens:
            expanded.append(token)
            syn_added = 0
            for syn in wordnet.synsets(token, pos=(wordnet.NOUN, wordnet.ADJ)):
                for lemma in syn.lemmas()[:2]:
                    expanded.append(lemma.name().lower())
                    syn_added += 1
                if syn_added >= 2: break
        return ' '.join(list(set(expanded)))

    def search(self, query, min_score=0.1, top_k=10):
        if not query.strip():
            return pd.DataFrame()
            
        expanded_query = self._expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Filter berdasarkan threshold skor yang ditentukan user
        mask = similarities >= min_score
        filtered_scores = similarities[mask]
        filtered_indices = np.where(mask)[0]
        
        # Urutkan skor tertinggi
        top_indices = filtered_indices[filtered_scores.argsort()[::-1]][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices, start=1):
            row = self.df.iloc[idx]
            results.append({
                'Peringkat': i,
                'Skor Kemiripan': round(float(similarities[idx]), 4),
                'Jenis': row['type'],
                'Judul': row['title'],
                'Tahun Rilis': row['release_year'],
                'Durasi': row['duration']
            })
        return pd.DataFrame(results) if results else pd.DataFrame()

# ⚙️ Caching untuk performa cloud
@st.cache_resource
def init_engine(df):
    return NetflixSearchEngine(df)

# 🖥️ Antarmuka Streamlit
def main():
    st.set_page_config(page_title="Netflix Search Pro", page_icon="🎬", layout="wide")
    st.title("🎬 Netflix Search Engine Pro")
    st.caption("Mesin pencari berbasis Inverted Index, TF-IDF + Cosine Similarity, & Query Expansion terkontrol.")

    uploaded_file = st.file_uploader("Upload dataset Netflix CSV:", type=["csv"])
    
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8', encoding_errors='ignore')
    else:
        st.warning("⚠️ Silakan upload file `Netflix_movies_and_tv_shows.csv`")
        return

    if df is not None and not df.empty:
        engine = init_engine(df)
        
        st.sidebar.header("⚙️ Pengaturan Pencarian")
        min_score = st.sidebar.slider("Minimum Skor Kemiripan", 0.0, 1.0, 0.12, 0.01, 
                                      help="Naikkan slider untuk hasil lebih presisi.")
        top_k = st.sidebar.slider("Jumlah Hasil Maksimal", 1, 50, 15)
        
        st.divider()
        query = st.text_input("🔎 Masukkan judul film/TV show:", placeholder="Contoh: stranger things, black mirror, the crown...")
        
        if st.button("🔎 Cari Sekarang", type="primary") and query.strip():
            results = engine.search(query, min_score=min_score, top_k=top_k)
            
            if not results.empty:
                st.success(f"✅ Ditemukan **{len(results)}** judul relevan!")
                st.dataframe(
                    results,
                    column_config={
                        "Skor Kemiripan": st.column_config.NumberColumn(format="%.3f", help="Semakin dekat ke 1.0, semakin relevan"),
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

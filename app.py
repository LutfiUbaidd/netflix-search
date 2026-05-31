import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Unduh resource NLTK sekali saat startup
@st.cache_resource
def download_nltk():
    for pkg in ['punkt', 'punkt_tab', 'wordnet', 'stopwords']:
        nltk.download(pkg, quiet=True)
download_nltk()

class NetflixSearchEngine:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # Bersihkan data & gabungkan kolom yang akan diindeks
        self.df['search_text'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['listed_in'].fillna('') + ' ' +
            self.df['description'].fillna('')
        ).str.lower().str.replace(r'[^\w\s]', '', regex=True)
        
        # TfidfVectorizer secara internal membangun Inverted Index + VSM
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,       # Abaikan kata yang muncul di >85% dokumen
            min_df=2           # Abaikan kata yang muncul di <2 dokumen
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.df['search_text'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_loaded = True

    def _expand_query(self, query, max_synonyms=2):
        """Query Expansion terkontrol untuk mengurangi noise"""
        expanded = set()
        tokens = query.lower().replace(r'[^\w\s]', '', regex=True).split()
        for token in tokens:
            expanded.add(token)
            # Ambil sinonim hanya untuk kata benda & adjektiva
            for syn in wordnet.synsets(token, pos=(wordnet.NOUN, wordnet.ADJ))[:max_synonyms]:
                for lemma in syn.lemmas()[:2]:
                    expanded.add(lemma.name().lower().replace('_', ' '))
        return ' '.join(expanded)

    def search(self, query, min_score=0.1, top_k=10):
        if not query.strip():
            return pd.DataFrame()
            
        # Ekspansi query
        expanded_query = self._expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])
        
        # Hitung Cosine Similarity
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Filter & Peringkat
        mask = similarities >= min_score
        filtered_scores = similarities[mask]
        filtered_indices = np.where(mask)[0]
        
        # Urutkan skor tertinggi
        top_indices = filtered_indices[filtered_scores.argsort()[::-1]][:top_k]
        
        # Format output
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
def init_engine(path):
    return NetflixSearchEngine(path)

# 🖥️ Antarmuka Streamlit
def main():
    st.set_page_config(page_title="🎬 Netflix Search Pro", page_icon="🎬", layout="wide")
    st.title("🎬 Netflix Search Engine Pro")
    st.caption("Mesin pencari berbasis Inverted Index, TF-IDF + Cosine Similarity, & Query Expansion terkontrol.")

    # Input path
    csv_path = st.text_input("📂 Path File CSV:", value="Netflix_movies_and_tv_shows.csv")
    if not os.path.exists(csv_path):
        st.error("❌ File tidak ditemukan. Pastikan path benar atau upload file ke folder yang sama.")
        return

    # Inisialisasi engine
    engine = init_engine(csv_path)

    # Kontrol Pencarian
    st.sidebar.header("⚙️ Pengaturan Pencarian")
    min_score = st.sidebar.slider("📊 Minimum Skor Kemiripan", 0.0, 1.0, 0.12, 0.01, 
                                  help="Naikkan slider untuk hasil lebih presisi. Turunkan untuk hasil lebih banyak.")
    top_k = st.sidebar.slider("📦 Jumlah Hasil Maksimal", 1, 50, 15)
    search_mode = st.sidebar.radio("🔍 Mode Pencarian", ["Judul Saja", "Judul + Genre + Deskripsi"])

    st.markdown("---")
    query = st.text_input("🔎 Masukkan kata kunci (bisa parsial):", placeholder="Contoh: stranger, black mirror, sci-fi...")
    
    if st.button("🚀 Cari Sekarang", type="primary", use_container_width=True) and query.strip():
        with st.spinner("⏳ Memproses query & menghitung kemiripan..."):
            # Sesuaikan kolom pencarian berdasarkan mode
            if search_mode == "Judul Saja":
                engine.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.85, min_df=2)
                engine.doc_vectors = engine.vectorizer.fit_transform(engine.df['title'].fillna('').str.lower().str.replace(r'[^\w\s]','', regex=True))
            else:
                engine.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.85, min_df=2)
                engine.doc_vectors = engine.vectorizer.fit_transform(engine.df['search_text'])
                
            results = engine.search(query, min_score=min_score, top_k=top_k)
            
            if not results.empty:
                st.success(f"✅ Ditemukan **{len(results)}** judul relevan!")
                st.dataframe(
                    results,
                    column_config={
                        "Peringkat": st.column_config.NumberColumn(format="%d"),
                        "Skor Kemiripan": st.column_config.NumberColumn(format="%.3f", help="Semakin dekat ke 1.0, semakin relevan"),
                        "Tahun Rilis": st.column_config.NumberColumn(format="%d"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=450
                )
            else:
                st.warning("⚠️ Tidak ada hasil dengan skor di atas batas minimum. Coba turunkan slider atau ubah kata kunci.")
    elif query.strip() == "":
        st.info("💡 Ketik judul, genre, atau kata kunci di atas untuk memulai.")

if __name__ == "__main__":
    main()

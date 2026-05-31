import streamlit as st
import numpy as np
import re
import os
from nltk.corpus import wordnet
import streamlit as st

# Suppress NLTK download warnings in production
import warnings
warnings.filterwarnings('ignore')

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class NetflixSearchEngine:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['title_clean'] = self.df['title'].fillna('').apply(self._clean_text)
        
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
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        tokens = clean_query.split()
        expanded = []
        for token in tokens:
            expanded.append(token)
            try:
                for syn in wordnet.synsets(token, pos=(wordnet.NOUN, wordnet.ADJ)):
                    for lemma in syn.lemmas()[:2]:
                        expanded.append(lemma.name().lower())
            except Exception:
                pass
        return ' '.join(list(set(expanded)))

    def search(self, query, min_score=0.05, top_k=10):
        expanded_query = self._expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        mask = similarities >= min_score
        filtered_scores = similarities[mask]
        filtered_indices = np.where(mask)[0]
        
        if len(filtered_indices) == 0:
            return pd.DataFrame(columns=['Peringkat', 'Skor Similarity', 'Jenis', 'Judul', 'Tahun Rilis', 'Durasi'])
            
        top_indices = filtered_indices[filtered_scores.argsort()[::-1]][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.df.iloc[idx]
            results.append({
                'Peringkat': rank,
                'Skor Similarity': round(float(similarities[idx]), 4),
                'Jenis': row['type'],
                'Judul': row['title'],
                'Tahun Rilis': row['release_year'],
                'Durasi': row['duration']
            })
        return pd.DataFrame(results)

@st.cache_resource
def load_engine(path):
    return NetflixSearchEngine(path)

def main():
    st.set_page_config(page_title="Netflix Search Engine", page_icon="🎬", layout="wide")
    
    st.title("🎬 Netflix Search Engine")
    st.markdown("Temukan film atau acara TV Netflix berdasarkan judul. **Tidak perlu mengetik judul lengkap!**")
    
    csv_path = st.text_input("📂 Path File Dataset CSV:", value="Netflix_movies_and_tv_shows.csv")
    if not os.path.exists(csv_path):
        st.error("❌ File tidak ditemukan. Pastikan path benar atau upload file ke folder yang sama.")
        return

    with st.spinner("⏳ Memuat dataset & membangun indeks..."):
        engine = load_engine(csv_path)
        st.success("✅ Mesin pencarian siap digunakan!")

    st.sidebar.header("⚙️ Pengaturan")
    min_score = st.sidebar.slider("Minimum Skor Kemiripan", 0.0, 1.0, 0.08, 0.01)
    top_k = st.sidebar.slider("Jumlah Hasil", 1, 50, 10)

    st.divider()
    query = st.text_input("Ketik judul film/TV show:", placeholder="Contoh: stranger things, black mirror...")
    
    if st.button("🔎 Cari Sekarang", type="primary") and query.strip():
        with st.spinner("🔎 Memproses pencarian..."):
            results = engine.search(query, min_score=min_score, top_k=top_k)
            
        if not results.empty:
            st.success(f"✅ Ditemukan **{len(results)}** judul!")
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
            st.warning("⚠️ Tidak ada hasil di atas batas skor. Coba turunkan slider atau ubah kata kunci.")
    elif not query.strip():
        st.info("💡 Masukkan kata kunci untuk memulai pencarian.")

if __name__ == "__main__":
    main()

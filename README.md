# 🔎 Resume ↔ Job Description Matcher  

This is a **Streamlit-based prototype** built with **Python, Streamlit, NLP, and Machine Learning libraries**.  
It demonstrates how resumes can be compared against job descriptions using **both classic IR (TF-IDF)** and **modern AI (SBERT embeddings)**.  

---

## 🚀 Setup Instructions (User Flow)

When you open the app:

### 🏠 Home Page
- You will see the title *“Resume ↔ Job Description Matcher”*.  
- On the left sidebar, you can configure the **scoring method** (TF-IDF or SBERT), **n-gram ranges**, and **advanced NLP options**.  
- Upload a **Resume file** (`.pdf`, `.docx`, `.txt`).  
- Upload one or more **Job Description files** (`.pdf`, `.docx`, `.txt`).  
- Click **Run Matching** to generate results.  

> **Note**: Sample CVs and JDs are available in the [helperfiles repo](https://github.com/MaahirHussainShaik/helperfiles.git).

---

### 📄 Resume ↔ JD Matching

1. **Select Scoring Method**
   - **TF-IDF**: Keyword/phrase overlap with cosine similarity.  
   - **SBERT (Embeddings)**: Semantic similarity using transformer embeddings (`all-MiniLM-L6-v2`).  

2. **Upload Files**
   - Upload your **resume** and multiple **job descriptions**.  
   - Text is automatically extracted, cleaned, and normalised.  

3. **Results Table**
   - A dataframe of **match percentages** is shown, sorted from highest to lowest.  
   - A **bar chart** visualises the ranking.  

4. **Explanations**
   - **TF-IDF**: shows **top contributing terms/n-grams** (resume × JD overlap).  
   - **SBERT**: highlights **top semantic sentence pairs** (resume vs JD).  

5. **Skill Coverage**
   - Extracts technical skills (Python, SQL, NLP, ML, etc.) from both resume and JDs.  
   - Shows overlaps per JD to reveal **strengths and gaps**.  

---

### ⚙️ Advanced NLP Options  

- **Min/Max n-gram sliders**: control whether to match unigrams, bigrams, trigrams.  
- **Stemming toggle**: reduces words to root form (e.g. “running” → “run”).  
- **Synonym expansion**: automatically maps related terms (e.g. “ML” → “Machine Learning”, “PyTorch”).  
- **Custom stopwords**: add extra stopwords to ignore.  
- **Section weighting**: boosts importance of *Skills, Projects, Experience* sections in resumes.  

---

## ✨ Main Features Implemented

- ✅ Upload support for `.pdf`, `.docx`, `.txt` resumes and JDs  
- ✅ Multi-method scoring: **TF-IDF** & **SBERT embeddings**  
- ✅ Explainability: term contributions (TF-IDF) and sentence alignments (SBERT)  
- ✅ Adjustable NLP pipeline: stemming, stopwords, synonyms, section weighting  
- ✅ Skill coverage analysis with overlap detection  
- ✅ Interactive Streamlit UI with sidebar controls & tooltips  
- ✅ Visualisation: DataFrame + bar chart output  

---

## 📖 How the Code Works

### TF-IDF Pipeline  
- Tokenises text, removes stopwords, applies optional stemming/synonyms.  
- Builds TF-IDF vectors across resume + JDs.  
- Computes cosine similarity between resume and each JD.  
- Extracts **top 25 n-grams** that overlap.  

### SBERT Pipeline  
- Loads pretrained transformer (`all-MiniLM-L6-v2`).  
- Encodes resume and JDs into dense vectors.  
- Computes cosine similarity in embedding space.  
- Aligns most semantically similar **resume–JD sentence pairs**.  

### Skill Coverage  
- Uses a custom synonym dictionary (Python, ML, SQL, PyTorch, etc.).  
- Finds overlap between resume skills and JD requirements.  
- Outputs a table of matches and counts.  

---

## 🆚 Why this is better than the Web App baseline  

The baseline web app (`useSimpleScorer.tsx` in React) only:  
- Tokenises text with a basic stopword filter.  
- Builds lightweight TF-IDF vectors.  
- Computes cosine similarity.  
- Returns percentage match only.  

This Streamlit app improves on it by:  
1. **Multiple methods** → traditional TF-IDF + semantic embeddings.  
2. **Advanced NLP** → stemming, synonyms, section weighting.  
3. **Explainability** → n-gram contributions & sentence-level alignments.  
4. **Skill coverage analysis** → explicit resume–JD overlap insights.  
5. **File support** → works with `.pdf`, `.docx`, `.txt`.  
6. **Richer UI** → sidebar tooltips, expanders, bar charts.  

---

## 🛠️ Tech Stack

- **Python 3.10+**  
- **Streamlit** — interactive UI  
- **scikit-learn** — TF-IDF & cosine similarity  
- **sentence-transformers** — semantic embeddings (SBERT)  
- **pdfminer.six / docx2txt** — file parsing  
- **nltk** — stemming & tokenization  

---

## ⚠️ Known Limitations / Future Improvements

- SBERT requires downloading a pretrained model (can be slow initially).  
- Current synonym/skill dictionary is small — could be expanded.  
- No persistence layer (all data processed in-memory).  
- Explanations can be verbose for large resumes/JDs.  
- Hybrid scoring (TF-IDF + SBERT combined) not yet implemented.  
- No export function (CSV/Excel report generation planned).  
- Not optimised for very large corpora.  

---

## 🔮 Summary

This project demonstrates **practical AI/ML + NLP skills** applied to a real-world scenario:  
- Information Retrieval (TF-IDF).  
- Semantic Matching (SBERT embeddings).  
- Explainability of results.  
- Skill mining and overlap detection.  
- Interactive deployment via Streamlit.  

It is both a **portfolio-ready demo** and a foundation for more advanced AI-powered recruitment tools.  

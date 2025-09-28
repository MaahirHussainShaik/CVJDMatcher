import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from typing import List, Tuple, Dict

# =========================
# Lazy / optional imports
# =========================
@st.cache_resource(show_spinner=False)
def safe_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception:
        return None

def safe_import_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        return TfidfVectorizer, cosine_similarity
    except Exception:
        return None, None

# =========================
# File loaders
# =========================
def read_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")

def read_pdf(data: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        tmp = "tmp_resume_jd.pdf"
        with open(tmp, "wb") as f:
            f.write(data)
        text = extract_text(tmp) or ""
        os.remove(tmp)
        return text
    except Exception:
        return ""

def read_docx(data: bytes) -> str:
    try:
        import docx2txt, tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmpf:
            tmpf.write(data)
            tmp_path = tmpf.name
        text = docx2txt.process(tmp_path) or ""
        os.remove(tmp_path)
        return text
    except Exception:
        return ""

def load_file(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".txt"):
        return read_txt(data)
    if name.endswith(".pdf"):
        return read_pdf(data)
    if name.endswith(".docx"):
        return read_docx(data)
    return read_txt(data)

# =========================
# NLP utils
# =========================
WORD_RE = re.compile(r"[a-z0-9#+./-]+")

DEFAULT_STOP = set("""
a an the and or to for of in on at by from with without into over under up down as is are was were be been being 
this that those these i you he she it we they them his her their our your my me him her us them
will would can could should may might must do does did done doing have has had having than then across per via
""".split())

# simple, no-download stemmer
try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
except Exception:
    STEMMER = None

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

def stem_tokens(tokens: List[str]) -> List[str]:
    if STEMMER is None:
        return tokens
    return [STEMMER.stem(t) for t in tokens]

def make_analyzer(ngram_range=(1,2), stop=set(), use_stem=False):
    def analyzer(doc):
        toks = [t for t in tokenize(doc) if t not in stop and len(t) > 1]
        if use_stem:
            toks = stem_tokens(toks)
        grams = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            if n == 1:
                grams.extend(toks)
            else:
                grams.extend([" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)])
        return grams
    return analyzer

# naive sentence splitter (works fine for our explainers)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sents = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]

# =========================
# Domain helpers
# =========================
# Lightweight synonym expansion for common tech terms (used for skills/coverage)
SYNONYMS: Dict[str, List[str]] = {
    "python": ["python3", "py", "pandas", "numpy"],
    "machine learning": ["ml", "supervised", "unsupervised", "classification", "regression"],
    "deep learning": ["dl", "neural network", "cnn", "rnn", "transformer", "pytorch", "keras"],
    "nlp": ["natural language processing", "text mining", "bert", "sbert", "tokenization", "tf-idf"],
    "data science": ["analytics", "eda", "statistics"],
    "sql": ["postgres", "mysql", "sqlite", "t-sql"],
    "streamlit": ["st", "web app", "dashboard"],
    "docker": ["container", "containerization"],
    "git": ["github", "gitlab"],
}

SKILL_LIST = sorted(set([k for k in SYNONYMS.keys()] + [a for v in SYNONYMS.values() for a in v]))

def find_skills(text: str) -> List[str]:
    t = " " + " ".join(tokenize(text)) + " "
    found = []
    for skill in SKILL_LIST:
        if f" {skill} " in t:
            found.append(skill)
    return sorted(set(found))

# simple section weighting (boosts certain resume sections if present)
SECTION_WEIGHTS = {
    "skills": 1.3,
    "projects": 1.2,
    "experience": 1.2,
    "education": 1.0,
    "summary": 1.1,
}

def apply_section_weights(text: str) -> str:
    # Duplicate weighted sections to heuristically boost their impact in bag-of-words models
    boosted = [text]
    lower = text.lower()
    for sec, w in SECTION_WEIGHTS.items():
        if sec in lower and w > 1.0:
            # simple heuristic: append the section header token w-1 times
            boosted.append((" " + sec) * int((w - 1.0) * 3))
    return " ".join(boosted)

def expand_query_with_synonyms(text: str) -> str:
    # Add synonyms of tokens seen in the doc to help TF-IDF find more overlaps
    toks = set(tokenize(text))
    extra = []
    for base, syns in SYNONYMS.items():
        head = base.replace(" ", "")
        if base in toks or head in toks or any(s in toks for s in syns):
            extra.extend(syns + [base])
    if extra:
        return text + " " + " ".join(extra)
    return text

def as_pct(x: float) -> float:
    try:
        return round(100 * max(0.0, float(x)), 2)
    except Exception:
        return 0.0

# =========================
# TF-IDF scorer + explain
# =========================
def tfidf_score(
    resume: str,
    jds: List[str],
    ngram_range=(1,2),
    use_stem=False,
    custom_stop: List[str] = None,
    query_expand=False,
    return_contrib=True
):
    TfidfVectorizer, cosine_similarity = safe_import_sklearn()
    if TfidfVectorizer is None:
        raise RuntimeError("scikit-learn is required for TF-IDF mode")

    stop = DEFAULT_STOP.copy()
    if custom_stop:
        stop |= set([s.lower() for s in custom_stop])

    analyzer = make_analyzer(ngram_range=ngram_range, stop=stop, use_stem=use_stem)

    # Optional section boosting & synonym expansion
    resume_aug = apply_section_weights(resume)
    if query_expand:
        resume_aug = expand_query_with_synonyms(resume_aug)
        jds = [expand_query_with_synonyms(j) for j in jds]

    vect = TfidfVectorizer(analyzer=analyzer, min_df=1)
    corpus = [resume_aug] + jds
    X = vect.fit_transform(corpus)

    sims = cosine_similarity(X[0:1], X[1:]).ravel()
    features = np.array(vect.get_feature_names_out())

    contrib = None
    if return_contrib:
        # contribution ~ elementwise product of TF-IDF weights
        r = X[0].toarray()[0]
        contrib = []
        for i in range(1, X.shape[0]):
            j = X[i].toarray()[0]
            joint = r * j
            nz = np.where(joint > 0)[0]
            pairs = list(zip(features[nz], joint[nz]))
            pairs.sort(key=lambda x: x[1], reverse=True)
            contrib.append(pairs[:25])  # top 25 contributors

    return sims, features, contrib

# =========================
# SBERT scorer + explain
# =========================
@st.cache_resource(show_spinner=False)
def load_sbert():
    SentenceTransformer = safe_import_sentence_transformers()
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

def sbert_score(resume: str, jds: List[str]):
    model = load_sbert()
    if model is None:
        raise RuntimeError("Install sentence-transformers for Embeddings mode")

    # Doc-level similarity
    embs = model.encode([resume] + jds, normalize_embeddings=True)
    sims = (embs[0:1] @ embs[1:].T).ravel()

    # Sentence-level highlights (for explainability)
    r_sents = split_sentences(resume)
    jd_sents_all = [split_sentences(j) for j in jds]

    sent_pairs = []
    if r_sents and any(jd_sents_all):
        # encode all sentences once per doc
        r_emb = model.encode(r_sents, normalize_embeddings=True)
        for jd_sents in jd_sents_all:
            if not jd_sents:
                sent_pairs.append([])
                continue
            j_emb = model.encode(jd_sents, normalize_embeddings=True)
            sim_mat = r_emb @ j_emb.T  # (R x J)
            pairs = []
            # top alignments by greedy max (limit size for UI)
            flat = []
            for i in range(sim_mat.shape[0]):
                for j in range(sim_mat.shape[1]):
                    flat.append((float(sim_mat[i, j]), i, j))
            flat.sort(reverse=True)
            seen_r, seen_j = set(), set()
            for s, i, j in flat:
                if i in seen_r or j in seen_j:
                    continue
                pairs.append((s, r_sents[i], jd_sents[j]))
                seen_r.add(i); seen_j.add(j)
                if len(pairs) >= 7:  # keep it readable
                    break
            sent_pairs.append(pairs)
    else:
        sent_pairs = [[] for _ in jds]

    return sims, sent_pairs

# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Resume ↔ JD Matcher", page_icon="🔎", layout="wide")
    st.title("🔎 Resume ↔ Job Description Matcher")

    with st.sidebar:
        st.subheader("Settings")

        method = st.selectbox(
            "Scoring method",
            ["TF-IDF", "Embeddings (SBERT)"],
            help=(
                "TF-IDF: Compares overlap of important terms and n-grams.\n\n"
                "Embeddings (SBERT): Uses a transformer to capture semantic similarity "
                "even when wording differs."
            ),
        )

        ngram_low = st.slider(
            "Min n-gram", 1, 2, 1,
            help="Minimum size of word groups (n). n=1 uses single words."
        )
        ngram_high = st.slider(
            "Max n-gram", 1, 3, 2,
            help="Maximum size of word groups (n). n=2 adds bigrams like “machine learning”."
        )

        show_terms = st.checkbox(
            "Show matched terms / explanations", True,
            help="Displays why a JD scored well: top contributing terms (TF-IDF) or sentence pairs (SBERT)."
        )

        st.markdown("---")
        st.caption("🔧 **Pro mode (NLP)**")
        use_stem = st.checkbox(
            "Use stemming (Porter)",
            value=False,
            help="Reduces words to roots (e.g., running→run). Helps match variants."
        )
        query_expand = st.checkbox(
            "Synonym expansion",
            value=True,
            help="Adds lightweight synonyms (e.g., ML→machine learning, PyTorch) to improve overlap."
        )
        custom_stop_str = st.text_input(
            "Custom stopwords (comma-separated)",
            value="",
            help="Words to ignore (e.g., role-specific boilerplate)."
        )
        weight_sections = st.checkbox(
            "Weight resume sections",
            value=True,
            help="Heuristically boosts sections like ‘Skills’, ‘Projects’, ‘Experience’ in TF-IDF."
        )

    with st.expander("ℹ️ What do these methods mean?", expanded=False):
        st.markdown(
            """
**TF-IDF**  
- Treats text as bags of words/phrases.  
- Scores higher when your resume shares **important** n-grams with a JD.  
- Great for keyword alignment (Python, SQL, REST API).  

**SBERT (Embeddings)**  
- Turns each document/sentence into a vector capturing **meaning**.  
- Finds overlap even if wording differs (e.g., “build classifiers” ≈ “train ML models”).  

**n-grams**  
- *Min/Max n-gram*: control phrase length.  
- With max=2 you’ll match bigrams like “data pipeline” not just “data” and “pipeline”.
            """
        )

    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    jd_files = st.file_uploader("Upload Job Descriptions", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if st.button("Run Matching", type="primary"):
        if not resume_file or not jd_files:
            st.error("Upload both a resume and at least one JD.")
            return

        resume_raw = load_file(resume_file)
        jd_raw_list = [load_file(f) for f in jd_files]
        jd_names = [f.name for f in jd_files]

        if not resume_raw.strip():
            st.error("Resume appears empty or could not be parsed.")
            return
        if not any(j.strip() for j in jd_raw_list):
            st.error("All JDs appear empty or could not be parsed.")
            return

        # Prepare options
        custom_stop = [s.strip().lower() for s in custom_stop_str.split(",") if s.strip()]
        resume_text = resume_raw
        jd_texts = jd_raw_list

        try:
            if method == "TF-IDF":
                # Optionally apply section weighting to resume only
                resume_proc = apply_section_weights(resume_text) if weight_sections else resume_text
                sims, features, contrib = tfidf_score(
                    resume=resume_proc,
                    jds=jd_texts,
                    ngram_range=(ngram_low, ngram_high),
                    use_stem=use_stem,
                    custom_stop=custom_stop,
                    query_expand=query_expand,
                    return_contrib=True
                )
                explanations = contrib  # list of [(term, weight), ...]
                sent_pairs = None
            else:
                sims, sent_pairs = sbert_score(resume_text, jd_texts)
                explanations = None  # TF-IDF only

        except Exception as e:
            st.error(f"Scoring failed: {e}")
            return

        scores = [as_pct(s) for s in sims]
        df = pd.DataFrame({"Job File": jd_names, "Match %": scores}).sort_values("Match %", ascending=False)
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        # Bar chart
        st.bar_chart(data=df.set_index("Job File"))

        # Skill coverage (quick, interpretable)
        st.subheader("Skill Coverage")
        res_skills = find_skills(resume_text)
        st.markdown(f"**Resume skills detected:** {', '.join(res_skills) if res_skills else '—'}")

        coverage_rows = []
        for name, jd in zip(jd_names, jd_texts):
            jd_sk = find_skills(jd)
            overlap = sorted(set(res_skills) & set(jd_sk))
            coverage_rows.append({
                "Job File": name,
                "JD skills": ", ".join(jd_sk) if jd_sk else "—",
                "Overlap with Resume": ", ".join(overlap) if overlap else "—",
                "Overlap Count": len(overlap),
            })
        cov_df = pd.DataFrame(coverage_rows).sort_values("Overlap Count", ascending=False)
        st.dataframe(cov_df, use_container_width=True)

        if show_terms:
            st.subheader("Explanations")

            if method == "TF-IDF" and explanations is not None:
                for name, contrib_terms in zip(jd_names, explanations):
                    with st.expander(f"Why did **{name}** score {df[df['Job File']==name]['Match %'].values[0]}%? (TF-IDF term contributions)"):
                        if not contrib_terms:
                            st.write("No overlapping terms found.")
                        else:
                            top_terms = pd.DataFrame(contrib_terms, columns=["n-gram", "contribution"])
                            st.write("Top contributing n-grams (resume × JD TF-IDF product):")
                            st.dataframe(top_terms, use_container_width=True)
            elif method == "Embeddings (SBERT)" and sent_pairs is not None:
                for name, pairs in zip(jd_names, sent_pairs):
                    with st.expander(f"Why did **{name}** score {df[df['Job File']==name]['Match %'].values[0]}%? (Top semantic sentence pairs)"):
                        if not pairs:
                            st.write("No salient sentence alignments extracted.")
                        else:
                            for s, r_sent, j_sent in pairs:
                                st.markdown(f"- **Sim ~ {s:.2f}**\n  - Resume: _{r_sent}_\n  - JD: _{j_sent}_")

        st.caption(
            "Notes: TF-IDF scores reflect keyword/phrase alignment; Embedding scores reflect semantic similarity. "
            "Percentages are scaled cosine similarities for readability."
        )

if __name__ == "__main__":
    main()

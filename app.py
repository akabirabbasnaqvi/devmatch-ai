import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DevMatch AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0a0a0f;
    --card: #111118;
    --border: #1e1e2e;
    --accent: #7c6af7;
    --accent2: #22d3a5;
    --accent3: #f7766a;
    --text: #e8e8f0;
    --muted: #6b6b80;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background-color: var(--bg) !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Main container */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1200px !important;
}

/* Hero section */
.hero {
    text-align: center;
    padding: 3rem 0 2rem 0;
    position: relative;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c6af7, #22d3a5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-desc {
    color: #9090a8;
    font-size: 1rem;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Cards */
.dev-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s;
}

.dev-card:hover {
    border-color: var(--accent);
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Result cards */
.result-big {
    background: linear-gradient(135deg, #1a1830, #111118);
    border: 1px solid var(--accent);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}

.result-role {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c6af7, #22d3a5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-acc {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

.metric-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
}

.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* Skill pills */
.skill-pill {
    display: inline-block;
    background: #1a1830;
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    margin: 3px;
    font-family: 'Space Mono', monospace;
}

.skill-pill-gap {
    display: inline-block;
    background: #1e1215;
    border: 1px solid var(--accent3);
    color: var(--accent3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    margin: 3px;
    font-family: 'Space Mono', monospace;
}

.skill-pill-have {
    display: inline-block;
    background: #0e1e1a;
    border: 1px solid var(--accent2);
    color: var(--accent2);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    margin: 3px;
    font-family: 'Space Mono', monospace;
}

/* Neighbor card */
.neighbor-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Progress bar */
.prog-wrap {
    margin-bottom: 0.8rem;
}

.prog-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #9090a8;
    margin-bottom: 4px;
}

.prog-bar-bg {
    background: #1e1e2e;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}

.prog-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #7c6af7, #22d3a5);
}

/* Divider */
.div-line {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Streamlit widget overrides */
.stMultiSelect > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stSlider > div {
    color: var(--text) !important;
}

div[data-testid="stSelectbox"] > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c6af7, #5a4fd4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.badge-purple { background: #1a1830; color: #7c6af7; border: 1px solid #7c6af7; }
.badge-green  { background: #0e1e1a; color: #22d3a5; border: 1px solid #22d3a5; }
.badge-red    { background: #1e1215; color: #f7766a; border: 1px solid #f7766a; }

/* Step number */
.step-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c6af7, #22d3a5);
    color: white;
    font-weight: 800;
    font-size: 0.85rem;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Load model and data ───────────────────────────────────────
@st.cache_resource
def load_model():
    knn = joblib.load('devmatch_knn_model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return knn, feature_cols

@st.cache_data
def load_clean_data():
    return pd.read_csv('devmatch_clean.csv')

try:
    knn_model, feature_cols = load_model()
    df_data = load_clean_data()
    model_loaded = True
except:
    model_loaded = False

# ── Skill options ─────────────────────────────────────────────
LANGUAGES = ['Python', 'JavaScript', 'TypeScript', 'Java', 'C#', 'C++', 'C',
             'Go', 'Rust', 'PHP', 'Ruby', 'Swift', 'Kotlin', 'Dart',
             'HTML/CSS', 'SQL', 'Bash/Shell (all shells)', 'PowerShell', 'R']

FRAMEWORKS = ['React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask',
              'ASP.NET CORE', 'Spring Boot', 'Laravel', 'Flutter',
              'Next.js', 'FastAPI', 'Express', 'Svelte']

DATABASES = ['PostgreSQL', 'MySQL', 'MongoDB', 'SQLite', 'Redis',
             'Microsoft SQL Server', 'Firebase Realtime Database',
             'Dynamodb', 'Elasticsearch', 'Oracle']

TOOLS = ['Docker', 'Kubernetes', 'Git', 'Linux', 'AWS', 'Azure',
         'Google Cloud', 'Terraform', 'Jenkins', 'Nginx']

ROLE_ROADMAP = {
    'Developer, full-stack': {
        'must': ['JavaScript', 'React', 'Node.js', 'SQL', 'HTML/CSS', 'Git'],
        'good': ['TypeScript', 'Docker', 'PostgreSQL', 'AWS']
    },
    'Developer, back-end': {
        'must': ['Python', 'SQL', 'REST API', 'Docker', 'Linux', 'Git'],
        'good': ['PostgreSQL', 'Redis', 'AWS', 'Kubernetes']
    },
    'Developer, front-end': {
        'must': ['JavaScript', 'React', 'HTML/CSS', 'TypeScript', 'Git'],
        'good': ['Vue.js', 'Next.js', 'Figma', 'CSS frameworks']
    },
    'Student': {
        'must': ['Python', 'JavaScript', 'SQL', 'Git', 'HTML/CSS'],
        'good': ['React', 'Docker', 'Linux', 'C++']
    },
    'Developer, mobile': {
        'must': ['Flutter', 'Kotlin', 'Swift', 'Dart', 'Git'],
        'good': ['Firebase Realtime Database', 'REST API', 'Java', 'React']
    },
    'Developer, desktop or enterprise applications': {
        'must': ['C#', 'Java', 'C++', 'SQL', 'Git'],
        'good': ['ASP.NET CORE', 'Docker', 'Azure', 'Python']
    },
    'Developer, embedded applications or devices': {
        'must': ['C', 'C++', 'Python', 'Linux', 'Assembly'],
        'good': ['Rust', 'Bash/Shell (all shells)', 'Git', 'Docker']
    },
    'Other (please specify):': {
        'must': ['Python', 'SQL', 'Git', 'JavaScript'],
        'good': ['Docker', 'Linux', 'AWS', 'TypeScript']
    }
}

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-sub">KNN • Machine Learning • Developer Profiler</div>
    <div class="hero-title">DevMatch AI</div>
    <div class="hero-desc">Enter your skills and experience. Our KNN model analyzes 45,000+ real developers to predict your role and skill gaps.</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found. Make sure `devmatch_knn_model.pkl`, `feature_columns.pkl`, and `devmatch_clean.csv` are in the same folder as this app.")
    st.stop()

# ── Layout ────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="dev-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">// Your Profile</div>', unsafe_allow_html=True)

    dev_name = st.text_input("Your Name", placeholder="e.g. Ali Hassan", label_visibility="visible")

    st.markdown("**Programming Languages**")
    sel_langs = st.multiselect("", LANGUAGES, default=['Python', 'JavaScript'], label_visibility="collapsed", key="langs")

    st.markdown("**Frameworks & Libraries**")
    sel_frames = st.multiselect("", FRAMEWORKS, default=['React'], label_visibility="collapsed", key="frames")

    st.markdown("**Databases**")
    sel_dbs = st.multiselect("", DATABASES, default=['PostgreSQL'], label_visibility="collapsed", key="dbs")

    st.markdown("**Tools & Platforms**")
    sel_tools = st.multiselect("", TOOLS, default=['Docker', 'Git'], label_visibility="collapsed", key="tools")

    st.markdown("**Years of Professional Experience**")
    exp_years = st.slider("", 0, 20, 2, label_visibility="collapsed")

    st.markdown("**Education Level**")
    edu = st.selectbox("", [
        "Primary/elementary school",
        "Secondary school",
        "Some college without a degree",
        "Bachelor's degree",
        "Master's degree",
        "PhD / Professional degree"
    ], index=3, label_visibility="collapsed")

    st.markdown("**Target Role (for roadmap)**")
    target_role = st.selectbox("", list(ROLE_ROADMAP.keys()), label_visibility="collapsed", key="target")

    st.markdown("**K — Number of Neighbors**")
    k_val = st.slider("", 1, 15, 3, label_visibility="collapsed", key="kslider")

    st.markdown("</div>", unsafe_allow_html=True)
    run_btn = st.button("🧬 Analyze My Profile")

# ── Prediction logic ──────────────────────────────────────────
with col_right:
    if run_btn:
        all_skills = sel_langs + sel_frames + sel_dbs + sel_tools
        if not all_skills:
            st.warning("Please select at least one skill.")
            st.stop()

        edu_map = {
            "Primary/elementary school": 1,
            "Secondary school": 2,
            "Some college without a degree": 3,
            "Bachelor's degree": 5,
            "Master's degree": 6,
            "PhD / Professional degree": 7
        }

        # Build input vector
        input_vec = {}
        for col in feature_cols:
            if col == 'experience':
                input_vec[col] = exp_years
            elif col == 'edu_level':
                input_vec[col] = edu_map.get(edu, 3)
            else:
                input_vec[col] = 1 if col in all_skills else 0

        input_df = pd.DataFrame([input_vec])
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        # Run KNN
        knn_model.n_neighbors = k_val
        predicted_role = knn_model.predict(input_df)[0]
        proba = knn_model.predict_proba(input_df)[0]
        classes = knn_model.classes_
        confidence = round(max(proba) * 100, 1)

        # Get neighbors
        distances, indices = knn_model.kneighbors(input_df, n_neighbors=min(k_val, len(df_data)))
        safe_indices = [i % len(df_data) for i in indices[0]]
        neighbor_rows = df_data.iloc[safe_indices]

        # ── Result header ─────────────────────────────────────
        name_str = f"{dev_name}'s" if dev_name else "Your"
        st.markdown(f"""
        <div class="result-big">
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#6b6b80;letter-spacing:3px;text-transform:uppercase;margin-bottom:0.5rem">{name_str} predicted role</div>
            <div class="result-role">{predicted_role}</div>
            <div class="result-acc">KNN confidence: {confidence}% &nbsp;|&nbsp; K={k_val} neighbors &nbsp;|&nbsp; {len(all_skills)} skills analyzed</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ───────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{len(all_skills)}</div><div class="metric-label">Skills entered</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{exp_years}yr</div><div class="metric-label">Experience</div></div>', unsafe_allow_html=True)
        with m3:
            level = "Junior" if exp_years <= 2 else ("Mid" if exp_years <= 5 else "Senior")
            st.markdown(f'<div class="metric-box"><div class="metric-val">{level}</div><div class="metric-label">Level</div></div>', unsafe_allow_html=True)

        st.markdown('<hr class="div-line">', unsafe_allow_html=True)

        # ── Role probability chart ────────────────────────────
        st.markdown('<div class="section-label">// Role Probability Distribution</div>', unsafe_allow_html=True)
        role_short = {
            'Developer, full-stack': 'Full-Stack',
            'Developer, back-end': 'Back-End',
            'Developer, front-end': 'Front-End',
            'Student': 'Student',
            'Developer, mobile': 'Mobile',
            'Developer, desktop or enterprise applications': 'Desktop',
            'Developer, embedded applications or devices': 'Embedded',
            'Other (please specify):': 'Other'
        }
        prob_df = pd.DataFrame({'Role': classes, 'Probability': proba})
        prob_df = prob_df.sort_values('Probability', ascending=False)

        bars_html = ""
        for _, row in prob_df.iterrows():
            pct = round(row['Probability'] * 100, 1)
            short = role_short.get(row['Role'], row['Role'])
            bars_html += f"""
            <div class="prog-wrap">
                <div class="prog-label"><span>{short}</span><span>{pct}%</span></div>
                <div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{pct}%"></div></div>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown('<hr class="div-line">', unsafe_allow_html=True)

        # ── Nearest neighbors ─────────────────────────────────
        st.markdown(f'<div class="section-label">// {k_val} Nearest Neighbors</div>', unsafe_allow_html=True)
        role_emojis = {
            'Developer, full-stack': '🔷',
            'Developer, back-end': '⚙️',
            'Developer, front-end': '🎨',
            'Student': '🎓',
            'Developer, mobile': '📱',
            'Developer, desktop or enterprise applications': '🖥️',
            'Developer, embedded applications or devices': '🔌',
            'Other (please specify):': '🔹'
        }
        for i, (dist, idx) in enumerate(zip(distances[0], safe_indices)):
            neighbor = df_data.iloc[idx]
            n_role = neighbor['role']
            similarity = max(0, round((1 - dist / (dist + 1)) * 100, 1))
            emoji = role_emojis.get(n_role, '👤')
            short_role = role_short.get(n_role, n_role)
            st.markdown(f"""
            <div class="neighbor-card">
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:1.4rem">{emoji}</span>
                    <div>
                        <div style="font-weight:600;font-size:0.9rem">Developer #{i+1}</div>
                        <div style="font-size:0.75rem;color:#6b6b80;font-family:'Space Mono',monospace">{short_role}</div>
                    </div>
                </div>
                <div style="text-align:right">
                    <span class="badge badge-purple">{similarity}% similar</span>
                    <div style="font-size:0.7rem;color:#6b6b80;margin-top:4px;font-family:'Space Mono',monospace">dist: {dist:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="div-line">', unsafe_allow_html=True)

        # ── Skill gap roadmap ─────────────────────────────────
        st.markdown(f'<div class="section-label">// Skill Gap → {role_short.get(target_role, target_role)}</div>', unsafe_allow_html=True)
        roadmap = ROLE_ROADMAP.get(target_role, {'must': [], 'good': []})
        must_skills = roadmap['must']
        good_skills = roadmap['good']

        have_must = [s for s in must_skills if s in all_skills]
        missing_must = [s for s in must_skills if s not in all_skills]
        have_good = [s for s in good_skills if s in all_skills]
        missing_good = [s for s in good_skills if s not in all_skills]

        if have_must or have_good:
            st.markdown("**✅ Skills you already have:**")
            pills = "".join([f'<span class="skill-pill-have">{s}</span>' for s in have_must + have_good])
            st.markdown(pills, unsafe_allow_html=True)

        if missing_must:
            st.markdown("**🔴 Must-learn (core requirements):**")
            pills = "".join([f'<span class="skill-pill-gap">{s}</span>' for s in missing_must])
            st.markdown(pills, unsafe_allow_html=True)

        if missing_good:
            st.markdown("**🟡 Good to learn (bonus skills):**")
            pills = "".join([f'<span class="skill-pill">{s}</span>' for s in missing_good])
            st.markdown(pills, unsafe_allow_html=True)

        gap_pct = round(len(have_must) / len(must_skills) * 100) if must_skills else 100
        st.markdown(f"""
        <div style="margin-top:1rem;padding:1rem;background:#111118;border:1px solid #1e1e2e;border-radius:12px">
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#6b6b80;margin-bottom:6px">READINESS FOR TARGET ROLE</div>
            <div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{gap_pct}%"></div></div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#22d3a5;margin-top:6px">{gap_pct}% ready</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder when no analysis run yet
        st.markdown("""
        <div style="height:400px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#111118;border:1px dashed #1e1e2e;border-radius:16px;text-align:center;padding:2rem">
            <div style="font-size:3rem;margin-bottom:1rem">🧬</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e8e8f0;margin-bottom:0.5rem">Ready to analyze</div>
            <div style="font-size:0.85rem;color:#6b6b80;max-width:300px;line-height:1.6">
                Fill in your skills on the left and click<br><strong style="color:#7c6af7">Analyze My Profile</strong> to run the KNN model
            </div>
            <div style="margin-top:1.5rem;font-family:'Space Mono',monospace;font-size:0.7rem;color:#3a3a50">
                MODEL: KNN • DATASET: 45,234 DEVELOPERS • ACCURACY: 73.46%
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;font-family:'Space Mono',monospace;font-size:0.7rem;color:#3a3a50">
    DEVMATCH AI &nbsp;•&nbsp; KNN MODEL &nbsp;•&nbsp; STACK OVERFLOW SURVEY 2024 &nbsp;•&nbsp; 45,234 DEVELOPERS
</div>
""", unsafe_allow_html=True)

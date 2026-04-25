import streamlit as st
import pandas as pd
import re
import random
import time
from PyPDF2 import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Talent Scout Pro", page_icon="🎯", layout="wide")

# Professional CSS Styling
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #eee; }
    .stDataFrame { background-color: white; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- CORE AGENT LOGIC ---
class TalentScoutAgent:
    def __init__(self):
        self.skill_bank = ["python", "java", "react", "angular", "node", "aws", "azure", "docker", "kubernetes", 
                          "sql", "nosql", "machine learning", "ml", "ai", "data science", "tableau", "excel", "c++"]

    def extract_text(self, file):
        try:
            if file.name.endswith('.pdf'):
                reader = PdfReader(file)
                return " ".join([page.extract_text() or "" for page in reader.pages])
            elif file.name.endswith('.docx'):
                doc = docx.Document(file)
                return " ".join([p.text for p in doc.paragraphs])
        except Exception as e:
            return f"Error: {e}"

    def parse_jd(self, jd_text):
        jd_lower = jd_text.lower()
        found_skills = [skill for skill in self.skill_bank if skill in jd_lower]
        exp_match = re.search(r'(\d+)\+?\s*years?', jd_lower)
        experience = exp_match.group(1) if exp_match else "Not specified"
        seniority = "Senior" if any(x in jd_lower for x in ["senior", "sr", "lead", "architect"]) else "Junior/Mid"
        return {"skills": list(set(found_skills)), "seniority": seniority, "exp": experience}

    def calculate_match_score(self, jd_text, cv_texts, criteria):
        if not cv_texts: return [], []
        all_docs = [jd_text] + cv_texts
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(all_docs)
        vectors = vectorizer.toarray()
        jd_vec = vectors[0]
        cv_vecs = vectors[1:]
        semantic_scores = cosine_similarity([jd_vec], cv_vecs)[0]

        final_match_scores = []
        explanations = []

        for i, cv in enumerate(cv_texts):
            cv_lower = cv.lower()
            matched_skills = [s for s in criteria['skills'] if s in cv_lower]
            skill_score = (len(matched_skills) / len(criteria['skills'])) if criteria['skills'] else 0.5
            
            combined = (semantic_scores[i] * 0.5) + (skill_score * 0.5)
            final_match_scores.append(round(combined * 100, 1))
            
            missing = list(set(criteria['skills']) - set(matched_skills))
            explanations.append({"matched": matched_skills, "missing": missing})

        return final_match_scores, explanations

    def simulate_engagement(self):
        personas = [
            {"msg": "I'm very excited! I've been looking for a role involving these exact technologies.", "interest": random.randint(85, 100), "tag": "Highly Interested"},
            {"msg": "The role sounds great. I am open to a call, though I am also talking to other companies.", "interest": random.randint(60, 84), "tag": "Passive Seeker"},
            {"msg": "I'm happy in my current role, but for the right compensation, I'd consider a change.", "interest": random.randint(40, 59), "tag": "Low Interest"},
            {"msg": "I am looking for a fully remote position only. Does this fit?", "interest": random.randint(50, 75), "tag": "Conditional"},
            {"msg": "I've just started a new role elsewhere, but thanks for reaching out!", "interest": random.randint(0, 20), "tag": "Unavailable"}
        ]
        return random.choice(personas)

# --- APP FLOW ---
def main():
    agent = TalentScoutAgent()
    
    # 1. INITIALIZE SESSION STATE (The "Memory" of the app)
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'last_processed' not in st.session_state:
        st.session_state.last_processed = None

    st.title("🤖 AI-Powered Talent Scouting & Engagement Agent")
    st.caption("End-to-End Discovery, Unbiased Matching, and Automated Engagement Simulation")

    # SIDEBAR
    st.sidebar.header("📁 Data Ingestion")
    uploaded_cvs = st.sidebar.file_uploader("Upload Candidate CVs", accept_multiple_files=True, type=['pdf', 'docx'])
    
    st.sidebar.divider()
    st.sidebar.header("⚙️ Agent Strategy")
    vacancy_count = st.sidebar.slider("Number of vacancies to fill:", 1, 20, 3)
    match_weight = st.sidebar.slider("Weight for Technical Match Score:", 0.0, 1.0, 0.7)
    
    # MAIN AREA
    col_jd, col_crit = st.columns([2, 1])
    
    with col_jd:
        st.subheader("1. Job Description")
        jd_text = st.text_area("Paste the JD here:", height=250, placeholder="Required: Python, AWS, and 3+ years experience...")

    with col_crit:
        st.subheader("2. Extracted Requirements")
        if jd_text:
            criteria = agent.parse_jd(jd_text)
            st.write(f"**Level:** {criteria['seniority']}")
            st.write(f"**Exp:** {criteria['exp']} Years")
            st.write("**Target Skills:**")
            for s in criteria['skills']:
                st.markdown(f"- `{s}`")
        else:
            st.info("Agent is waiting for a JD...")

    # --- ACTION BUTTON ---
    if st.button("🚀 Run Scouting Agent"):
        if not uploaded_cvs or not jd_text:
            st.error("Missing Data: Provide JD and CVs.")
        else:
            with st.status("Agent performing scouting...") as status:
                cv_texts = [agent.extract_text(f) for f in uploaded_cvs]
                candidate_names = [f.name.split('.')[0] for f in uploaded_cvs]
                
                match_scores, explains = agent.calculate_match_score(jd_text, cv_texts, criteria)
                
                results = []
                for i in range(len(candidate_names)):
                    eng = agent.simulate_engagement()
                    # Calculate final score based on sidebar weights
                    interest_weight = 1.0 - match_weight
                    f_score = (match_scores[i] * match_weight) + (eng['interest'] * interest_weight)
                    
                    results.append({
                        "Shortlist": False,
                        "Candidate": candidate_names[i],
                        "Match Score": match_scores[i],
                        "Interest Score": eng['interest'],
                        "Final Fit Score": round(f_score, 1),
                        "Status": eng['tag'],
                        "Response": eng['msg'],
                        "Skills Found": ", ".join(explains[i]['matched']),
                        "Missing": ", ".join(explains[i]['missing'])
                    })

                # SAVE TO SESSION STATE
                st.session_state.results_df = pd.DataFrame(results).sort_values("Final Fit Score", ascending=False)
                st.session_state.last_processed = time.time()
                status.update(label="Scouting Complete!", state="complete")

    # --- PERSISTENT RESULTS SECTION ---
    # This runs every time a user clicks a checkbox because results_df is saved in state
    if st.session_state.results_df is not None:
        st.divider()
        df = st.session_state.results_df
        
        st.subheader(f"🏆 Top {vacancy_count} Matches")
        top_candidates = df.head(vacancy_count)
        cols = st.columns(min(len(top_candidates), 4))
        for idx, (_, row) in enumerate(top_candidates.iterrows()):
            with cols[idx % 4]:
                st.metric(label=row['Candidate'], value=f"{row['Final Fit Score']}%", delta=row['Status'])

        st.subheader("📋 Candidate Evaluation Dashboard")
        st.info("Check the 'Shortlist' box to finalize candidates for interview.")
        
        # INTERACTIVE DATA EDITOR
        edited_df = st.data_editor(
            df,
            column_config={
                "Shortlist": st.column_config.CheckboxColumn("Shortlist", default=False),
                "Match Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                "Interest Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
            },
            hide_index=True,
            use_container_width=True,
            key="data_editor_key" # Important for persistence
        )

        # Update session state with checkmarks
        st.session_state.results_df = edited_df

        # SHORTLIST ACTIONS
        shortlisted_final = edited_df[edited_df["Shortlist"] == True]
        if not shortlisted_final.empty:
            st.success(f"Selected {len(shortlisted_final)} candidates for the final shortlist.")
            st.download_button(
                "📥 Download Selected Shortlist CSV", 
                shortlisted_final.to_csv(index=False).encode('utf-8'), 
                "shortlist.csv", 
                "text/csv"
            )
            
        # DECISION LOGS
        st.subheader("🔍 Agent Intelligence Logs")
        lcol, rcol = st.columns(2)
        for i, row in df.iterrows():
            target = lcol if i % 2 == 0 else rcol
            with target.expander(f"Analysis for {row['Candidate']}"):
                st.write(f"**Found Skills:** {row['Skills Found'] if row['Skills Found'] else 'None'}")
                if row['Missing']: 
                    st.write(f"**Gap Analysis:** Missing {row['Missing']}")
                st.divider()
                st.write(f"**Agent Engagement:**")
                st.markdown(f"*\"{row['Response']}\"*")
                st.progress(row['Interest Score']/100, text=f"Interest Level: {row['Interest Score']}%")

if __name__ == "__main__":
    main()

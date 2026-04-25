# 🎯 AI-Powered Talent Scouting & Engagement Agent

This project is an AI-driven recruitment agent designed to automate the initial stages of hiring. It parses Job Descriptions, matches candidate CVs using semantic analysis, and simulates engagement outreach to assess interest.

## 🚀 Features
- **JD Parsing:** Dynamically extracts skills, seniority, and experience requirements.
- **Match Scoring:** A hybrid algorithm combining TF-IDF similarity and Hard Skill overlap.
- **Engagement Simulation:** Conducts simulated candidate outreach with persona-based responses.
- **Shortlisting:** Ranks candidates and allows recruiters to select top fits based on vacancy count.
- **Explainability:** Provides detailed "Decision Logs" for every candidate fit.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Processing:** Pandas, Python-Docx, PyPDF2
- **ML/NLP:** Scikit-Learn (TF-IDF & Cosine Similarity)

## 📋 How to Run
1. Install the requirements:
   ```bash
   pip install -r requirements.txt

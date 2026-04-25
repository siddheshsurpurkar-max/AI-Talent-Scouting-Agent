# 🎯 AI-Powered Talent Scouting Agent

## 📖 Project Overview
This AI agent automates the recruitment process by:
1. **Parsing** Job Descriptions to find key skills.
2. **Matching** CVs (PDF/Docx) against the JD using semantic AI.
3. **Simulating** candidate engagement to assess interest.
4. **Ranking** a final shortlist based on your available vacancies.

## 🛠️ How to Run
1. Install the required libraries:
   `pip install -r requirements.txt`
2. Start the Agent:
   `streamlit run app.py`

## 🧠 Agent Logic
- **Match Score:** Uses TF-IDF and Cosine Similarity to compare CV text to JD text.
- **Interest Score:** Analyzes candidate responses using sentiment and keyword triggers.
- **Final Score:** A weighted average (70% Tech Fit / 30% Interest).
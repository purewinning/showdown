# DK Showdown Builder – "Stochastic Banger" Framework (Streamlit)

This version is built to avoid the classic trap:
**highest-projected CPT + punt city**.

Key upgrades:
- Outcome-based CPT ranking option (from standings) OR projection/ceiling CPT ranking
- Punts control: cap low-salary plays + optional min UTIL projection
- Correlation rules: CPT + at least N teammates, plus optional bring-back
- Candidate generation + Monte Carlo proxy scoring for "first place probability"
- Final selection enforces exposure caps + diversity

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Cloud)
Push these files to GitHub and set `app.py` as the entry point.

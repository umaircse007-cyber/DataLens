# DataLens

An intelligent data dictionary agent that helps you actually understand what's in your dataset before you do anything with it.

Upload a CSV or Excel file and DataLens profiles every column, generates plain-English descriptions using Gemini and Groq, detects sensitive attributes and proxy columns, flags fairness risks with EU AI Act mappings, scores your dataset's ML readiness, and exports the whole thing as a PDF, Excel, or JSON report.

Built with FastAPI and Python. Currently under active development — working locally, deployment coming soon.

---

## Why we built this

Every data team has the same problem. Someone hands you a 40-column spreadsheet with headers like `rev_adj_v2`, `flag_1`, or `col_C`. Nobody documented what any of it means. The person who built it left six months ago. You waste hours just figuring out what the data is before you can do anything useful with it.

DataLens fixes that. Upload the file, get a full dictionary back in under 60 seconds.

---

## What it does

- **Column profiling** — data types, null rates, unique counts, top values, sample values for every column
- **AI-generated descriptions** — Gemini and Groq independently describe each column in plain English; findings are confidence-scored based on whether both models agree
- **Sensitive and proxy column detection** — flags gender, age, zip code, college tier, and similar columns that could cause bias or compliance issues if used in a model
- **Anomaly storytelling** — doesn't just flag outliers and nulls, explains what they likely mean in a business context
- **Relationship detection** — catches strongly correlated columns, redundant pairs (like `age` and `dob`), and derived columns (like `annual_salary = monthly_salary × 12`)
- **Fairness metrics** — demographic parity, disparate impact ratio with the 80% rule, statistical significance checks, counterfactual flip test
- **EU AI Act mapping** — maps detected issues to Article 10 and Article 13 where applicable
- **ML readiness score** — scores the dataset out of 100 across null rates, class imbalance, sensitive column presence, duplicate rows, and dataset size
- **Query suggestions** — suggests 5 analytical questions you could actually answer with this data, with pandas and SQL one-liners for each
- **Export** — PDF audit report, Excel dictionary, and JSON for integration with other tools
- **Audit history** — every run is saved locally so you can track how a dataset changes over time

---

## Tech stack

| Layer | What we used |
|---|---|
| Backend | Python 3, FastAPI, Uvicorn |
| AI scanning | Google Gemini 2.5 Flash, Groq |
| Data + stats | pandas, NumPy, SciPy, scikit-learn |
| Report generation | ReportLab (PDF), openpyxl (Excel) |
| Security | cryptography (Fernet encryption) |
| Frontend | Vanilla HTML, CSS, JavaScript |

---

## Project structure

```
DataLens/
├── frontend/          # Static browser interface
├── routes/            # FastAPI route handlers
├── services/          # Column profiling, AI scan, fairness,
│                      # anomaly, relationship, export, security
├── data/
│   ├── uploads/       # Uploaded files — encrypted, git-ignored
│   └── reports/       # Generated exports — git-ignored
├── main.py
├── requirements.txt
└── .env.example
```

---

## Setup

**1. Clone and create a virtual environment**

```powershell
git clone https://github.com/azim-iqbal/DataLens.git
cd DataLens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**2. Install dependencies**

```powershell
pip install -r requirements.txt
```

**3. Set up environment variables**

```powershell
copy .env.example .env
```

Open `.env` and fill in:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
FILE_ENCRYPTION_KEY=generate_with_command_below
ENABLE_HTTPS_REDIRECT=false
```

Generate your encryption key (run once, paste the output into `.env`):

```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

The app runs without API keys using local fallback logic — useful for testing and demos when you don't want to burn API quota.

---

## Run locally

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Security

Uploaded files are encrypted on disk using Fernet symmetric encryption and automatically deleted one hour after processing. Only column headers and a small sample of values are sent to Gemini and Groq — full rows never leave the server. The encryption key lives in `.env` and is never committed to the repo.

---

## What's not done yet

Being honest about where things are:

- Deployment config (Render/Railway) — coming once the core is stable
- Natural language chat interface — ask questions about your dataset in plain English
- Schema evolution detection — upload two versions and see what changed
- Mobile-responsive UI — currently desktop only
- Column-level inline editing persisted across sessions

---

## Privacy

The following are excluded from Git:

```
data/uploads/       encrypted dataset files
data/reports/       generated PDF and Excel exports
*.db                SQLite audit history
*.log               application logs
.env                API keys and secrets
.venv/              virtual environment
```

---

## Contributing

Open an issue before starting a PR so we can discuss the approach. This is a hackathon project so things move fast and the codebase changes frequently.

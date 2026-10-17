# Secure EHR Insight – Clinical Validator

A zero-trust, retrieval-augmented chat assistant for Electronic Health Records (EHR). Clinicians pick a patient, ask questions in plain language, and get answers grounded in that patient's records — while personal health information (PHI) is redacted **before** anything reaches the LLM, and requests for diagnoses or prescribing advice are blocked by guardrails.

Built on MIMIC-IV clinical data, PostgreSQL + pgvector, a biomedical embedding model, Microsoft Presidio, and NVIDIA NeMo Guardrails.

---

## Key features

- **Patient-scoped semantic search** – questions are embedded with `NeuML/bioclinical-modernbert-base-embeddings` (768-dim) and matched against that patient's records only, using pgvector cosine distance.
- **PII/PHI redaction (Presidio)** – retrieved context is scrubbed for names, phone numbers, emails, SSNs, locations, organisations and dates before it is sent to the LLM. Includes a catch-all SSN pattern and a hospital-name deny list.
- **Clinical guardrails (NeMo Guardrails)** – the assistant answers questions about patient *history* but refuses to diagnose, prescribe, or recommend medication changes.
- **Output disclaimer** – every UI response is tagged as an AI-generated summary not for diagnostic use.
- **Multi-turn chat** – conversation history is passed through the guardrails on every turn.

---

## Architecture

```text
 Browser
    │
    ▼
 Streamlit UI (:8501)  ── select patient, chat
    │  HTTP
    ▼
 FastAPI (:8000)
    │ 1. Embed question ─────────── BioClinical ModernBERT (local)
    │ 2. Vector search (per patient) ─ PostgreSQL + pgvector
    │ 3. Redact PHI in context ───── Microsoft Presidio (spaCy en_core_web_lg)
    │ 4. Policy check + generation ─ NeMo Guardrails ──► DeepSeek (OpenAI-compatible API)
    ▼
 Answer (+ disclaimer)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| API | FastAPI, Uvicorn |
| Database | PostgreSQL 14 + pgvector, SQLAlchemy, psycopg 3 |
| Embeddings | sentence-transformers, `NeuML/bioclinical-modernbert-base-embeddings` |
| PHI redaction | Presidio Analyzer / Anonymizer, spaCy `en_core_web_lg` |
| Guardrails | NeMo Guardrails (Colang rails) |
| LLM | DeepSeek `deepseek-chat` via OpenAI-compatible endpoint |
| Data | MIMIC-IV cleaned medical transcripts (Kaggle) |

---

## Project structure

```text
.
├── data/
│   ├── MIMIC_IV_Trasncript.csv      # source dataset (~232k rows)
│   └── data.txt                     # dataset & HIPAA reference links
├── scripts/                         # run in order to build the database
│   ├── 01_ingest_baseline_data.py   # create table, load CSV
│   ├── 02_verify_ingestion.py       # row count + sample check
│   ├── 03_apply_vector_schema.py    # add clinical_embedding vector(768)
│   ├── 04_generate_embeddings.py    # embed records and store vectors
│   ├── 05_test_vector_search.py     # semantic search smoke test
│   └── 06_test_guardrails.py        # allowed vs. blocked prompt test
├── src/
│   ├── api/main.py                  # FastAPI service
│   ├── ui/app.py                    # Streamlit chat UI
│   ├── database/schema.sql          # patient_encounters table
│   ├── pii_redaction/presidio_service.py
│   └── guardrails/
│       ├── config.yml               # LLM + embedding model config
│       └── rails.co                 # Colang rules (no medical advice)
├── instructor_notes/                # setup notes & AWS EC2/Docker guide
└── requirements.txt
```

---

## Getting started

### Prerequisites

- Python 3.12
- PostgreSQL 14+ with the [pgvector](https://github.com/pgvector/pgvector) extension
- A DeepSeek API key (or another OpenAI-compatible endpoint)
- ~2–3 GB disk for PyTorch, the embedding model and the spaCy model

### 1. Clone and install

```bash
git clone https://github.com/nimowhyca/Secure-EHR-Insight-Clinical-Validator.git
cd Secure-EHR-Insight-Clinical-Validator

# with uv (recommended)
uv venv --python 3.12 .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install tqdm                # used by 04_generate_embeddings.py

# or with plain pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt tqdm
```

### 2. Prepare PostgreSQL

```sql
CREATE DATABASE ehr_db;
CREATE USER ehr_app WITH PASSWORD '<choose-a-strong-password>';
GRANT ALL PRIVILEGES ON DATABASE ehr_db TO ehr_app;
\c ehr_db
GRANT ALL ON SCHEMA public TO ehr_app;
-- as a superuser:
CREATE EXTENSION IF NOT EXISTS vector;
```

On Ubuntu, pgvector can be installed with `sudo apt install postgresql-14-pgvector`. See `instructor_notes/instruction.txt` for a full EC2 walkthrough.

### 3. Configure environment

Create a `.env` file in the project root (it is git-ignored):

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ehr_db
DB_USER=ehr_app
DB_PASSWORD=<your-password>

OPENAI_API_KEY=<your-deepseek-api-key>
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_BASE=https://api.deepseek.com/v1
```

If `DB_HOST` is unset, the API falls back to `POSTGRES_URL` (default `postgresql+psycopg://postgres:password@localhost:5432/clinical_db`).

### 4. Build the database

Run from the project root:

```bash
python scripts/01_ingest_baseline_data.py   # load MIMIC-IV CSV into patient_encounters
python scripts/02_verify_ingestion.py       # sanity check
python scripts/03_apply_vector_schema.py    # add vector(768) column
python scripts/04_generate_embeddings.py    # embed records
python scripts/05_test_vector_search.py     # optional: test semantic search
python scripts/06_test_guardrails.py        # optional: test guardrails
python src/pii_redaction/presidio_service.py  # optional: see redaction demo
```

> `04_generate_embeddings.py` embeds only **100 records** by default (`demo_limit`, `batch_size = 2`) so the demo runs quickly. Raise these values to vectorise more of the dataset. Only patients with embeddings appear in the UI.

### 5. Run the app

In two terminals:

```bash
uvicorn src.api.main:app --reload          # API on http://localhost:8000
streamlit run src/ui/app.py                # UI on http://localhost:8501
```

The API loads Presidio, NeMo Guardrails and the embedding model at startup, so the first boot takes a little while. Interactive API docs are at `http://localhost:8000/docs`.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/patients` | Distinct `subject_id`s that have embeddings |
| `POST` | `/api/v1/chat` | Patient-scoped RAG chat (vector search → redaction → guardrails → LLM) |
| `POST` | `/api/v1/clinical-query` | Demo endpoint using a hard-coded mock record to show redaction + guardrails |

**`POST /api/v1/chat`**

```json
{
  "patient_id": "10000032",
  "messages": [
    { "role": "user", "content": "What medications were prescribed upon discharge?" }
  ]
}
```

```json
{ "status": "success", "llm_response": "..." }
```

---

## Try these questions

Answered from the record:

- What medications were prescribed to this patient upon discharge?
- Can you summarise the last few reports of this patient?
- What liver-related diagnoses are noted in the patient's file?
- Was the patient admitted urgently or routinely?

Should be refused by the guardrails:

- Based on the positive peritoneal fluid culture, what broad-spectrum antibiotic should I start the patient on?
- Should I increase the patient's dosage?

---

## How the safety layers work

**PHI redaction** – `ClinicalPIIRedactor` runs Presidio over the retrieved context with a 0.4 confidence threshold and replaces detected entities with placeholders such as `<PERSON>` or `<DATE_TIME>`. Only the redacted text is sent to the LLM.

**Guardrails** – `src/guardrails/rails.co` defines two intents: *asking about patient history* (allowed) and *asking for medical advice* (diagnosis, prescribing, dosage changes). The latter triggers a fixed refusal pointing the user to the attending physician. Intent matching uses the same BioClinical ModernBERT embeddings.

**Patient scoping** – every vector search is filtered by `subject_id`, so the model only sees the selected patient's records.

---

## Deployment

A step-by-step guide for running FastAPI and Streamlit in a single Docker container on AWS EC2 (only port 8501 exposed publicly, database on RDS or a separate host) is in [`instructor_notes/AWS_EC2_Docker_Deployment_Guide.md`](instructor_notes/AWS_EC2_Docker_Deployment_Guide.md).

---

## Known limitations

- `tqdm` is used by the embedding script but is not listed in `requirements.txt`.
- The Streamlit UI calls the API at a hard-coded `http://localhost:8000`.
- `/api/v1/clinical-query` uses mock data and is intended for demos only.
- Presidio's general-purpose recognisers can miss domain-specific identifiers (e.g. MRNs) and may over-redact clinical terms; review before any real-world use.
- The guardrails cover a single policy (no medical advice); prompts outside both defined intents are passed to the LLM as normal.

---

## Data & compliance

The dataset comes from [MIMIC-IV cleaned medical transcripts on Kaggle](https://www.kaggle.com/datasets/isaacritharson/mimic-iv-cleaned-medical-transcripts). MIMIC-IV is de-identified, but its use is governed by the PhysioNet credentialed data use agreement — check the licence terms before redistributing the CSV.

This project is an educational prototype. It is **not** a certified medical device, is not HIPAA-certified, and must not be used for diagnosis or treatment decisions.

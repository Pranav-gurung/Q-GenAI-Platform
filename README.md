# AI-powered industrial defect inspection - Module A (Synthetic Data Engine) + Module C (GenAI RCA)

| Module | Owner | In this repo |
|---|---|---|
| A - Synthetic Data Engine | you | `module_a/` |
| B - Defect Detection | teammate | consumes Module A's dataset, calls Module C's `POST /rca` |
| C - GenAI Root-Cause Analysis (RAG) | you | `module_c/` |
| D - Security & Audit | teammate | plugs into the audit/auth hooks (`docs/MODULE_D_AUDIT.md`) |

Start here: [`docs/HANDOFF_MODULE_B.md`](docs/HANDOFF_MODULE_B.md) (dataset + API), [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md),
[`docs/MODULE_D_AUDIT.md`](docs/MODULE_D_AUDIT.md), [`dataset_README.md`](dataset_README.md).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # full stack (FastAPI, PyMuPDF, ChromaDB, sentence-transformers, pytest...)
cp .env.example .env                         # then edit; export the variables (or use your process manager)
```
Minimum for Module A only: `numpy`, `Pillow`. Module C core also runs with just `numpy` + `pypdf`; FastAPI/Pydantic are needed only for the HTTP service.

## Module A
```bash
# 1) Try the pipeline on a synthetic TEST fixture (not industrial data)
python -m module_a.prepare_dataset --use-fixture --out-root data

# 2) Real, lawfully obtained dataset (CSV or JSON annotations; see module_a/ingest.py docstring for columns)
python -m module_a.validate_dataset --image-dir /data/images --annotations /data/ann.csv --classes scratch,crack --report data/reports/validation.json
python -m module_a.prepare_dataset --image-dir /data/images --annotations /data/ann.csv --classes scratch,crack \
    --source-name "Vendor X" --license-notes "internal use only, contract #..." --formats coco,yolo --out-root data
```
Pipeline: ingest -> validate (errors stop the run unless `--quarantine-invalid`, which excludes AND reports bad images) ->
**split first** (group + duplicate aware) -> augment **train only** -> export adapters -> manifest + README.
Outputs (per version `ds-<hash>`; version id is a hash of file hashes + config, so identical inputs give the same id):
```
data/splits/<v>/{train,val,test}/images/   annotations/<v>/canonical_<split>.json   processed/<v>/{coco,yolo,anomaly}/
data/manifests/<v>.json                    reports/<v>/{validation,split_report,augmentation_log}.json, DATASET_README.md
data/feedback/feedback.jsonl               (operator corrections; see module_a/feedback.py)
```
Operator feedback: `FeedbackStore.submit -> review (different person) -> export_approved -> apply_approved`; pending/rejected items are never applied, and applying produces new records for a **new** dataset version (it never edits an existing one).

## Module C
```bash
# put authorised PDFs/TXT + <file>.meta.json sidecars in documents/approved/  (see documents/approved/README.md)
python -m module_c.retrieval --query "scratch on machined housing"     # ingest + test retrieval (no LLM needed)
export RCA_LLM_PROVIDER=anthropic RCA_LLM_MODEL=<model-id> ANTHROPIC_API_KEY=...   # or openai_compatible + OPENAI_API_KEY
uvicorn module_c.api:create_app --factory --port 8001
curl -s localhost:8001/health
curl -s -X POST localhost:8001/rca -H 'content-type: application/json' -d '{"inspection_id":"QC-0001","image_id":"IMG-0001","model_version":"v1","defects":[{"class":"scratch","confidence":0.94,"bbox":[120,80,260,190]}]}'
```
Flow: validate -> per defect class build a defect-specific query -> retrieve (approved + active + access-filtered) -> relevance threshold **and**
class-term gate -> if no evidence: `insufficient_evidence`, no LLM call -> else LLM (JSON only) -> strict validation (schema, chunk ids must be ones supplied,
unsupported "supported_by_document" claims downgraded) -> server assembles response (observations, citations come from detector input / index, never from the LLM) -> audit.

Swapping components: embedder (`RCA_EMBEDDING_BACKEND/MODEL`), vector store (`RCA_VECTOR_STORE`), LLM (`RCA_LLM_PROVIDER/MODEL`) are behind small interfaces in
`vectorstore.py` / `llm.py`; `rca.py` does not import any vendor.

## Tests
```bash
python -m unittest discover -s tests -t . -v      # stdlib runner, no extra deps
pytest -q                                          # tests are unittest-style and should run unchanged (pytest not installed where this was built, so not verified)
```
LLM calls are always stubbed; documents in tests are synthetic fixtures labelled as such.
API (HTTP) tests skip automatically when `fastapi`/`httpx` are missing.

## Known limitations (summary)
- **FastAPI layer, ChromaDB store, SentenceTransformer embedder and PyMuPDF extractor are written but were not executed** in the environment where this was built (no network, packages unavailable). They need a first run with `requirements-dev.txt`; the API tests exist and will exercise `api.py`. The `JsonVectorStore`, `HashingEmbedder` and pypdf paths are what the automated tests actually cover.
- The default `hashing` embedder is a **lexical** baseline (with crude stemming), not semantic. Use `sentence_transformers` for real use and **calibrate `RCA_MIN_SCORE`** (and `RCA_CLASS_ALIASES`) on your own documents; thresholds here were only sanity-checked on synthetic fixtures.
- The class-term gate favours precision: a relevant passage that never uses the class word or a configured alias is not evidence (result: `insufficient_evidence`).
- The server verifies citations exist and were supplied; it cannot verify that a passage truly *entails* the LLM's wording. Human review is required.
- No OCR: scanned PDFs are reported `ocr_required`. Chunks never cross page boundaries. Section headings are a heuristic.
- Splits depend on accurate `group_id` metadata; perceptual hashes (dHash) miss rotated/cropped/relit copies. Greedy splitting is not optimal on tiny datasets.
- Rotation augmentation inflates boxes of elongated defects; flips are off unless you enable them; masks/segmentation are not supported (explicit errors).
- No evidence is claimed that augmentation improves detection performance.
- Audit and auth are hooks (see `docs/MODULE_D_AUDIT.md`); the system must not be called secure until Module D's controls exist and are tested.

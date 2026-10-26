# Approved documents

Place ONLY documents you are authorised to use (BIS/ISO standards you have licensed, factory SOPs) here.
This repository contains no real standards and no real SOPs, and nothing is downloaded automatically.

For each `file.pdf` add `file.pdf.meta.json`:
```json
{"document_id": "SOP-001", "title": "Component Handling SOP", "version": "2.0",
 "access_level": "internal", "license_note": "licensed copy #123, internal use only"}
```
`access_level` is `public | internal | restricted`; only levels in `RCA_ALLOWED_ACCESS_LEVELS` are retrievable.
Without a sidecar the id/title are inferred from the filename and the version is `unspecified` (flagged at ingestion).
Documents are git-ignored; remove a file (or run `python -m module_c.retrieval --deactivate SOP-001`) to stop it being used.

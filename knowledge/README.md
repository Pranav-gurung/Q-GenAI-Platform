# knowledge/: SOP knowledge base for RAG (Part 5)

> **DEMO / SYNTHETIC.** Every document here was written for this project as demo content. None of them is an
> official procedure of Blinkit, Zepto, Swiggy Instamart, Amazon or any other company, and they are not
> food-safety or legal guidance. Replace them with your own approved SOPs before real use.

One Markdown file per document, with YAML front matter and `## ` sections (this README is ignored by ingestion):

```markdown
---
document_id: PACKING-SOP          # stable id used in citations (A-Z, 0-9, '-')
title: Packing and Sealing SOP
stage: PACKING                    # RECEIVING | STORAGE | PICKING | PACKING | DISPATCH | DELIVERY | GENERAL
document_type: SOP                # SOP | CHECKLIST
version: demo-1.0
defect_types: [LEAKING_PACKAGE]   # platform DefectType values the document addresses
provenance: DEMO_SYNTHETIC        # DEMO_SYNTHETIC | APPROVED (reviewed by your QA team)
---
# Packing and Sealing SOP

## Seal Inspection
- Inspect sealing equipment and verify seal integrity before releasing affected inventory.
```

Each `## ` section becomes one chunk (`PACKING-SOP#seal-inspection`), the unit that is retrieved and cited.
Validate and rebuild the index after any change:

```bash
python -m backend.rag.ingest --validate-only   # exit 1 on invalid metadata / duplicates / empty KB
python -m backend.rag.ingest                   # writes data/rag/index (git-ignored)
```

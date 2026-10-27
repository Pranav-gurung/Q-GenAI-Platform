"""Run the canonical demo against a RUNNING server and print what each layer produced.

    python -m scripts.demo_canonical [--base-url http://localhost:8000] [--images data/cv/synthetic-v1]

RECEIVING, STORAGE, PICKING pass; PACKING shows a leak. Then GET /rca (GenAI + SOP grounding), an operator review,
and the operations analytics. Standard library only; it calls the public API exactly like the UI does.

* With --images (the SYNTHETIC CV dataset) and CV_ENGINE=yolov8, the model sees real (synthetic) defect images.
* Without it, plain generated images are uploaded and, where the pipeline has no automated decision (stub CV), the
  operator verdict (PASS / FAIL + LEAKING_PACKAGE) is recorded instead - the Part 3 operator-feedback path.
All data is SYNTHETIC / DEMO. The LLM is whatever the server is configured with (LLM_PROVIDER=mock needs no key).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

STAGES = ["RECEIVING", "STORAGE", "PICKING", "PACKING"]


class Api:
    def __init__(self, base: str, api_key: str | None) -> None:
        self.base, self.api_key = base.rstrip("/") + "/api", api_key

    def call(self, method: str, path: str, body: bytes | None = None, ctype: str = "application/json"):
        req = urllib.request.Request(self.base + path, data=body, method=method, headers={"content-type": ctype})
        if self.api_key:
            req.add_header("X-API-Key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            err = json.loads(e.read() or b"{}").get("error", {})
            raise SystemExit(f"{method} {path} -> {e.code} {err.get('code')}: {err.get('message')}") from None

    def json(self, method: str, path: str, payload: dict):
        return self.call(method, path, json.dumps(payload).encode())

    def upload(self, iid: str, name: str, data: bytes, ctype: str):
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        return self.call("POST", f"/inspections/{iid}/image", body, f"multipart/form-data; boundary={boundary}")


def generated_png(seed: int) -> bytes:
    from PIL import Image  # installed with the backend
    img = Image.new("RGB", (320, 240), (60 + seed * 30, 140, 90))
    for x in range(0, 320, 16):  # texture so the basic quality gate (blur check) accepts it
        for y in range(240):
            img.putpixel((x, y), (250, 250, 250))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def dataset_images(root: Path) -> tuple[list[Path], Path]:
    labels = sorted((root / "labels" / "test").glob("*.txt"))
    clean = [p for p in labels if not p.read_text().strip()][:3]
    leak = next(p for p in labels if p.read_text().strip() and {ln.split()[0] for ln in p.read_text().splitlines()} == {"2"})
    img = lambda p: root / "images" / "test" / f"{p.stem}.jpg"  # noqa: E731
    return [img(p) for p in clean], img(leak)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--images", type=Path, help="synthetic CV dataset root (e.g. data/cv/synthetic-v1)")
    p.add_argument("--api-key", help="X-API-Key, if the server sets API_KEY")
    args = p.parse_args(argv)
    api = Api(args.base_url, args.api_key)
    health = api.call("GET", "/health")
    print(f"server {health['status']}: CV={health['pipeline_modules']['cv_engine']['name']} "
          f"LLM={health['llm']['provider'] if health['llm']['enabled'] else 'off'} RAG={health['rag']['status']}")
    item = f"DEMO-CANON-{int(time.time())}"
    clean, leak = dataset_images(args.images) if args.images else (None, None)

    for i, stage in enumerate(STAGES):
        iid = api.json("POST", "/inspections", {"store_id": "DS-BLR-001", "sku_id": "SKU-MILK-500ML", "stage": stage,
                                                "item_id": item, "operator_id": "OP-DEMO"})["inspection_id"]
        if args.images:
            path = leak if stage == "PACKING" else clean[i]
            res = api.upload(iid, path.name, path.read_bytes(), "image/jpeg")
        else:
            res = api.upload(iid, f"{stage.lower()}.png", generated_png(i), "image/png")
        defects = sorted({d["defect_type"] for d in ((res.get("result") or {}).get("cv") or {}).get("detections", [])})
        source = "model"
        if res["decision"] is None:  # no automated decision (e.g. stub CV): the operator records the verdict
            verdict = {"outcome": "FAIL", "defect_types": ["LEAKING_PACKAGE"]} if stage == "PACKING" else {"outcome": "PASS"}
            api.json("POST", f"/inspections/{iid}/feedback", {"operator_id": "OP-DEMO", **verdict})
            source, defects = "operator verdict", verdict.get("defect_types", [])
            res["decision"] = verdict["outcome"]
        print(f"  {stage:<10} {res['decision']:<6} defects={defects or '-'} ({source})")

    pipe = api.call("GET", f"/inspections/{iid}/pipeline")
    for a in pipe["defect_analyses"]:
        print(f"pipeline: {a['defect_type']} first observed at {a['first_observed_stage']} (last known good "
              f"{a['last_known_good_stage']}), evidence {a['evidence_level']}, {a['classification']}")
    rca = api.call("GET", f"/inspections/{iid}/rca")
    g = rca["generation"]
    print(f"GenAI RCA: {rca['status']} ({g.get('provider')}/{g.get('model')}"
          f"{', ' + g['error_code'] if g.get('error_code') else ''})")
    for d in rca["defects"]:
        for c in d["likely_causes"]:
            print(f"  hypothesis: {c['hypothesis']}")
    rag = rca.get("rag") or {}
    print(f"SOP grounding: {rag.get('status', 'disabled')}")
    for a in rag.get("grounded_actions", [])[:3]:
        print(f"  [SOP] {a['action']}  <- {a['source_document']} / {a['source_section']} / {a['source_version']}")
    review = {"operator_id": "OP-DEMO", "outcome": "FAIL", "rca_useful": True, "action_useful": True,
              "comment": "Recommendation matches the packing SOP."}
    api.json("POST", f"/inspections/{iid}/feedback", review)
    print("operator review: recorded for future system improvement")
    o = api.call("GET", "/analytics/overview")
    i = o["inspections"]
    print(f"analytics: {i['total']} inspections, observed defect rate {i['observed_defect_rate']}, top hotspot "
          f"{o['hotspots'][0]['stage'] if o['hotspots'] else '-'}"
          + (f", investigate first: {o['investigation']['stage']}" if o["investigation"] else ""))
    print(f"open the UI: {args.base_url}/  (inspection {iid}; Operations Intelligence tab for the dashboard)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# End-to-end API smoke test against a running server.  Usage: scripts/smoke_test.sh [base_url]
set -euo pipefail
BASE="${1:-http://localhost:8000}/api"
IMG="$(mktemp --suffix=.png)"
trap 'rm -f "$IMG"' EXIT
python3 -c "from PIL import Image; Image.new('RGB',(64,48),(200,120,40)).save('$IMG')"

echo "== health";            curl -fsS "$BASE/health"; echo
echo "== create inspection"
INS=$(curl -fsS -X POST "$BASE/inspections" -H 'content-type: application/json' \
  -d '{"store_id":"DS-BLR-001","sku_id":"SKU-MILK-500ML","stage":"PACKING","order_id":"ORD-100245","operator_id":"OP-17"}')
echo "$INS"
ID=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['inspection_id'])" "$INS")
echo "== upload image + run pipeline ($ID)"
curl -fsS -X POST "$BASE/inspections/$ID/image" -F "file=@$IMG;type=image/png" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('status:', d['status'], '| decision:', d['decision']); [print('  ', h['event_type'], h['from_state'], '->', h['to_state'], h['module_status'] or '') for h in d['history']]"
echo "== feedback";          curl -fsS -X POST "$BASE/inspections/$ID/feedback" -H 'content-type: application/json' \
  -d '{"operator_id":"OP-17","outcome":"FAIL","defect_types":["LEAKING_PACKAGE"],"suspected_stage":"PICKING"}'; echo
echo "== get";               curl -fsS "$BASE/inspections/$ID" | python3 -c "import json,sys; d=json.load(sys.stdin); print('status:', d['status'], '| decision:', d['decision'])"
echo "== invalid request (expect 422)"; curl -sS -X POST "$BASE/inspections" -H 'content-type: application/json' -d '{"store_id":"DS-BLR-001","stage":"TELEPORT"}'; echo

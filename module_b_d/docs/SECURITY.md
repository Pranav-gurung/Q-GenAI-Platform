# Q-GenAI — Security & Compliance Design (Section D)

## 1. Authentication
* **Passwords:** Argon2id (`argon2-cffi` defaults) with a random salt per hash. Hashes are upgraded
  on login if the parameters change. Policy: 10–128 characters with letters and digits.
* **Tokens:** HS256 JWTs signed with `QGENAI_JWT_SECRET_KEY`. The app refuses to start if the key
  is shorter than 32 characters. Claims: `sub`, `iss`, `iat`, `nbf`, `exp` (default 30 min), `jti`, `ver`.
  * Decoding pins the algorithm list, so `alg=none` and algorithm confusion are rejected.
  * **The role is never read from the token.** Every request reloads the user from the database.
  * Logout stores the `jti` in `revoked_tokens` until it expires.
  * Disabling a user or changing their role, site or password increments `token_version`. That
    instantly invalidates all of their existing tokens.
* **Unknown users** are checked against a dummy hash, so response time does not reveal valid usernames.
  The error message is always the same: "Incorrect username or password."
* **Rate limiting:** 5 failures per 5 minutes per (client IP, username) returns HTTP 429 with
  `Retry-After`. This is in-memory and per process. For multiple workers, use Redis or nginx `limit_req`.
* **No default credentials.** The first admin is created explicitly with `python -m scripts.create_admin`
  (interactive prompt or env vars). The script refuses to run if an admin already exists.

## 2. Authorization (RBAC)
Permissions are defined once in `app/services/authorization.py` and enforced by FastAPI dependencies
(`require_permission`) on the server. The UI only hides buttons for convenience.

| Permission | Operator | Quality manager | Admin |
|---|:-:|:-:|:-:|
| Run inspection | ✔ | ✔ | ✘ (separation of duties) |
| View own inspections | ✔ | ✔ | — |
| View site inspections | ✘ | ✔ | — |
| View all inspections | ✘ | ✘ | ✔ |
| Threshold outside operator range | ✘ | ✔ (reason required) | ✘ |
| Record review decision / override | ✘ | ✔ (same site, not own inspection) | ✘ |
| View policy | ✔ | ✔ | ✔ |
| Edit policy | ✘ | ✘ | ✔ |
| Manage users / models | ✘ | ✘ | ✔ |
| View audit log | ✘ | own site | all |
| Verify audit chain / checkpoints | ✘ | ✘ | ✔ |

**IDOR protection:** every lookup by ID goes through `get_visible_inspection()`. An inspection
outside the caller's scope returns **404**, not 403, so IDs cannot be probed. List endpoints apply the
same scope as a SQL filter.

**Mass-assignment protection:** request schemas use `extra="forbid"`, so fields like `is_superuser`
are rejected with 422. Only admins can reach `PATCH /admin/users/{id}`, and admins cannot demote
or disable themselves.

## 3. Audit log: SHA-256 hash chain

```
hash_n = SHA-256( b"qgenai-audit-v1\n" ||
                  canonical_json({seq, event_id, timestamp, actor_id, actor_role, action,
                                  target_type, target_id, metadata_json, prev_hash: hash_{n-1}}) )
hash_0 (genesis prev_hash) = "000…0" (64 zeros)
```
`canonical_json` sorts keys, uses compact separators and UTF-8, and rejects NaN/Infinity, so
every event has exactly one byte representation. The timestamp and metadata are stored as the
exact strings that were hashed, so database type conversions cannot change them.

**What verification detects** (`POST /admin/audit-logs/verify`), stopping at the first failure:
| Tampering | Detected as |
|---|---|
| Field edited (metadata, actor, action, timestamp …) | "Content altered" at that event |
| Event deleted from the middle | "Sequence gap" at the next event |
| Event inserted | sequence gap or broken link |
| Events reordered | content altered (`seq` is part of the hash) |
| One event edited and its own hash recomputed | "Broken link" at the next event |
| Tail events deleted | **only with a checkpoint** |
| Whole chain rewritten with all hashes recomputed | **only with a checkpoint** |

### ⚠️ Limitation: tamper-evident, not immutable
The chain lives in the same database as the data. An attacker with write access to the database
can rewrite events and recompute every following hash, and the chain will verify again. The test
`test_full_rewrite_is_NOT_detected_without_checkpoint__documented_limitation` demonstrates this on
purpose. SHA-256 does not make records immutable. It makes unauthorized changes **detectable**
for anyone who cannot also recompute the chain unnoticed.

### Stronger protection (implemented as an option): signed checkpoints
`POST /admin/audit-logs/checkpoint` returns `{seq, hash, created_at, signature}`, where
`signature = HMAC-SHA256(QGENAI_AUDIT_CHECKPOINT_KEY, canonical_json(seq, hash, created_at))`.
* Store checkpoints **outside** the application database: another server, write-once (WORM) or
  object-lock storage, an email to the quality head, or a printout.
* Keep the checkpoint key out of the database. Ideally the database administrator does not have it.
* Verifying with a checkpoint detects truncation and full-chain rewrites up to that checkpoint,
  and rejects forged checkpoints.

Possible further hardening (not implemented):
* Asymmetric signatures (Ed25519) so verifiers do not need the signing key.
* Automatic periodic checkpoints pushed to a separate service.
* An append-only database role (INSERT-only grants on `audit_events`) in PostgreSQL.
* RFC 3161 trusted timestamps.

Blockchain is not used. A single organisation has no consensus problem that it would solve.

### What is logged
Successful, failed and blocked logins; logout; inspection requested, rejected, completed and result
created; threshold overrides; manual overrides; policy initialisation and updates; user creation,
role changes and updates; model registration, approval and activation; audit verification and
checkpoints; admin bootstrap.

### What is never logged
Passwords, password hashes, JWTs and API keys. `sanitize_metadata` redacts keys that match
`password|secret|token|authorization|api_key|cookie|credential|…hash` and caps strings at 500
characters. Images are never logged, only their SHA-256. Failed logins record only the attempted
username and client IP.

## 4. Inspection record integrity
* The original AI output (`ai_decision`, `ai_decision_reasons`, `defects_json`, model version and
  SHA-256, policy version, threshold) is written once and never updated.
* Human decisions are separate `inspection_overrides` rows: actor, role, old value, new value,
  mandatory reason, and timestamp. Each one is also hash-chained in the audit log.
* `final_decision` only reflects the latest human decision. Reports show both values.
* Reports carry a disclaimer: model confidence is not defect severity, and the report is **not**
  a BIS/ISO conformity certificate.

## 5. Application security checklist
| Control | Implementation |
|---|---|
| Input validation | Pydantic schemas (`extra="forbid"`, ranges, regex), Form/Query limits |
| Upload limits | `QGENAI_MAX_UPLOAD_MB` (reads at most limit+1 bytes), pixel cap, JPEG/PNG magic-byte check, `Image.verify()` |
| Safe file handling | Stored files are named `<server-uuid>.png`; weight files must be plain names inside `models/` (rejects `../`) |
| No arbitrary file access | No endpoint accepts a path; there are no static file mounts |
| CORS | Explicit allow-list from env; wildcard refused in production; no credentials |
| Error handling | Generic 500 with a reference ID; no stack traces or paths to clients |
| Debug | `QGENAI_DEBUG=false` by default; refuses to start in production with debug on |
| Security headers | `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Cache-Control: no-store` |
| Secrets | `.env` (git-ignored); `.env.example` holds placeholders only |

## 6. Deployment hardening
* Put a reverse proxy (nginx or Caddy) with TLS in front and run `uvicorn --proxy-headers`.
  Only expose the proxy; bind the API to 127.0.0.1.
* Use PostgreSQL: `QGENAI_DATABASE_URL=postgresql+psycopg://…` and `pip install psycopg[binary]`.
  Manage the schema with Alembic. Give the app user INSERT/SELECT only on `audit_events`.
  Serialize chain appends with `SELECT … FOR UPDATE` on a chain-head row or
  `pg_advisory_xact_lock`. The current in-process lock plus the `seq` primary key are enough
  for a single process.
* Keep the JWT secret and checkpoint key in a secrets manager or OS keyring. Rotate the JWT secret
  if you suspect a leak (this logs everyone out).
* Back up the DB and checkpoints separately. Schedule a daily verification.

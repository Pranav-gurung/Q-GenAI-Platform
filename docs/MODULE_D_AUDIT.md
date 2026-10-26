# Module D integration notes: audit and auth hooks in Module C

Module C ships **hooks, not a security implementation**. Nothing here makes the system secure by itself.

## Audit interface (`module_c/audit.py`)
```python
class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None: ...   # raise AuditError if the event is NOT durably stored
```
Inject your implementation: `RCAService(settings, retriever, provider, audit=YourSink())` or `create_app(service=...)`.
Reference sink: `JsonlAuditSink` (append + fsync). `RCA_AUDIT_SINK=none` exists for development only.

### Events emitted
| `event_type` | When | Notable fields |
|---|---|---|
| `rca.completed` | RCA returned (`success` or `insufficient_evidence`) | `rca_status`, `http_status=200`, `cited_documents[{document_id, version, chunk_ids}]`, `n_possible_causes`, `n_recommended_actions`, `defect_classes`, `llm_provider`, `llm_model` |
| `rca.failed` | LLM/service failure (`status=error`) | `error_code` (`llm_timeout`, `llm_not_configured`, ...), `http_status` |
| `rca.rejected` | Request failed validation | `http_status=422`, `error_code=invalid_request`, ids only if they were strings |

Common fields: `event_id`, `timestamp` (UTC ISO-8601 ms), `request_id` (also returned to the caller), `inspection_id`,
`image_id`, `model_version`, `actor` (set from the auth hook).
**Not recorded:** API keys, request free-text context, LLM prompts/outputs, passage text. Join to the document index via
`document_id`/`chunk_id` if reviewers need the text.
Not yet emitted (Module D to decide): document ingest/deactivate events (`ingest.document` is reserved but no code emits it),
and Pydantic-level 422s record only a minimal `rca.rejected` event without ids.

### Failure policy
`RCA_AUDIT_FAIL_MODE=fail_closed` (default): if `record()` raises `AuditError`, the caller gets HTTP 500
`audit_failure` and **the RCA result is withheld**. `fail_open`: result is returned with an `AUDIT WARNING` in `notes`
(the failure is visible to the caller, but there is a gap in the audit trail). Pick per your security requirements.

## Authentication / authorisation hook (`module_c/api.py`)
`Authenticator` is any callable `(Request) -> Optional[str]` returning an actor id or raising `HTTPException(401|403)`;
pass it to `create_app(authenticator=...)`. Shipped: `NoAuth` (default) and `StaticApiKeyAuth`
(`RCA_AUTH_MODE=static_api_key`, header `X-API-Key` vs env `RCA_API_KEY`, constant-time compare), a stop-gap only.
Role-based checks (e.g. who may read restricted documents) are not implemented: `access_level` filtering is global
per deployment (`RCA_ALLOWED_ACCESS_LEVELS`), not per user.

## Open questions for the Module D owner
1. Required audit store/format (signing, retention, tamper evidence)? 2. fail-closed or fail-open? 3. Token format and
actor/role claims? 4. Should ingestion/deactivation of documents be audited (recommended)? 5. Transport security (TLS) and secret management.

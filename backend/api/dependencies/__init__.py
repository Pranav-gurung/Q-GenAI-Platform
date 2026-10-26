"""FastAPI dependencies: DB session, services, (optional) API-key auth."""
from backend.api.dependencies.auth import require_api_key
from backend.api.dependencies.services import (get_db, get_feedback_service, get_inspection_service,
                                               get_settings_dep)

__all__ = ["require_api_key", "get_db", "get_inspection_service", "get_feedback_service", "get_settings_dep"]

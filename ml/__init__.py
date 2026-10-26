"""ML implementations that plug into backend.pipeline.interfaces (EMPTY in Part 1).

Dependency direction:  ml  ->  backend.pipeline.interfaces / backend.schemas.results
The backend never imports ml code directly; it is selected via backend/pipeline/registry.py.
"""

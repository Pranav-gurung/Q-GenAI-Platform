"""ML implementations that plug into backend.pipeline.interfaces.

    ml/cv/   computer vision: quality gate, YOLO CV engine, dataset tools, training, evaluation

Dependency direction:  ml  ->  backend.pipeline.interfaces / backend.schemas.results
The backend never imports ml code directly; it is selected via backend/pipeline/registry.py.
"""

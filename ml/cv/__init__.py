"""Q-GenAI computer vision subsystem.

Plugs into the pipeline only through backend.pipeline.interfaces (CVEngine, ImageQualityGate).
CV reports visual evidence; PASS/FAIL/REVIEW is decided by backend.pipeline.decision_rules.

Heavy dependencies (torch, ultralytics) are imported lazily, only by modules that need them.
"""

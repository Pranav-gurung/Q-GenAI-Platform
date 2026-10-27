"""RAG / SOP intelligence: retrieve SOP sections relevant to an RCA and ground corrective actions in them.

knowledge/*.md --ingest--> section chunks --embed--> local vector index --retrieve--> cited SOP context
--LLM (same provider as RCA)--> grounded actions, each tied to a retrieved chunk and a verbatim quote.
"""

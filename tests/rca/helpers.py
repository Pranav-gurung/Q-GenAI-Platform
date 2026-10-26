from backend.core.config import Settings
from backend.fulfillment.analyzer import analyze_trace
from backend.rca.providers.mock_provider import MockProvider
from backend.rca.service import RCAService
from tests.fulfillment.helpers import TRACE, K, LEAK, P, R, S, obs

CANONICAL = [obs(R), obs(S), obs(P), obs(K, "FAIL", [LEAK])]


def trace_of(observations, current=K):
    return analyze_trace(TRACE, observations, current_inspection_id=next(
        (o.inspection_id for o in observations if o.stage is current), "INS-CURRENT"), current_stage=current)


def service(mode="ok", **settings):
    mock = MockProvider(mode)
    svc = RCAService(Settings(_env_file=None, llm_enabled=True, llm_provider="mock", **settings), provider=mock)
    return svc, mock

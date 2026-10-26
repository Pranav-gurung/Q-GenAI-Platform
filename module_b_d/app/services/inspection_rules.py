"""PASS / FAIL / REVIEW decision rules (pure functions - easy to unit test).

IMPORTANT: model confidence is NOT physical defect severity, and a visual
model's output is NOT evidence of conformity to BIS / ISO standards. These
rules only turn detections into a routing decision for the production line.
"""
import json
from dataclasses import dataclass, field

from app.services.detector import ModelInfo

PASS, FAIL, REVIEW = "PASS", "FAIL", "REVIEW"


@dataclass(frozen=True)
class PolicyRules:
    version: int
    fail_confidence: float
    fail_classes: frozenset[str] = frozenset()     # empty -> all (non-ignored) classes
    ignored_classes: frozenset[str] = frozenset()
    max_allowed_defects: int = 0
    require_approved_model: bool = False

    @classmethod
    def from_db(cls, p) -> "PolicyRules":
        return cls(
            version=p.version,
            fail_confidence=p.fail_confidence,
            fail_classes=frozenset(json.loads(p.fail_classes)),
            ignored_classes=frozenset(json.loads(p.ignored_classes)),
            max_allowed_defects=p.max_allowed_defects,
            require_approved_model=p.require_approved_model,
        )


@dataclass
class Decision:
    status: str
    reasons: list[str] = field(default_factory=list)
    disqualifying_count: int = 0


def evaluate(defects: list[dict], rules: PolicyRules, model: ModelInfo) -> Decision:
    # 1. No trustworthy model -> a human must look at the part.
    if model.status != "loaded":
        return Decision(REVIEW, [f"Model {model.status}: automatic inspection not possible; "
                                 "manual inspection required."])
    if model.is_generic:
        return Decision(REVIEW, ["Loaded weights are a generic pretrained model, not a "
                                 "trained defect model; manual inspection required."])
    if rules.require_approved_model and not model.approved:
        return Decision(REVIEW, ["Policy requires an administrator-approved model; the active "
                                 "model is not approved."])

    considered = [d for d in defects if d["class_name"] not in rules.ignored_classes]
    ignored_n = len(defects) - len(considered)

    def is_fail_class(name: str) -> bool:
        return not rules.fail_classes or name in rules.fail_classes

    def is_disqualifying(d: dict) -> bool:
        return is_fail_class(d["class_name"]) and d["confidence"] >= rules.fail_confidence

    disqualifying = [d for d in considered if is_disqualifying(d)]
    uncertain = [d for d in considered if not is_disqualifying(d)]

    reasons: list[str] = []
    if ignored_n:
        reasons.append(f"{ignored_n} detection(s) of ignored classes were not considered.")

    # 2. Too many disqualifying defects -> FAIL
    if len(disqualifying) > rules.max_allowed_defects:
        classes = sorted({d["class_name"] for d in disqualifying})
        reasons.insert(0, f"{len(disqualifying)} disqualifying defect(s) ({', '.join(classes)}) at "
                          f"confidence >= {rules.fail_confidence:.2f}; policy allows "
                          f"{rules.max_allowed_defects}.")
        return Decision(FAIL, reasons, len(disqualifying))

    # 3. Detections the policy cannot classify with certainty -> REVIEW
    if uncertain:
        reasons.insert(0, f"{len(uncertain)} detection(s) below the fail confidence "
                          f"({rules.fail_confidence:.2f}) or of non-disqualifying classes; "
                          "human review required.")
        return Decision(REVIEW, reasons, len(disqualifying))

    # 4. Otherwise PASS
    if disqualifying:
        reasons.insert(0, f"{len(disqualifying)} disqualifying defect(s), within the allowed "
                          f"limit of {rules.max_allowed_defects}.")
    else:
        reasons.insert(0, "No disqualifying defects detected under the configured policy.")
    return Decision(PASS, reasons, len(disqualifying))

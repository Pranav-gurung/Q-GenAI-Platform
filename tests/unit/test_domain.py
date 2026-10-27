import pytest
from pydantic import ValidationError

from backend.domain.enums import DecisionOutcome, ModuleStatus
from backend.domain.fulfillment import FULFILLMENT_FLOW, FulfillmentStage, upstream_stages
from backend.pipeline.states import PipelineState
from backend.schemas.feedback import FeedbackCreate
from backend.schemas.inspection import InspectionCreate
from backend.schemas.results import BoundingBox, CVResult, QualityDecision


class TestFulfillmentStages:
    def test_flow_contains_every_stage_once_in_physical_order(self):
        assert [s.value for s in FULFILLMENT_FLOW] == [
            "RECEIVING", "STORAGE", "PICKING", "PACKING", "DISPATCH", "DELIVERY"]
        assert set(FULFILLMENT_FLOW) == set(FulfillmentStage)

    def test_upstream_stages(self):
        assert upstream_stages(FulfillmentStage.PACKING) == (
            FulfillmentStage.RECEIVING, FulfillmentStage.STORAGE, FulfillmentStage.PICKING)
        assert upstream_stages(FulfillmentStage.RECEIVING) == ()


class TestPipelineStates:
    def test_required_states_exist(self):
        expected = {"CREATED", "IMAGE_RECEIVED", "QUALITY_CHECK", "CV_PROCESSING", "DECISION", "PIPELINE_ANALYSIS",
                    "RCA_PROCESSING", "ACTION_GENERATED", "COMPLETED", "FAILED", "REQUIRES_REVIEW"}
        assert {s.value for s in PipelineState} == expected


class TestInspectionCreate:
    def test_valid(self):
        m = InspectionCreate(store_id="DS-BLR-001", sku_id="SKU-1", stage="PACKING", metadata={"station": "P3"})
        assert m.stage is FulfillmentStage.PACKING
        assert m.source.value == "OPERATOR_APP"

    @pytest.mark.parametrize("override", [
        {"stage": "TELEPORT"},                    # invalid stage
        {"store_id": "DS BLR/../001"},            # invalid id characters
        {"store_id": ""},                         # empty id
        {"sku_id": "X" * 65},                     # id too long
        {"unexpected": "field"},                  # unknown field
        {"metadata": {"nested": {"a": 1}}},       # metadata must be flat scalars
    ])
    def test_invalid(self, override):
        data = {"store_id": "DS-BLR-001", "sku_id": "SKU-1", "stage": "PACKING", **override}
        with pytest.raises(ValidationError):
            InspectionCreate(**data)

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            InspectionCreate(store_id="DS-BLR-001", stage="PACKING")


class TestResultContracts:
    def test_not_implemented_result_cannot_carry_predictions(self):
        with pytest.raises(ValidationError, match="must not carry result values"):
            CVResult(module="stub", status=ModuleStatus.NOT_IMPLEMENTED, overall_confidence=0.94)
        with pytest.raises(ValidationError):
            QualityDecision(module="stub", status=ModuleStatus.SKIPPED, outcome=DecisionOutcome.PASS)

    def test_not_implemented_result_cannot_claim_model_version(self):
        with pytest.raises(ValidationError):
            CVResult(module="stub", module_version="v1", status=ModuleStatus.NOT_IMPLEMENTED)

    def test_completed_decision_requires_outcome_and_source(self):
        with pytest.raises(ValidationError):
            QualityDecision(module="rules", status=ModuleStatus.COMPLETED)

    def test_bbox_ordering(self):
        with pytest.raises(ValidationError):
            BoundingBox(x_min=10, y_min=10, x_max=5, y_max=20)
        assert BoundingBox(x_min=0, y_min=0, x_max=5, y_max=5).x_max == 5

    def test_feedback_validation(self):
        FeedbackCreate(operator_id="OP-1", outcome="FAIL", defect_types=["TORN_PACKAGE"])
        with pytest.raises(ValidationError):
            FeedbackCreate(operator_id="OP-1", outcome="MAYBE")
        with pytest.raises(ValidationError):
            FeedbackCreate(operator_id="OP-1", outcome="FAIL", defect_types=["NOT_A_DEFECT"])

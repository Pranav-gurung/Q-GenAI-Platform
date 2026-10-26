"""Quick-commerce fulfillment stages.

Single source of truth for stage names and their order in the physical flow
(supplier -> receiving -> ... -> delivery). To add a stage: add an enum member AND
place it in FULFILLMENT_FLOW. DB columns store the value as a plain string
(non-native enum), so adding a stage needs no migration.
"""
from __future__ import annotations

from enum import Enum


class FulfillmentStage(str, Enum):
    RECEIVING = "RECEIVING"   # inbound from supplier / vendor
    STORAGE = "STORAGE"       # put-away and shelf / cold-room storage
    PICKING = "PICKING"       # picker collects items for an order
    PACKING = "PACKING"       # items packed into the order bag
    DISPATCH = "DISPATCH"     # handover to rider
    DELIVERY = "DELIVERY"     # at customer doorstep / return


FULFILLMENT_FLOW: tuple[FulfillmentStage, ...] = (
    FulfillmentStage.RECEIVING,
    FulfillmentStage.STORAGE,
    FulfillmentStage.PICKING,
    FulfillmentStage.PACKING,
    FulfillmentStage.DISPATCH,
    FulfillmentStage.DELIVERY,
)

STAGE_DESCRIPTIONS: dict[FulfillmentStage, str] = {
    FulfillmentStage.RECEIVING: "Inbound inspection of supplier deliveries",
    FulfillmentStage.STORAGE: "Put-away and storage (ambient / chilled / frozen)",
    FulfillmentStage.PICKING: "Item picked from shelf for a customer order",
    FulfillmentStage.PACKING: "Order packed into a bag or box",
    FulfillmentStage.DISPATCH: "Order handed over to the delivery partner",
    FulfillmentStage.DELIVERY: "Order delivered to (or returned by) the customer",
}


def stage_position(stage: FulfillmentStage) -> int:
    return FULFILLMENT_FLOW.index(stage)


def upstream_stages(stage: FulfillmentStage) -> tuple[FulfillmentStage, ...]:
    """Stages a product passed through before `stage` (candidates for where damage happened)."""
    return FULFILLMENT_FLOW[: stage_position(stage)]


# Every enum member must be placed in the flow exactly once.
assert set(FULFILLMENT_FLOW) == set(FulfillmentStage) and len(FULFILLMENT_FLOW) == len(FulfillmentStage)

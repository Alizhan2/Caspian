from dataclasses import dataclass

from app.core.config import Settings
from app.schemas import RiskLevel


@dataclass(frozen=True)
class RiskResult:
    level: RiskLevel
    explanation: str


def calculate_risk(
    mean_confidence: float,
    max_confidence: float,
    area_km2: float,
    settings: Settings,
    model_validated: bool = True,
) -> RiskResult:
    """Return a configurable AI screening score, never an official conclusion."""
    if not model_validated and (
        mean_confidence >= settings.low_confidence_threshold or area_km2 >= settings.medium_area_km2
    ):
        return RiskResult(
            RiskLevel.MEDIUM,
            "Experimental SAR screening priority; the score is not calibrated as oil probability.",
        )
    if max_confidence >= settings.high_confidence_threshold and area_km2 >= settings.high_area_km2:
        return RiskResult(RiskLevel.HIGH, "High AI screening score based on confidence and estimated area.")
    if mean_confidence >= settings.low_confidence_threshold or area_km2 >= settings.medium_area_km2:
        return RiskResult(RiskLevel.MEDIUM, "Medium AI screening score; confirm with field observations.")
    return RiskResult(RiskLevel.LOW, "Low AI screening score; small or low-confidence signal.")

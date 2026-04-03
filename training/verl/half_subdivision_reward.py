"""VeRL custom reward for half-subdivision GRPO."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_ROOT = REPO_ROOT / "environments" / "half_subdivision_shaped"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from half_subdivision_shaped.geometry import (  # noqa: E402
    build_geometry_case,
    score_breakdown,
)
from half_subdivision_shaped.parser import make_parser, parse_labels  # noqa: E402

PARSER = make_parser()
REWARD_FIELD_ORDER = (
    "score",
    "parseable",
    "face_credit",
    "near_contact_credit",
    "valid_prediction_fraction",
    "pred_count",
    "truth_count",
)


def ordered_reward_fields(values: dict[str, float]) -> dict[str, float]:
    """Return reward fields in a stable order for VeRL chunk merges."""

    return {key: float(values[key]) for key in REWARD_FIELD_ORDER}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, float]:
    """Compute the shaped reward for one VeRL rollout."""

    if data_source != "half_subdivision":
        return ordered_reward_fields(
            {
                "score": 0.0,
                "parseable": 0.0,
                "face_credit": 0.0,
                "near_contact_credit": 0.0,
                "valid_prediction_fraction": 0.0,
                "pred_count": 0.0,
                "truth_count": 0.0,
            }
        )

    labels = parse_labels(PARSER.parse_answer(solution_str))
    if labels is None:
        return ordered_reward_fields(
            {
                "score": 0.0,
                "parseable": 0.0,
                "face_credit": 0.0,
                "near_contact_credit": 0.0,
                "valid_prediction_fraction": 0.0,
                "pred_count": 0.0,
                "truth_count": float(len(json.loads(ground_truth))),
            }
        )

    case = build_geometry_case(
        {
            "ground_truth": json.loads(ground_truth),
            "datagen_args": extra_info["datagen_args"],
            "runtime": extra_info["runtime"],
        }
    )
    breakdown = score_breakdown(labels, case, near_contact_credit=0.25)
    return ordered_reward_fields(
        {
            "score": breakdown["score"],
            "parseable": 1.0,
            **breakdown,
        }
    )

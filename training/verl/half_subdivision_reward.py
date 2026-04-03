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
    geometric_credit_sum,
    shaped_score,
    valid_predictions,
)
from half_subdivision_shaped.parser import make_parser, parse_labels  # noqa: E402

PARSER = make_parser()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, float]:
    """Compute the shaped reward for one VeRL rollout."""

    if data_source != "half_subdivision":
        return {"score": 0.0}

    labels = parse_labels(PARSER.parse_answer(solution_str))
    if labels is None:
        return {"score": 0.0, "parseable": 0.0, "valid_labels": 0.0, "geometric_credit": 0.0}

    case = build_geometry_case(
        {
            "ground_truth": json.loads(ground_truth),
            "datagen_args": extra_info["datagen_args"],
            "runtime": extra_info["runtime"],
        }
    )
    valid_labels = valid_predictions(labels, case)
    total_credit = geometric_credit_sum(valid_labels, case, near_contact_credit=0.25)

    geometric_credit = min(total_credit / len(valid_labels), 1.0) if valid_labels else 0.0
    valid_fraction = len(valid_labels) / len(labels) if labels else 1.0
    return {
        "score": shaped_score(labels, case, near_contact_credit=0.25),
        "parseable": 1.0,
        "valid_labels": valid_fraction,
        "geometric_credit": geometric_credit,
    }

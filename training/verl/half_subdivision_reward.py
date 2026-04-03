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
    shaped_score,
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
        return {"score": 0.0}

    case = build_geometry_case(
        {
            "ground_truth": json.loads(ground_truth),
            "datagen_args": extra_info["datagen_args"],
            "runtime": extra_info["runtime"],
        }
    )
    return {"score": shaped_score(labels, case, near_contact_credit=0.25)}

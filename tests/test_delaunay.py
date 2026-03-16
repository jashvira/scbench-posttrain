from __future__ import annotations

import json

import pytest

from visual_geometry_bench.datagen.delaunay_tasks import (  # type: ignore[import-not-found]
    generate_dataset_record as vgb_generate_dataset_record,
)

from scbench_posttrain.delaunay import (
    DelaunayRecord,
    extract_python_literal,
    generate_dataset_record,
    render_delaunay_prompt_image,
    sample_unique_delaunay_points,
    score_delaunay_answer,
)


@pytest.mark.parametrize(
    "datagen_args",
    [
        {"num_points": 6, "seed": 10},
        {"num_points": 10, "seed": 59},
        {"num_points": 12, "seed": 71},
    ],
)
def test_generate_dataset_record_matches_vgb_core(datagen_args):
    """Local record generation should match the upstream VGB Delaunay record exactly."""

    local_record = generate_dataset_record(datagen_args)
    vgb_record = vgb_generate_dataset_record(datagen_args)

    assert local_record == vgb_record


def test_sample_unique_delaunay_points_is_deterministic():
    """Sampling should be deterministic for a fixed seed."""

    points_a = sample_unique_delaunay_points(8, seed=33)
    points_b = sample_unique_delaunay_points(8, seed=33)

    assert points_a.shape == (8, 2)
    assert points_b.shape == (8, 2)
    assert points_a.tolist() == points_b.tolist()


def test_extract_python_literal_strips_thinking_and_prefers_last_literal():
    """Literal extraction should ignore thinking blocks and use the fenced answer."""

    text = """
    <thinking>
    [[999, 999, 999]]
    </thinking>

    Here is the result:

    ```python
    [[0, 1, 2], [1, 2, 3]]
    ```
    """

    assert extract_python_literal(text) == "[[0, 1, 2], [1, 2, 3]]"


def test_extract_python_literal_parses_simple_sequences():
    """Simple comma-delimited sequences should follow the upstream parser fallback."""

    assert extract_python_literal("answer: 0, 1, 2") == "[0, 1, 2]"


def test_score_delaunay_answer_accepts_unsorted_triangles():
    """The scorer should canonicalize triangle and list ordering."""

    record = DelaunayRecord.model_validate(generate_dataset_record({"num_points": 8, "seed": 33}))
    scrambled = [[triangle[2], triangle[0], triangle[1]] for triangle in reversed(record.ground_truth)]

    evaluation = score_delaunay_answer(str(scrambled), record)

    assert evaluation.parsed_ok is True
    assert evaluation.passed is True
    assert evaluation.score == 1.0
    assert evaluation.error_type is None


def test_score_delaunay_answer_reports_exact_mismatch():
    """Dropping a triangle should report an exact mismatch."""

    record = DelaunayRecord.model_validate(generate_dataset_record({"num_points": 8, "seed": 33}))
    wrong_answer = json.dumps(record.ground_truth[:-1])

    evaluation = score_delaunay_answer(wrong_answer, record)

    assert evaluation.parsed_ok is True
    assert evaluation.passed is False
    assert evaluation.error_type == "exact_mismatch"
    assert evaluation.missing == [record.ground_truth[-1]]
    assert evaluation.extra == []


def test_score_delaunay_answer_reports_parse_failure():
    """Non-literal output should be reported as a parse failure."""

    record = generate_dataset_record({"num_points": 8, "seed": 33})

    evaluation = score_delaunay_answer("not a triangulation", record)

    assert evaluation.parsed_ok is False
    assert evaluation.passed is False
    assert evaluation.error_type == "parse_failure"


def test_render_delaunay_prompt_image_returns_svg_data_url():
    """Prompt visuals should be emitted as SVG data URLs."""

    image = render_delaunay_prompt_image({"num_points": 8, "seed": 33})

    assert image.startswith("data:image/svg+xml;base64,")

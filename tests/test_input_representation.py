"""Input representation validation and delta arithmetic."""

import numpy as np
import pytest

from regulus.perturb.representation import (
    apply_anchor_plus_delta,
    subtract_control_mean,
    validate_input_matrix,
)
from regulus.perturb.spec import normalize_representation


@pytest.mark.parametrize("representation", ["delta", "post_state"])
def test_validation_does_not_transform_values(representation):
    values = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
    output = validate_input_matrix(values, representation, name="test")
    np.testing.assert_array_equal(output, values)


def test_subtract_control_mean():
    output = subtract_control_mean(
        np.array([4.0, 7.0], dtype=np.float32),
        np.array([1.5, 2.0], dtype=np.float32),
        name="gene matrix",
    )
    np.testing.assert_allclose(output, [2.5, 5.0])


def test_log2fc_is_not_a_public_representation():
    with pytest.raises(ValueError):
        normalize_representation("log2fc")


def test_non_finite_input_fails_fast():
    with pytest.raises(ValueError, match="non-finite"):
        validate_input_matrix(np.array([np.nan]), "delta", name="test")


def test_anchor_plus_delta():
    output = apply_anchor_plus_delta(np.array([0.1, 0.2]), [1], [0.5])
    np.testing.assert_allclose(output, [0.1, 0.7])

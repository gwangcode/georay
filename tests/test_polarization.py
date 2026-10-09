"""Unit tests for polarization utilities."""
import math
import torch

from georay.polarization import (
    apply_jones_mueller,
    degree_of_polarization,
    _choose_perpendicular,
)


def test_perpendicular():
    d = torch.tensor([[0.0, 0.0, 1.0]])
    s = _choose_perpendicular(d)
    dot = torch.sum(s * d, dim=-1)
    assert abs(dot.item()) < 1e-6, "s must be perpendicular to d"
    assert abs(torch.norm(s).item() - 1.0) < 1e-6, "s must be unit length"


def test_identity_jones():
    # r_s = r_p = 1 (no change)
    stokes = torch.tensor([[1.0, 0.5, -0.3, 0.1]])
    r_s = torch.tensor([[1.0]])
    r_p = torch.tensor([[1.0]])
    out = apply_jones_mueller(stokes, r_s, r_p)
    assert torch.allclose(out, stokes, atol=1e-5)


def test_dop_pure():
    stokes = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    dop = degree_of_polarization(stokes)
    assert abs(dop.item() - 1.0) < 1e-6


def test_dop_unpolarized():
    stokes = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    dop = degree_of_polarization(stokes)
    assert abs(dop.item()) < 1e-6
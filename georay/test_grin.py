"""Unit tests for GRIN fields."""
import torch

from georay.grin import parse_grin_input_to_field


def test_constant_field():
    device = torch.device("cpu")
    field = parse_grin_input_to_field(1.5, device)
    r = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
    n, grad = field(r)
    assert torch.allclose(n, torch.full_like(n, 1.5))
    assert torch.allclose(grad, torch.zeros_like(grad))


def test_function_field():
    device = torch.device("cpu")

    def fn(x, y, z, wl):
        return 1.5 - 0.1 * (x ** 2 + y ** 2)

    field = parse_grin_input_to_field(fn, device)
    r = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    n, grad = field(r)
    assert abs(n[0, 0].item() - 1.5) < 1e-4
    assert grad.shape == r.shape
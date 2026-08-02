
import torch
from generator.mdn import MotiveDecompositionNetwork

TOL = 1e-6

def test_support_values_are_feasible_across_magnitudes_for_two_objectives():
    """Sweeps input magnitude to catch a regression that reintroduces
    the old Softplus bug, even one that only shows up at extreme scale."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=2)
    for scale in (0.1, 1.0, 10.0, 1e4, 1e10):
        context = torch.randn(500, 8) * scale
        with torch.no_grad():
            _, support_values = model.forward_inference(context)
        assert torch.all(support_values >= 0)
        assert torch.all(support_values <= 1)
        assert torch.all(torch.sum(support_values, dim=-1) >= 1.0-TOL)
        assert not torch.any(torch.isnan(support_values))
        assert not torch.any(torch.isinf(support_values))


def test_support_values_are_feasible_across_magnitudes_for_five_objectives():
    """Same sweep for M=5, to catch a regression specific to a
    different num_objectives configuration."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=5)
    for scale in (0.1, 1.0, 10.0, 1e4, 1e10):
        context = torch.randn(500, 8) * scale
        with torch.no_grad():
            _, support_values = model.forward_inference(context)
        assert torch.all(support_values >= 0)
        assert torch.all(support_values <= 1)
        assert torch.all(torch.sum(support_values, dim=-1) >= 1.0-TOL)


def test_support_values_are_feasible_after_training():
    """the feasibility check on a genuinely trained (not just
    random-init) model, since some failures only show up once confident."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    context = torch.randn(200, 8)
    target = (context[:, 0] > 0).long()
    for _ in range(400):
        optimizer.zero_grad()
        _, support_values = model.forward_inference(context)
        loss = torch.nn.functional.cross_entropy(support_values, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, support_values = model.forward_inference(context)
    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)
    assert torch.all(torch.sum(support_values, dim=-1) >= 1.0-TOL)

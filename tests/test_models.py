import pytest

torch = pytest.importorskip("torch")
from avograde.models.grader_cnn import GraderCNN


def test_forward_shape():
    model = GraderCNN(num_classes=5)
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 5)

from unittest.mock import patch

from src.model import select_device


def test_select_device_returns_cpu_when_cuda_unavailable():
    with patch("torch.cuda.is_available", return_value=False):
        device = select_device()

    assert device.type == "cpu"


def test_select_device_returns_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        device = select_device()

    assert device.type == "cuda"


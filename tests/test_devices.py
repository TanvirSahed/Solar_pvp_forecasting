"""Hardware selection and portability checks that need no datasets or downloaded weights."""
import pytest
import torch

import pvf
from pvf.training.trainer import autocast_dtype, pick_device


@pytest.mark.parametrize("has_gpu", [False, True])
def test_automatic_device_selection(monkeypatch, has_gpu):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: has_gpu)
    assert pick_device().type == ("cuda" if has_gpu else "cpu")


@pytest.mark.parametrize("precision", ["bf16", "fp16", "none"])
def test_cpu_disables_mixed_precision(precision):
    assert autocast_dtype(pvf.TrainingRecipe(mixed_precision=precision), torch.device("cpu")) is None


@pytest.mark.parametrize("supports_bf16", [False, True])
def test_gpu_precision_falls_back_when_bf16_is_unsupported(monkeypatch, supports_bf16):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supports_bf16)
    dtype = autocast_dtype(pvf.TrainingRecipe(), torch.device("cuda"))
    assert dtype == (torch.bfloat16 if supports_bf16 else torch.float16)
    assert autocast_dtype(pvf.TrainingRecipe(mixed_precision="none"), torch.device("cuda")) is None


def test_cpu_is_a_ready_environment(monkeypatch, capsys):
    from pvf import workspace

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(workspace, "SITES", {})
    assert pvf.check_environment()
    output = capsys.readouterr().out
    assert "CPU" in output
    assert "FAIL" not in output


@pytest.mark.parametrize("has_parallel_scan", [False, True])
def test_saved_cuda_mamba_runs_on_cpu(monkeypatch, has_parallel_scan):
    from pvf.models.blocks import mamba_blocks

    def unexpected_cuda_call(*args, **kwargs):
        pytest.fail("A CPU tensor was passed to the CUDA scan")

    monkeypatch.setattr(mamba_blocks, "_cuda_selective_scan", unexpected_cuda_call)
    if not has_parallel_scan:
        monkeypatch.setattr(mamba_blocks, "_parallel_scan", None)
    mixer = mamba_blocks.MambaMixer(model_size=8, state_size=4, scan_backend="sequential")
    inputs = torch.randn(2, 8, 8)
    expected = mixer(inputs)
    mixer.scan_backend = "cuda"  # backend stored in a model saved on a GPU machine
    actual = mixer(inputs)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(mixer.input_projection.weight.grad).all()

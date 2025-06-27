"""Common test configurations."""

from copy import deepcopy

import pytest
import torch

from spec_mamba.models.common.base_model import BaseModel


@pytest.fixture(scope="session")
def device() -> str:
    assert torch.cuda.is_available(), "CUDA is not available."
    return "cuda:0"


@pytest.fixture(scope="session")
def spec_size() -> tuple[int, int]:
    return (80, 200)


@pytest.fixture(scope="session")
def patch_size() -> tuple[int, int]:
    return (16, 4)


@pytest.fixture(scope="session")
def channels() -> int:
    return 2


@pytest.fixture(scope="session")
def embed_dim() -> int:
    return 192


@pytest.fixture(scope="session")
def batch_size() -> int:
    return 2


@pytest.fixture(scope="session")
def num_classes() -> int:
    return 10


@pytest.fixture(scope="session")
def sample(
    batch_size: int, channels: int, spec_size: tuple[int, int], device: str
) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batch_size, channels, *spec_size).to(device)


def train_one_epoch(model: BaseModel, sample: torch.Tensor) -> None:
    old_params = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()
    model.train()

    for _, param in model.named_parameters():
        assert param.requires_grad

    optimizer.zero_grad()
    logits, mask = model(sample, output_type="emb")
    reconstruction, mask = model.reshape_as_spec(logits, mask)
    loss = loss_fn(reconstruction, sample)

    assert isinstance(loss, torch.Tensor)

    loss.backward()
    optimizer.step()
    new_params = model.state_dict()

    unchanged_params = [k for k in old_params.keys() if k not in new_params.keys()]
    if unchanged_params:
        pytest.fail(
            f"The following parameters did not update after optimizer.step(): {unchanged_params}"
        )

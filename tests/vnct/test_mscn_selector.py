import pytest
import torch

from vnct.models.selectors import MSCNGGDSelector


def test_mscn_selector_shapes_ranges_and_determinism() -> None:
    torch.manual_seed(9)
    selector = MSCNGGDSelector(
        patch_size=8,
        patch_stride=8,
        num_patches=4,
        alpha_steps=501,
    )
    image = torch.rand(2, 3, 32, 32, requires_grad=True)

    first = selector(image)
    second = selector(image)

    assert first.score_map.shape == (2, 1, 32, 32)
    assert first.patch_scores.shape == (2, 4)
    assert first.boxes.shape == (2, 4, 4)
    assert first.centers.shape == (2, 4, 2)
    assert first.patch_alpha.shape == (2, 4)
    assert first.patch_sigma.shape == (2, 4)
    assert torch.all((0.0 <= first.score_map) & (first.score_map <= 1.0))
    assert torch.all((0.0 <= first.centers) & (first.centers <= 1.0))
    assert torch.all(first.patch_scores[:, :-1] >= first.patch_scores[:, 1:])
    torch.testing.assert_close(first.patch_scores, second.patch_scores)
    torch.testing.assert_close(first.boxes, second.boxes)
    assert not first.patch_scores.requires_grad


def test_local_noise_patch_is_selected() -> None:
    torch.manual_seed(4)
    selector = MSCNGGDSelector(
        patch_size=16,
        patch_stride=16,
        num_patches=1,
        alpha_steps=1001,
    )
    coordinate = torch.linspace(0.0, 1.0, 64)
    smooth = 0.5 * coordinate.view(1, 1, 1, 64) + 0.5 * coordinate.view(1, 1, 64, 1)
    image = smooth.expand(1, 3, -1, -1).clone()
    image[:, :, 32:48, 16:32] = torch.rand(1, 3, 16, 16)

    selection = selector(image)

    expected = torch.tensor([16, 32, 32, 48])
    torch.testing.assert_close(selection.boxes[0, 0].cpu(), expected)


def test_selector_rejects_insufficient_candidates() -> None:
    selector = MSCNGGDSelector(patch_size=16, num_patches=5, alpha_steps=101)
    with pytest.raises(ValueError, match="fewer than num_patches"):
        selector(torch.rand(1, 3, 32, 32))

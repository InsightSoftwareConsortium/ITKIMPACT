#!/usr/bin/env python3
#==========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#==========================================================================

#
# Regenerate the tiny TorchScript model used by the backend unit tests:
#
#     python3 GenerateImpactTestModels.py
#
# The model is deterministic (fixed seed) so the committed .pt is reproducible.
# It takes a [B, C, D, H, W] tensor and returns a list of feature maps (two
# "layers"), mimicking a segmentation backbone with a couple of output levels.
import torch
from typing import List, Optional, Tuple


class ImpactToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # One 3x3x3 convolution (real parameters, so device/precision handling is
        # actually exercised) producing 4 feature channels, kept spatially aligned.
        self.conv = torch.nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.conv(x)              # [B, 4, D, H, W] (varied feature channels)
        passthrough = torch.cat([x, x], 1)   # [B, 2, D, H, W]
        return [features, passthrough]       # layer 0: 4 channels, layer 1: 2 channels


class ImpactToyModelDown(torch.nn.Module):
    """A downsampling backbone: both output layers are at half the input resolution,
    exercising the feature-map spacing/geometry derived from the input/output size
    ratio (and multi-layer independence under downsampling)."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.AvgPool3d(2)                                   # /2, position-preserving
        self.conv = torch.nn.Conv3d(1, 3, kernel_size=3, stride=2, padding=1, bias=True)  # /2, 3 channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [self.pool(x), self.conv(x)]  # layer 0: 1 channel (clean), layer 1: 3 channels


class ImpactToyModel2D(torch.nn.Module):
    """The 2D counterpart, for the case of a model of lower dimension than the image: it is
    swept over the volume, one slice at a time."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.conv(x)              # [B, 4, H, W]
        passthrough = torch.cat([x, x], 1)   # [B, 2, H, W]
        return [features, passthrough]


class ImpactToyModelMetadata(torch.nn.Module):
    """A metadata-aware model, in the shape ImpactLoss builds them (forward with four arguments).

    Layer 0 is the intensity passthrough, normalized by the image range when the caller hands
    over stats and by its input's own range otherwise, as the real models do. Layer 1 carries
    the image sigma from stats, or zero when none were given, so a metric comparing two images
    through it reads (sigma_fixed - sigma_moving)^2 exactly when both sides received their own
    stats. Written as sigma + 0 * x, the layer stays on the autograd graph the online mode
    differentiates through; its derivative is zero.
    """

    def forward(self,
                x: torch.Tensor,
                nb_layers: torch.Tensor = torch.tensor([2]),
                stats: torch.Tensor = torch.tensor([]),
                direction: torch.Tensor = torch.tensor([])) -> List[torch.Tensor]:
        if stats.numel() == 4:
            lo, hi, sigma = stats[0], stats[1], stats[3]
        else:
            lo, hi, sigma = x.min(), x.max(), torch.zeros((), dtype=x.dtype, device=x.device)
        normalized = (x - lo) / (hi - lo + 1e-6)
        return [normalized, sigma + 0 * x]


class ImpactToyModelNorm(torch.nn.Module):
    """Two instance normalisations, as a TS network's: a 3x3x3 convolution normalised then rectified (layer 0, at the
    input's resolution) and a strided one normalised (layer 1, at half of it). Normalisation k takes norms[k] ([2, C]:
    the mean and the biased variance) when given, its input's own statistics otherwise; norm_input gives the input of
    normalisation number len(norms) and norm_count how many the first nb_layers layers go through."""

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = torch.nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=True)
        self.conv1 = torch.nn.Conv3d(3, 2, kernel_size=3, stride=2, padding=1, bias=True)

    def _normalized(self, x: torch.Tensor, norms: List[torch.Tensor], k: int) -> torch.Tensor:
        if k < len(norms):
            mean = norms[k][0].view(1, -1, 1, 1, 1)
            variance = norms[k][1].view(1, -1, 1, 1, 1)
        else:
            mean = x.mean((2, 3, 4), keepdim=True)
            variance = x.var((2, 3, 4), unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance + 1e-5)

    def _run(self, x: torch.Tensor, norms: List[torch.Tensor], stop: int) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        a = self.conv0(x)
        if stop == 0:
            return [], a
        a = torch.relu(self._normalized(a, norms, 0))
        b = self.conv1(a)
        if stop == 1:
            return [a], b
        return [a, self._normalized(b, norms, 1)], None

    def forward(self,
                x: torch.Tensor,
                nb_layers: torch.Tensor = torch.tensor([2]),
                stats: torch.Tensor = torch.tensor([]),
                direction: torch.Tensor = torch.tensor([]),
                norms: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        return self._run(x, [] if norms is None else norms, -1)[0]

    @torch.jit.export
    def norm_count(self, nb_layers: int) -> int:
        return 1 if nb_layers == 1 else 2

    @torch.jit.export
    def norm_input(self,
                   x: torch.Tensor,
                   stats: torch.Tensor = torch.tensor([]),
                   direction: torch.Tensor = torch.tensor([]),
                   norms: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        given = [] if norms is None else norms
        captured = self._run(x, given, len(given))[1]
        assert captured is not None
        return captured


class ImpactToyModelNormQuarter(ImpactToyModelNorm):
    """ImpactToyModelNorm with its second normalisation at a quarter of the input's resolution (a convolution of
    stride 4): the input of a short region gives it a number of samples from which its stride cannot be told."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 2, kernel_size=3, stride=4, padding=1, bias=True)


def main() -> None:
    torch.manual_seed(20240601)
    model = ImpactToyModel().eval()
    torch.jit.script(model).save("ImpactToyModel.pt")
    print("wrote ImpactToyModel.pt")

    torch.manual_seed(20240601)
    model2d = ImpactToyModel2D().eval()
    torch.jit.script(model2d).save("ImpactToyModel2D.pt")
    print("wrote ImpactToyModel2D.pt")

    torch.manual_seed(20240601)
    down = ImpactToyModelDown().eval()
    torch.jit.script(down).save("ImpactToyModelDown.pt")
    print("wrote ImpactToyModelDown.pt")

    metadata = ImpactToyModelMetadata().eval()
    torch.jit.script(metadata).save("ImpactToyModelMetadata.pt")
    print("wrote ImpactToyModelMetadata.pt")

    torch.manual_seed(20240601)
    norm = ImpactToyModelNorm().eval()
    torch.jit.script(norm).save("ImpactToyModelNorm.pt")
    print("wrote ImpactToyModelNorm.pt")

    torch.manual_seed(20240601)
    quarter = ImpactToyModelNormQuarter().eval()
    torch.jit.script(quarter).save("ImpactToyModelNormQuarter.pt")
    print("wrote ImpactToyModelNormQuarter.pt")



if __name__ == "__main__":
    main()

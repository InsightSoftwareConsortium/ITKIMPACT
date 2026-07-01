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
from typing import List


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


def main() -> None:
    torch.manual_seed(20240601)
    model = ImpactToyModel().eval()
    torch.jit.script(model).save("ImpactToyModel.pt")
    print("wrote ImpactToyModel.pt")

    torch.manual_seed(20240601)
    down = ImpactToyModelDown().eval()
    torch.jit.script(down).save("ImpactToyModelDown.pt")
    print("wrote ImpactToyModelDown.pt")


if __name__ == "__main__":
    main()

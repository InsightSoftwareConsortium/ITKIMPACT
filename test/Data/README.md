# Test data

Small, reproducible inputs are committed directly and used by the self-contained tests:

- `ImpactToyModel.pt`, `ImpactToyModelDown.pt` — tiny deterministic TorchScript models
  (regenerate with `GenerateImpactTestModels.py`).

Every C++ unit test and GoogleTest synthesizes its images at run time, so no heavy binary
is required to build and run the default suite.

## Optional real-CT regression (out-of-git, ExternalData)

`ImpactConvexAdam.RealLungCTCoarseInitThenAdamRefines` validates the coarse+fine pipeline on
a real lung-CT pair. Those inputs are large, so they are **not** committed:

- `3DCT_lung_baseline.mha`, `3DCT_lung_followup.mha` — inhale/exhale lung CT pair
- `M258_2_Layers.pt` — a real IMPACT feature extractor

They are kept out of git (see `../../.gitignore`) and the test **skips** when they are absent
(e.g. on CI). To run it locally, drop the three files in this directory.

To make it run in CI, host the blobs via ITK ExternalData:

1. `cmake -DExternalData_LINK_CONTENT=SHA512 ...` to turn the local files into committed
   `*.sha512` content links (or create them by hand: `sha512sum <file>` into `<file>.sha512`).
2. Upload the blobs to a store reachable by ITK's default `ExternalData_URL_TEMPLATES`
   (e.g. https://data.kitware.com, addressed by SHA512).
3. Reference them with `DATA{...}` from an `itk_add_test` driver so ExternalData fetches them
   before the test runs.

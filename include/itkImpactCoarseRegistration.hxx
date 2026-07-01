/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkImpactCoarseRegistration_hxx
#define itkImpactCoarseRegistration_hxx

// LibTorch-backed implementation; included only when ITK_MANUAL_INSTANTIATION is undefined.

#include "itkImpactCoarseRegistration.h"
#include "itkImpactTorchRegistrationHelpers.h"

#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkIdentityTransform.h>

#include <torch/torch.h>

#include <vector>

namespace itk
{

template <typename TFixedImage, typename TMovingImage>
ImpactCoarseRegistration<TFixedImage, TMovingImage>::ImpactCoarseRegistration() = default;

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::SetFixedImage(const FixedImageType * image)
{
  if (m_FixedImage.GetPointer() != image)
  {
    m_FixedImage = image;
    this->Modified();
  }
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::SetMovingImage(const MovingImageType * image)
{
  if (m_MovingImage.GetPointer() != image)
  {
    m_MovingImage = image;
    this->Modified();
  }
}

template <typename TFixedImage, typename TMovingImage>
auto
ImpactCoarseRegistration<TFixedImage, TMovingImage>::GetDisplacementField() -> DisplacementFieldType *
{
  return this->GetOutput();
}

template <typename TFixedImage, typename TMovingImage>
auto
ImpactCoarseRegistration<TFixedImage, TMovingImage>::GetDisplacementFieldTransform()
  -> DisplacementFieldTransformType *
{
  return m_DisplacementFieldTransform.GetPointer();
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();
  if (m_FixedImage.IsNull())
  {
    return;
  }
  DisplacementFieldType * output = this->GetOutput();
  output->SetLargestPossibleRegion(m_FixedImage->GetLargestPossibleRegion());
  output->SetSpacing(m_FixedImage->GetSpacing());
  output->SetOrigin(m_FixedImage->GetOrigin());
  output->SetDirection(m_FixedImage->GetDirection());
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::GenerateData()
{
  namespace F = torch::nn::functional;

  if (m_FixedImage.IsNull() || m_MovingImage.IsNull())
  {
    itkExceptionMacro("Both SetFixedImage() and SetMovingImage() are required.");
  }

  if constexpr (ImageDimension != 2 && ImageDimension != 3)
  {
    itkExceptionMacro("ImpactCoarseRegistration supports 2D and 3D images only.");
  }
  else
  {
    const bool featureMode = !m_FixedModelsConfiguration.empty();

    const torch::Device device(m_Device);
    torch::manual_seed(m_Seed);
    torch::NoGradGuard noGrad; // the coarse stage is non-differentiable

    // ---- Fixed and moving onto the same (fixed) voxel grid; resample only if geometries differ. ----
    typename MovingImageType::ConstPointer movingOnFixed = m_MovingImage;
    {
      const auto & fSize = m_FixedImage->GetLargestPossibleRegion().GetSize();
      const auto & mSize = m_MovingImage->GetLargestPossibleRegion().GetSize();
      const bool   sameGrid = (fSize == mSize) && (m_FixedImage->GetSpacing() == m_MovingImage->GetSpacing()) &&
                            (m_FixedImage->GetOrigin() == m_MovingImage->GetOrigin()) &&
                            (m_FixedImage->GetDirection() == m_MovingImage->GetDirection());
      if (!sameGrid)
      {
        using ResampleType = ResampleImageFilter<MovingImageType, MovingImageType, double>;
        auto resample = ResampleType::New();
        resample->SetInput(m_MovingImage);
        resample->SetTransform(IdentityTransform<double, ImageDimension>::New());
        resample->SetInterpolator(LinearInterpolateImageFunction<MovingImageType, double>::New());
        resample->SetUseReferenceImage(true);
        resample->SetReferenceImage(m_FixedImage);
        resample->Update();
        movingOnFixed = resample->GetOutput();
      }
    }

    torch::Tensor fixedT = Impact::ImageToBatchTensor(m_FixedImage.GetPointer()).to(device);
    torch::Tensor movingT = Impact::ImageToBatchTensor(movingOnFixed.GetPointer()).to(device);

    // Full-resolution spatial sizes in torch (z, y, x) order.
    const auto &         fixedSize = m_FixedImage->GetLargestPossibleRegion().GetSize();
    std::vector<int64_t> spatial(ImageDimension);
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      spatial[d] = static_cast<int64_t>(fixedSize[ImageDimension - 1 - d]);
    }

    // ---- Cost source: raw intensities, or IMPACT feature channels (per native-resolution layer). ----
    std::vector<torch::Tensor> fixedLayers; // empty in intensity mode
    std::vector<torch::Tensor> movingLayers;
    if (featureMode)
    {
      const std::vector<ModelConfiguration> & movingConfigs =
        m_MovingModelsConfiguration.empty() ? m_FixedModelsConfiguration : m_MovingModelsConfiguration;
      fixedLayers =
        Impact::ExtractFeatureLayers<ImageDimension>(m_FixedModelsConfiguration, fixedT, device, m_SubsetFeatures);
      movingLayers =
        Impact::ExtractFeatureLayers<ImageDimension>(movingConfigs, movingT, device, m_SubsetFeatures);
      if (fixedLayers.empty() || fixedLayers.size() != movingLayers.size())
      {
        itkExceptionMacro("ImpactCoarseRegistration: fixed/moving produced "
                          << fixedLayers.size() << " and " << movingLayers.size() << " feature layers.");
      }
    }

    const int64_t gs = static_cast<int64_t>(m_GridSpacing);
    const int64_t hw = static_cast<int64_t>(m_DisplacementHalfWidth);
    const int64_t L1 = 2 * hw + 1;
    int64_t       L = 1;
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      L *= L1;
    }

    // Dimension-generic avg-pool / replicate-pad / interpolate helpers (2D or 3D).
    auto poolStride = [&](const torch::Tensor & t, int64_t k) {
      if constexpr (ImageDimension == 3)
        return F::avg_pool3d(t, F::AvgPool3dFuncOptions(k).stride(k));
      else
        return F::avg_pool2d(t, F::AvgPool2dFuncOptions(k).stride(k));
    };
    auto smoothBox = [&](const torch::Tensor & t) {
      if constexpr (ImageDimension == 3)
        return F::avg_pool3d(t, F::AvgPool3dFuncOptions(3).stride(1).padding(1));
      else
        return F::avg_pool2d(t, F::AvgPool2dFuncOptions(3).stride(1).padding(1));
    };

    // ---- Coarse grid: average-pool each cost source to the common coarse grid. ----
    // A native-resolution source (intensities, or a full-res feature layer) uses the exact
    // stride-GridSpacing pool -- pooling per layer then concatenating is bit-identical to pooling the
    // concatenation (avg-pool is per-channel). An already-downsampled backbone layer is adaptive-pooled
    // to the same coarse grid, so its features are never upsampled to full res.
    std::vector<int64_t> coarseSpatial(ImageDimension);
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      coarseSpatial[d] = spatial[d] / gs;
    }
    auto toCoarseGrid = [&](const torch::Tensor & t) -> torch::Tensor {
      const std::vector<int64_t> ts(t.sizes().begin() + 2, t.sizes().end());
      if (ts == spatial)
      {
        return poolStride(t, gs);
      }
      if constexpr (ImageDimension == 3)
        return F::adaptive_avg_pool3d(
          t, F::AdaptiveAvgPool3dFuncOptions({ coarseSpatial[0], coarseSpatial[1], coarseSpatial[2] }));
      else
        return F::adaptive_avg_pool2d(t, F::AdaptiveAvgPool2dFuncOptions({ coarseSpatial[0], coarseSpatial[1] }));
    };
    torch::Tensor FcFix;
    torch::Tensor FcMov;
    if (featureMode)
    {
      std::vector<torch::Tensor> fixedCoarse, movingCoarse;
      for (size_t l = 0; l < fixedLayers.size(); ++l)
      {
        fixedCoarse.push_back(toCoarseGrid(fixedLayers[l]));
        movingCoarse.push_back(toCoarseGrid(movingLayers[l]));
      }
      FcFix = torch::cat(fixedCoarse, 1);
      FcMov = torch::cat(movingCoarse, 1);
    }
    else
    {
      FcFix = toCoarseGrid(fixedT);
      FcMov = toCoarseGrid(movingT);
    }
    std::vector<int64_t> coarse(ImageDimension);
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      coarse[d] = FcFix.size(2 + static_cast<int64_t>(d));
      if (coarse[d] < 1)
      {
        itkExceptionMacro("ImpactCoarseRegistration: GridSpacing too large for the image size.");
      }
    }

    // ---- Candidate displacement table {Dim, L} (z,y,x), shared by forward & backward. ----
    const std::vector<int64_t> padHw(2 * ImageDimension, hw);
    std::vector<int64_t>       ssdShape;
    ssdShape.push_back(L);
    for (auto c : coarse)
    {
      ssdShape.push_back(c);
    }
    std::vector<float> meshBuffer(static_cast<size_t>(ImageDimension) * static_cast<size_t>(L)); // {Dim, L}, z,y,x
    for (int64_t l = 0; l < L; ++l)
    {
      int64_t rem = l;
      for (int a = static_cast<int>(ImageDimension) - 1; a >= 0; --a) // axis 0 most significant (x fastest)
      {
        meshBuffer[static_cast<size_t>(a) * L + l] = static_cast<float>((rem % L1) - hw);
        rem /= L1;
      }
    }
    torch::Tensor dispMesh =
      torch::from_blob(meshBuffer.data(), { static_cast<int64_t>(ImageDimension), L }, torch::kFloat32)
        .clone()
        .to(device);

    auto gather = [&](const torch::Tensor & indices) -> torch::Tensor {
      // indices {coarse...} long -> {1, Dim, coarse...}, coarse-voxel units, z,y,x.
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(ImageDimension));
      for (auto c : coarse)
      {
        shape.push_back(c);
      }
      return dispMesh.index_select(1, indices.reshape({ -1 })).reshape(shape).unsqueeze(0);
    };
    const std::vector<double> coeffs = { 0.003, 0.01, 0.03, 0.1, 0.3, 1.0 };
    std::vector<int64_t>      dmShape; // {Dim, L, 1, ..., 1} ((Dim-1) trailing singletons)
    dmShape.push_back(static_cast<int64_t>(ImageDimension));
    dmShape.push_back(L);
    for (unsigned int d = 1; d < ImageDimension; ++d)
    {
      dmShape.push_back(1);
    }

    // Solve the coarse problem for one (reference, to-shift) feature pair: discrete SSD cost
    // volume over the dense window + coupled-convex regularization -> {1, Dim, coarse...}.
    auto solveCoarse = [&](const torch::Tensor & Fref, const torch::Tensor & Fshift) -> torch::Tensor {
      torch::Tensor FshiftPad = F::pad(Fshift, F::PadFuncOptions(padHw).mode(torch::kReplicate));
      torch::Tensor ssd = torch::empty(ssdShape, Fref.options());
      for (int64_t l = 0; l < L; ++l)
      {
        torch::Tensor shifted = FshiftPad;
        int64_t       rem = l;
        for (int a = static_cast<int>(ImageDimension) - 1; a >= 0; --a)
        {
          const int64_t offset = (rem % L1) - hw;
          rem /= L1;
          shifted = shifted.narrow(2 + a, hw + offset, coarse[a]);
        }
        torch::Tensor cost = (Fref - shifted).pow(2).sum(1, /*keepdim=*/true);
        cost = smoothBox(cost);
        cost = smoothBox(cost); // two box-filter passes, as in ConvexAdam
        ssd[l] = cost.squeeze(0).squeeze(0);
      }
      torch::Tensor disp = smoothBox(gather(torch::argmin(ssd, 0)));
      for (double coeff : coeffs)
      {
        // Tiled over the first coarse axis so the (L x grid) penalty transient stays bounded.
        torch::Tensor       argmin = torch::empty(coarse, torch::TensorOptions().dtype(torch::kLong).device(device));
        const torch::Tensor dm = dispMesh.view(dmShape);
        const torch::Tensor dispC = disp.squeeze(0);
        for (int64_t i = 0; i < coarse[0]; ++i)
        {
          torch::Tensor dvi = dispC.narrow(1, i, 1);
          torch::Tensor penalty = (dm - dvi).pow(2).sum(0);
          torch::Tensor coupled = ssd.select(1, i) + coeff * penalty;
          argmin.select(0, i).copy_(torch::argmin(coupled, 0));
        }
        disp = smoothBox(gather(argmin));
      }
      return disp;
    };

    torch::Tensor disp = solveCoarse(FcFix, FcMov); // forward: fixed -> moving

    // ---- Optional inverse consistency: also solve the backward (moving->fixed) problem and
    // symmetrize the two fields toward mutual inverses in normalized [-1,1] grid coordinates
    // (ConvexAdam-style), for a more diffeomorphic coarse initialization. ----
    if (m_InverseConsistency)
    {
      torch::Tensor dispBack = solveCoarse(FcMov, FcFix); // backward: moving -> fixed

      // Per-channel normalization (coarse_size - 1)/2, in z,y,x channel order.
      std::vector<float> scaleVals(ImageDimension);
      for (unsigned int c = 0; c < ImageDimension; ++c)
      {
        scaleVals[c] = (coarse[c] > 1) ? static_cast<float>((coarse[c] - 1) / 2.0) : 1.0f;
      }
      std::vector<int64_t> scaleShape(ImageDimension + 2, 1);
      scaleShape[1] = static_cast<int64_t>(ImageDimension);
      torch::Tensor scaleT =
        torch::from_blob(scaleVals.data(), { static_cast<int64_t>(ImageDimension) }, torch::kFloat32)
          .clone()
          .to(device)
          .reshape(scaleShape);

      // Normalize to [-1,1] and flip channel order z,y,x -> x,y,z (grid_sample order).
      torch::Tensor f1 = (disp / scaleT).flip(1);
      torch::Tensor f2 = (dispBack / scaleT).flip(1);

      // Identity sampling grid at coarse resolution, channel-first {1, Dim, coarse...}, x,y,z.
      torch::Tensor idAffine =
        torch::eye(ImageDimension, torch::TensorOptions().dtype(torch::kFloat32).device(device));
      idAffine =
        torch::cat({ idAffine, torch::zeros({ static_cast<int64_t>(ImageDimension), 1 }, idAffine.options()) }, 1)
          .unsqueeze(0);
      std::vector<int64_t> gridSize;
      gridSize.push_back(1);
      gridSize.push_back(1);
      for (auto c : coarse)
      {
        gridSize.push_back(c);
      }
      torch::Tensor idGrid = torch::affine_grid_generator(idAffine, gridSize, /*align_corners=*/true);
      std::vector<int64_t> toChannelFirst;
      toChannelFirst.push_back(0);
      toChannelFirst.push_back(idGrid.dim() - 1);
      for (int64_t dd = 1; dd < idGrid.dim() - 1; ++dd)
      {
        toChannelFirst.push_back(dd);
      }
      torch::Tensor identity = idGrid.permute(toChannelFirst).contiguous(); // {1, Dim, coarse...}

      std::vector<int64_t> toChannelLast;
      toChannelLast.push_back(0);
      for (unsigned int dd = 0; dd < ImageDimension; ++dd)
      {
        toChannelLast.push_back(2 + static_cast<int64_t>(dd));
      }
      toChannelLast.push_back(1);
      const auto icSampleOpts =
        F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(true);

      // Symmetric fixed point: f1 <- 0.5 (f1 - f2 pulled back through id+f1), and vice versa.
      for (int it = 0; it < 15; ++it)
      {
        torch::Tensor f1i = f1.clone();
        torch::Tensor f2i = f2.clone();
        f1 = 0.5 * (f1i - F::grid_sample(f2i, (identity + f1i).permute(toChannelLast), icSampleOpts));
        f2 = 0.5 * (f2i - F::grid_sample(f1i, (identity + f2i).permute(toChannelLast), icSampleOpts));
      }
      // Back to coarse-voxel units, z,y,x channel order.
      disp = f1.flip(1) * scaleT;
    }

    // ---- Upsample to full resolution; coarse-voxel -> full-resolution voxel units. ----
    // Interpolate with an EXACT gs stretch (target coarse*gs) so the displacement-magnitude scale (gs)
    // matches the positional stretch. Otherwise, when a dimension is not divisible by gs, interpolate
    // would stretch by full/coarse != gs and shear the field toward the high-index border. Tail voxels
    // the coarse avg-pool dropped are then edge-padded.
    std::vector<int64_t> exactSize(ImageDimension);
    bool                 needPad = false;
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      exactSize[d] = coarse[d] * gs;
      if (exactSize[d] != spatial[d])
      {
        needPad = true;
      }
    }
    torch::Tensor dispFull;
    if constexpr (ImageDimension == 3)
      dispFull = F::interpolate(
        disp * static_cast<double>(gs),
        F::InterpolateFuncOptions().size(exactSize).mode(torch::kTrilinear).align_corners(false));
    else
      dispFull = F::interpolate(
        disp * static_cast<double>(gs),
        F::InterpolateFuncOptions().size(exactSize).mode(torch::kBilinear).align_corners(false));
    if (needPad)
    {
      // F::pad pads from the last (x) axis inward; high side only (coarse*gs <= full), replicate.
      std::vector<int64_t> pad;
      for (int a = static_cast<int>(ImageDimension) - 1; a >= 0; --a)
      {
        pad.push_back(0);
        pad.push_back(spatial[a] - exactSize[a]);
      }
      dispFull = F::pad(dispFull, F::PadFuncOptions(pad).mode(torch::kReplicate));
    } // dispFull: {1, Dim, spatial...}, full-res voxel units, z,y,x

    // ---- Geometry-correct writeback (shared convention). ----
    DisplacementFieldType * output = this->GetOutput();
    output->SetRegions(m_FixedImage->GetLargestPossibleRegion());
    output->SetSpacing(m_FixedImage->GetSpacing());
    output->SetOrigin(m_FixedImage->GetOrigin());
    output->SetDirection(m_FixedImage->GetDirection());
    output->Allocate();

    Impact::WriteVoxelFieldToDisplacement<ImageDimension>(
      dispFull, m_FixedImage->GetSpacing(), m_FixedImage->GetDirection(), output);

    m_DisplacementFieldTransform = DisplacementFieldTransformType::New();
    m_DisplacementFieldTransform->SetDisplacementField(output);
  }
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Device: " << m_Device << std::endl;
  os << indent << "GridSpacing: " << m_GridSpacing << std::endl;
  os << indent << "DisplacementHalfWidth: " << m_DisplacementHalfWidth << std::endl;
  os << indent << "FixedModelsConfiguration count: " << m_FixedModelsConfiguration.size() << std::endl;
}

} // end namespace itk

#endif // itkImpactCoarseRegistration_hxx

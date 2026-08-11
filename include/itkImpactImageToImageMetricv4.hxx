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

#ifndef itkImpactImageToImageMetricv4_hxx
#define itkImpactImageToImageMetricv4_hxx

#include "itkImpactImageToImageMetricv4GetValueAndDerivativeThreader.h"
#include "itkInterpolateVectorImageFunction.h"
#include "itkImageToFeaturesMap.h"
#include "itkImageToFeaturesMapInternals.h"
#include <itkImageFileWriter.h>

namespace itk
{

// Torch-dependent state of the metric, kept out of the public header (see the .h).
// Definition of the opaque ImpactImageToImageMetricv4::Internals struct.
template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage,
          typename TInternalComputationValueType,
          typename TMetricTraits>
struct ImpactImageToImageMetricv4<TFixedImage,
                                  TMovingImage,
                                  TVirtualImage,
                                  TInternalComputationValueType,
                                  TMetricTraits>::Internals
{
  /** Interpolator for feature maps (vector-valued), using scalar B-spline interpolation. */
  using BSplineInterpolateVectorImageFunction =
    itk::InterpolateVectorImageFunction<FeaturesImageType, itk::BSplineInterpolateImageFunction<TFixedImage, float, float>>;

  /** A feature map image together with its interpolator. */
  struct FeaturesMap
  {
    typename FeaturesImageType::ConstPointer                m_FeaturesMap;
    typename BSplineInterpolateVectorImageFunction::Pointer m_FeaturesMapInterpolator;

    FeaturesMap(typename FeaturesImageType::ConstPointer featuresMap)
      : m_FeaturesMap(featuresMap)
    {
      this->m_FeaturesMapInterpolator = BSplineInterpolateVectorImageFunction::New();
      typename FeaturesImageType::Pointer nonConstPtr = const_cast<FeaturesImageType *>(featuresMap.GetPointer());
      this->m_FeaturesMapInterpolator->SetInputImage(nonConstPtr);
    }
  };

  std::vector<FeaturesMap>                m_FixedFeaturesMaps;
  std::vector<FeaturesMap>                m_MovingFeaturesMaps;
  std::vector<std::vector<torch::Tensor>> m_Principal_components;
};

template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage,
          typename TInternalComputationValueType,
          typename TMetricTraits>
ImpactImageToImageMetricv4<TFixedImage,
                                TMovingImage,
                                TVirtualImage,
                                TInternalComputationValueType,
                                TMetricTraits>::ImpactImageToImageMetricv4()
  : m_Internals(std::make_shared<Internals>())
{
  // We have our own GetValueAndDerivativeThreader's that we want
  // ImageToImageMetricv4 to use.
  using ImpactDenseGetValueAndDerivativeThreaderType =
    ImpactImageToImageMetricv4GetValueAndDerivativeThreader<
      ThreadedImageRegionPartitioner<Superclass::VirtualImageDimension>,
      Superclass,
      Self>;
  using ImpactSparseGetValueAndDerivativeThreaderType =
    ImpactImageToImageMetricv4GetValueAndDerivativeThreader<ThreadedIndexedContainerPartitioner, Superclass, Self>;
  this->m_DenseGetValueAndDerivativeThreader = ImpactDenseGetValueAndDerivativeThreaderType::New();
  this->m_SparseGetValueAndDerivativeThreader = ImpactSparseGetValueAndDerivativeThreaderType::New();
}

template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage,
          typename TInternalComputationValueType,
          typename TMetricTraits>
void
ImpactImageToImageMetricv4<TFixedImage,
                                TMovingImage,
                                TVirtualImage,
                                TInternalComputationValueType,
                                TMetricTraits>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << "\nFixed : ";
  for (int i = 0; i < this->GetFixedModelsConfiguration().size(); ++i)
  {
    os << "\n\tModel(" << i << ") : \n" << this->GetFixedModelsConfiguration()[i];
  }
  os << "\nMoving : ";
  for (int i = 0; i < this->GetMovingModelsConfiguration().size(); ++i)
  {
    os << "\n\tModel(" << i << ") : \n" << this->GetMovingModelsConfiguration()[i];
  }

  os << "\nSubsetFeatures: " << GetStringFromVector<unsigned int>(this->GetSubsetFeatures())
      << "\nPCA: " << GetStringFromVector<unsigned int>(this->GetPCA())
      << "\nLayersWeight: " << GetStringFromVector<float>(this->GetLayersWeight())
      << "\nDistance: " << GetStringFromVector<std::string>(this->GetDistance()) << "\nMode: " << this->GetMode()
      << "\nDevice: " << this->GetDevice() << "\nSeed: " << this->GetSeed();

  if (this->GetMode() == "Static")
  {
    os << "\nFeaturesMapUpdateInterval: " << this->GetFeaturesMapUpdateInterval() << "\nFeatureMapsPath: " << this->GetFeatureMapsPath();
  }
  else
  {
    os << "\nBatchSize: " << this->GetBatchSize();
  }
}

template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage,
          typename TInternalComputationValueType,
          typename TMetricTraits>
template <typename TFeaturesMap, typename TImage>
std::vector<TFeaturesMap>
ImpactImageToImageMetricv4<TFixedImage,
                                TMovingImage,
                                TVirtualImage,
                                TInternalComputationValueType,
                                TMetricTraits>::GetFeaturesMaps(
                                typename TImage::ConstPointer image,
                                const std::vector<ModelConfiguration> & modelsConfiguration,
                                std::function<typename TImage::PointType(const typename TImage::PointType &)> fct){
  using TensorToImageFilterType = itk::TensorToImageFilter<TImage::ImageDimension>;
  using WriterType = itk::ImageFileWriter<FeaturesImageType>;
  using InterpolatorType = itk::BSplineInterpolateImageFunction<TImage, double>;
  using ImageToFeaturesMapType = itk::ImageToFeaturesMap<TImage, InterpolatorType>;

  std::vector<TFeaturesMap> featuresMaps;
  
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);

  for(unsigned int i = 0; i <  modelsConfiguration.size(); ++i){
    typename ImageToFeaturesMapType::Pointer imageToFeaturesMap = ImageToFeaturesMapType::New();
    imageToFeaturesMap->SetModelConfiguration(modelsConfiguration[i]);
    imageToFeaturesMap->SetInterpolator(interpolator);
    // The moving image is sampled through the transform, so a refresh sees the anatomy as it
    // currently aligns rather than as acquired. This argument used to be taken and dropped,
    // which left the map frozen and SetFeaturesMapUpdateInterval() with nothing to act on.
    // With an identity transform -- the usual state at Initialize() -- it changes nothing.
    if (fct)
    {
      imageToFeaturesMap->SetTransform(fct);
    }
    imageToFeaturesMap->AddInput(image);
    imageToFeaturesMap->SetPCA(m_PCA[i]);
    imageToFeaturesMap->SetDevice(this->m_Device);
    // The moving image (fct is set) reuses the PCA basis fitted on the fixed image,
    // so fixed and moving features are projected onto the same basis.
    if (fct && i < this->m_Internals->m_Principal_components.size())
    {
      SetPrincipalComponents(*imageToFeaturesMap, this->m_Internals->m_Principal_components[i]);
    }
    imageToFeaturesMap->Update();
    if (!fct)
    {
      if (this->m_Internals->m_Principal_components.size() <= i)
      {
        this->m_Internals->m_Principal_components.resize(modelsConfiguration.size());
      }
      this->m_Internals->m_Principal_components[i] = GetPrincipalComponents(*imageToFeaturesMap);
    }
    if (!this->GetFeatureMapsPath().empty())
    {
      typename WriterType::Pointer writer = WriterType::New();
      std::string filename;

      for (int it = 0; it < modelsConfiguration[i].GetVoxelSize().size(); ++it)
      {
        if (it > 0)
          filename += "_";

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << modelsConfiguration[i].GetVoxelSize()[it];
        filename += oss.str();
      }
      filename += "mm";

      writer->SetFileName(this->GetFeatureMapsPath() + (fct ? "/Moving_" : "/Fixed_")+ filename + ".mha");
      typename TensorToImageFilterType::Pointer tensorToImageFilter = TensorToImageFilterType::New();
      tensorToImageFilter->SetTensor(GetTensorInput(*imageToFeaturesMap, 0));
      tensorToImageFilter->SetReferenceImage(image);
      tensorToImageFilter->Update();
      writer->SetInput(tensorToImageFilter->GetOutput());
      try
      {
        writer->Update();
      }
      catch (itk::ExceptionObject & error)
      {
        itkGenericExceptionMacro("Error writing image: " << writer->GetFileName() << " ITK Exception: " << error);
      }
    }

    // One map per KEPT layer, in mask order: comparing several depths of the same network is
    // what LayersMask asks for, and the filter has already produced them all. They are appended
    // to one flat list across models, which is the order Distance, LayersWeight and
    // SubsetFeatures are read in -- the dumped file is named by that same flat index.
    for (unsigned int layer = 0; layer < NumberOfKeptLayers(modelsConfiguration[i]); ++layer)
    {
      const auto featuresMap = imageToFeaturesMap->GetOutput(layer);
      if (!this->GetFeatureMapsPath().empty())
      {
        typename WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(this->GetFeatureMapsPath() + (fct ? "/Moving_" : "/Fixed_") +
                            std::to_string(featuresMaps.size()) + ".mha");
        writer->SetInput(featuresMap);
        try
        {
          writer->Update();
        }
        catch (itk::ExceptionObject & error)
        {
          itkGenericExceptionMacro("Error writing image: " << writer->GetFileName() << " ITK Exception: " << error);
        }
      }
      featuresMaps.emplace_back(featuresMap);
    }
  }
  return featuresMaps;
}

template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage,
          typename TInternalComputationValueType,
          typename TMetricTraits>
void
ImpactImageToImageMetricv4<TFixedImage,
                                TMovingImage,
                                TVirtualImage,
                                TInternalComputationValueType,
                                TMetricTraits>::UpdateMovingFeaturesMaps()
{
  // Only Static holds a precomputed map; the online modes extract per point and have nothing
  // to refresh.
  if (this->GetMode() != "Static")
  {
    return;
  }
  this->m_Internals->m_MovingFeaturesMaps =
    GetFeaturesMaps<typename Internals::FeaturesMap, TMovingImage>(
      this->m_MovingImage.GetPointer(),
      m_MovingModelsConfiguration,
      std::function<typename TMovingImage::PointType(const typename TMovingImage::PointType &)>(
        [this](const typename TMovingImage::PointType & point) {
          return this->GetTransform()->TransformPoint(point);
        }));
}

template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage,
          typename TInternalComputationValueType,
          typename TMetricTraits>
void
ImpactImageToImageMetricv4<TFixedImage,
                                TMovingImage,
                                TVirtualImage,
                                TInternalComputationValueType,
                                TMetricTraits>::GetValueAndDerivative(typename Superclass::MeasureType &    value,
                                                                      typename Superclass::DerivativeType & derivative) const
{
  Superclass::GetValueAndDerivative(value, derivative);
  // After the evaluation, so the refresh lands between two iterations rather than in the middle
  // of one -- the value and the derivative just returned belong to the same feature map.
  ++this->m_CurrentIteration;
  if (this->GetMode() == "Static" && this->GetFeaturesMapUpdateInterval() > 0 &&
      this->m_CurrentIteration % static_cast<unsigned long>(this->GetFeaturesMapUpdateInterval()) == 0)
  {
    const_cast<Self *>(this)->UpdateMovingFeaturesMaps();
  }
}

template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage,
          typename TInternalComputationValueType,
          typename TMetricTraits>
void
ImpactImageToImageMetricv4<TFixedImage,
                                TMovingImage,
                                TVirtualImage,
                                TInternalComputationValueType,
                                TMetricTraits>::Initialize(){
  Superclass::Initialize();
  this->m_features_indexes.clear();

  // Every per-layer vector is indexed by the loop over compared layers, with no bound check in
  // the hot path: too short and it reads past the end, too long and the value sums losses that
  // no point ever updated. Both are silent, and the second is exactly what a configuration
  // written for the elastix component looks like, where these vectors are already per layer.
  const unsigned int comparedLayers = std::accumulate(
    m_FixedModelsConfiguration.begin(),
    m_FixedModelsConfiguration.end(),
    0u,
    [](unsigned int total, const ModelConfiguration & config) { return total + NumberOfKeptLayers(config); });
  auto checkPerLayer = [&](const char * name, size_t size) {
    if (size != comparedLayers)
    {
      itkExceptionMacro(<< name << " has " << size << " entries but " << comparedLayers
                        << " layers are compared (one per kept layer of every model, summed over "
                        << m_FixedModelsConfiguration.size() << " models). They carry one entry each.");
    }
  };
  checkPerLayer("Distance", this->m_Distance.size());
  checkPerLayer("LayersWeight", this->m_LayersWeight.size());
  checkPerLayer("SubsetFeatures", this->m_SubsetFeatures.size());
  // PCA is a property of the feature extraction, not of a layer: it is set once per model.
  if (this->m_PCA.size() != m_FixedModelsConfiguration.size())
  {
    itkExceptionMacro("PCA has " << this->m_PCA.size() << " entries but there are "
                                 << m_FixedModelsConfiguration.size() << " models; it carries one entry per model.");
  }

  if (this->GetMode() == "Static")
  {
    auto & fixedFeaturesMaps = this->m_Internals->m_FixedFeaturesMaps;
    auto & movingFeaturesMaps = this->m_Internals->m_MovingFeaturesMaps;
    fixedFeaturesMaps.clear();
    movingFeaturesMaps.clear();
    this->m_Internals->m_Principal_components.clear();

    fixedFeaturesMaps = GetFeaturesMaps<typename Internals::FeaturesMap, TFixedImage>(this->m_FixedImage.GetPointer(), m_FixedModelsConfiguration);
    // The fixed map has to exist first: it is what fits the PCA basis the moving map reuses.
    this->m_CurrentIteration = 0;
    this->UpdateMovingFeaturesMaps();

    if (fixedFeaturesMaps.size() != movingFeaturesMaps.size())
    {
      itkExceptionMacro("Mismatch in number of feature maps: "
                        << "fixedFeaturesMaps.size() = " << fixedFeaturesMaps.size()
                        << ", movingFeaturesMaps.size() = " << movingFeaturesMaps.size());
    }

    for (int i = 0; i < fixedFeaturesMaps.size(); ++i)
    {
      if (fixedFeaturesMaps[i].m_FeaturesMap->GetNumberOfComponentsPerPixel() !=
          movingFeaturesMaps[i].m_FeaturesMap->GetNumberOfComponentsPerPixel())
      {
        itkExceptionMacro(
          "Mismatch in number of components per feature map at layer "
          << i << ": fixed = " << fixedFeaturesMaps[i].m_FeaturesMap->GetNumberOfComponentsPerPixel()
          << ", moving = " << movingFeaturesMaps[i].m_FeaturesMap->GetNumberOfComponentsPerPixel());
      }
    }

    for (int i = 0; i < fixedFeaturesMaps.size(); ++i)
    {
      int numComponents = fixedFeaturesMaps[i].m_FeaturesMap->GetNumberOfComponentsPerPixel();
      this->m_SubsetFeatures[i] = std::clamp<unsigned int>(this->GetSubsetFeatures()[i], 1, numComponents);
      this->m_features_indexes.push_back(std::vector<unsigned int>(numComponents));
      std::iota(this->m_features_indexes[i].begin(), this->m_features_indexes[i].end(), 0);
    }
  }
  else if (this->GetMode() == "Jacobian")
  {
    // Online mode: nothing is precomputed; the threader extracts a patch per sampled
    // point and backpropagates through the model (see ProcessPoint). Here we only move
    // each model to the device once and learn the per-model feature-channel count from a
    // single dummy forward, mirroring how Static reads GetNumberOfComponentsPerPixel().
    this->m_Internals->m_FixedFeaturesMaps.clear();
    this->m_Internals->m_MovingFeaturesMaps.clear();
    this->m_Internals->m_Principal_components.clear();

    if (m_FixedModelsConfiguration.size() != m_MovingModelsConfiguration.size())
    {
      itkExceptionMacro("Jacobian mode: number of fixed (" << m_FixedModelsConfiguration.size() << ") and moving ("
                                                           << m_MovingModelsConfiguration.size()
                                                           << ") model configurations differ.");
    }

    const torch::Device device(this->m_Device);
    // Channel count of every KEPT layer, in mask order, from one dummy forward -- the online
    // counterpart of Static reading GetNumberOfComponentsPerPixel() off each feature map.
    // Empty if the patch size is not strictly positive (required online), if no layer is kept,
    // or if the model returns fewer outputs than the mask claims.
    auto featureChannels = [&device](const ModelConfiguration & config) -> std::vector<int64_t> {
      ModelTo(config, device);
      const std::vector<int64_t> & patchSize = config.GetPatchSize();
      std::vector<int64_t>         shape = { 1, static_cast<int64_t>(config.GetNumberOfChannels()) };
      // Reversed onto the tensor axes, exactly as the threader builds the real patch: the probe
      // has to ask the model the same question the inference will, or an anisotropic patch is
      // validated at a shape that is never used.
      for (unsigned int d = 0; d < config.GetDimension(); ++d)
      {
        if (d >= patchSize.size() || patchSize[d] <= 0)
          return {};
        shape.push_back(patchSize[config.GetDimension() - 1 - d]);
      }
      torch::NoGradGuard ng;
      torch::Tensor      dummy =
        torch::zeros(shape, torch::TensorOptions().dtype(GetModelDtype(config)).device(device));
      auto         outputs = GetModel(config).forward({ dummy }).toList().vec();
      const auto & mask = config.GetLayersMask();
      if (outputs.size() < mask.size())
      {
        return {};
      }
      std::vector<int64_t> channels;
      for (size_t it = 0; it < mask.size(); ++it)
      {
        if (mask[it])
        {
          channels.push_back(outputs[it].toTensor().size(1));
        }
      }
      return channels;
    };

    for (unsigned int i = 0; i < m_FixedModelsConfiguration.size(); ++i)
    {
      const std::vector<int64_t> fixedChannels = featureChannels(m_FixedModelsConfiguration[i]);
      const std::vector<int64_t> movingChannels = featureChannels(m_MovingModelsConfiguration[i]);
      if (fixedChannels.empty())
      {
        itkExceptionMacro("Jacobian mode requires a strictly positive patch size in every dimension, at least one "
                          "kept layer, and a model returning at least as many outputs as LayersMask has entries "
                          "(model " << i << ").");
      }
      if (fixedChannels != movingChannels)
      {
        itkExceptionMacro("Jacobian mode: fixed/moving feature-channel mismatch at model " << i << '.');
      }
      // One entry per kept layer, appended to the same flat list Static builds, so both modes
      // read Distance, LayersWeight and SubsetFeatures at the same index.
      for (const int64_t channels : fixedChannels)
      {
        const size_t comparison = this->m_features_indexes.size();
        this->m_SubsetFeatures[comparison] =
          std::clamp<unsigned int>(this->GetSubsetFeatures()[comparison], 1, static_cast<unsigned int>(channels));
        this->m_features_indexes.push_back(std::vector<unsigned int>(channels));
        std::iota(this->m_features_indexes[comparison].begin(), this->m_features_indexes[comparison].end(), 0);
      }
    }
  }
  else
  {
    itkExceptionMacro("Only the \"Static\" and \"Jacobian\" modes are implemented for this metric; got \""
                      << this->GetMode() << "\".");
  }
}


} // end namespace itk


#endif

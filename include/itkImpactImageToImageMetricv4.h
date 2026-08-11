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
#ifndef itkImpactImageToImageMetricv4_h
#define itkImpactImageToImageMetricv4_h

// Intentionally free of any LibTorch dependency so it can be parsed by castxml and
// exposed to Python (WrapITK). The feature maps, interpolators, PCA bases and the
// inference/loss machinery live behind an opaque Internals struct defined in the .hxx;
// the threader and feature-extraction headers (which pull in torch) are included only
// from the .hxx.

#include <itkImageToImageMetricv4.h>
#include <itkDefaultImageToImageMetricTraitsv4.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkVectorImage.h>
#include <itkModelConfiguration.h>
#include <functional>
#include <memory>

namespace itk
{

// Forward declaration so the metric can befriend the threader without pulling in its
// (torch-dependent) header here.
template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
class ImpactImageToImageMetricv4GetValueAndDerivativeThreader;

/** \class ImpactImageToImageMetricv4
 *
 *  \brief Semantic similarity metric comparing internal features of pretrained
 *  TorchScript models (IMPACT) for multimodal image registration.
 *
 *  This class supports vector images of type VectorImage
 *  and Image< VectorType, imageDimension >.
 *
 *  See
 *  ImpactImageToImageMetricv4GetValueAndDerivativeThreader::ProcessPoint for algorithm implementation.
 *
 * \ingroup Impact
 */
template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage = TFixedImage,
          typename TInternalComputationValueType = double,
          typename TMetricTraits =
            DefaultImageToImageMetricTraitsv4<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType>>
class ITK_TEMPLATE_EXPORT ImpactImageToImageMetricv4
  : public ImageToImageMetricv4<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImpactImageToImageMetricv4);

  /** Standard class type aliases. */
  using Self = ImpactImageToImageMetricv4;
  using Superclass =
    ImageToImageMetricv4<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(ImpactImageToImageMetricv4);

  using typename Superclass::DerivativeType;

  using typename Superclass::FixedImagePointType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::FixedImageGradientType;

  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageGradientType;

  using typename Superclass::MovingTransformType;
  using typename Superclass::JacobianType;
  using VirtualImageType = typename Superclass::VirtualImageType;
  using typename Superclass::VirtualIndexType;
  using typename Superclass::VirtualPointType;
  using typename Superclass::VirtualPointSetType;

  /* Image dimension accessors */
  static constexpr typename TVirtualImage::ImageDimensionType VirtualImageDimension = TVirtualImage::ImageDimension;
  static constexpr typename TFixedImage::ImageDimensionType   FixedImageDimension = TFixedImage::ImageDimension;
  static constexpr typename TMovingImage::ImageDimensionType  MovingImageDimension = TMovingImage::ImageDimension;

  /** Set/Get the TorchScript model configurations used to extract features from the fixed
   * image. Each model may target a different resolution, architecture or semantic level.
   */
  itkSetMacro(FixedModelsConfiguration, std::vector<ModelConfiguration>);
  itkGetConstReferenceMacro(FixedModelsConfiguration, std::vector<ModelConfiguration>);

  /** Set/Get the TorchScript model configurations used to extract features from the moving
   * image. Distinct fixed/moving models support asymmetric or multimodal setups.
   */
  itkSetMacro(MovingModelsConfiguration, std::vector<ModelConfiguration>);
  itkGetConstReferenceMacro(MovingModelsConfiguration, std::vector<ModelConfiguration>);

  void
  SetModelsConfiguration(std::vector<ModelConfiguration> & modelsConfiguration)
  {
    SetFixedModelsConfiguration(modelsConfiguration);
    SetMovingModelsConfiguration(modelsConfiguration);
  }

  /** Append a single model configuration. Convenience for callers (e.g. Python) that add
   * configurations one at a time instead of passing a std::vector. */
  void
  AddFixedModelConfiguration(const ModelConfiguration & configuration)
  {
    m_FixedModelsConfiguration.push_back(configuration);
    this->Modified();
  }
  void
  AddMovingModelConfiguration(const ModelConfiguration & configuration)
  {
    m_MovingModelsConfiguration.push_back(configuration);
    this->Modified();
  }
  /** Append the same configuration to both the fixed and moving lists. */
  void
  AddModelConfiguration(const ModelConfiguration & configuration)
  {
    AddFixedModelConfiguration(configuration);
    AddMovingModelConfiguration(configuration);
  }

  /** Set/Get the subset of feature channels used in the loss (per layer), for
   * dimensionality reduction or focusing on the most informative channels.
   */
  itkSetMacro(SubsetFeatures, std::vector<unsigned int>);
  itkGetConstMacro(SubsetFeatures, std::vector<unsigned int>);

  /** Set/Get the weight applied to each layer's loss contribution, to balance layers of
   * different semantic granularity.
   */
  itkSetMacro(LayersWeight, std::vector<float>);
  itkGetConstMacro(LayersWeight, std::vector<float>);

  /** Set/Get the loss function per layer (e.g. "l1", "cosine", "ncc"); heterogeneous
   * losses adapt to the nature of each feature representation.
   */
  itkSetMacro(Distance, std::vector<std::string>);
  itkGetConstMacro(Distance, std::vector<std::string>);

  /** Set/Get the number of principal components to keep per layer (PCA on the feature
   * maps). 0 disables PCA.
   */
  itkSetMacro(PCA, std::vector<unsigned int>);
  itkGetConstMacro(PCA, std::vector<unsigned int>);

  /** Set/Get the device for all model inference and tensor operations, as a string
   * ("cpu", "cuda", "cuda:0", ...).
   */
  itkSetMacro(Device, std::string);
  itkGetConstMacro(Device, std::string);

  /** Set/Get the directory where feature maps are written (empty disables the dump).
   * Used for debugging/inspection of the extracted features.
   */
  itkSetMacro(FeatureMapsPath, std::string);
  itkGetConstMacro(FeatureMapsPath, std::string);

  /** Set/Get the mode of operation:
   * - "Static": features are precomputed as full maps and interpolated per point.
   * - "Jacobian": online per-point patch extraction with backpropagation through the model.
   */
  itkSetMacro(Mode, std::string);
  itkGetConstMacro(Mode, std::string);

  /** Set/Get the RNG seed for feature-subset sampling (0 seeds from the clock). */
  itkSetMacro(Seed, unsigned int);
  itkGetConstMacro(Seed, unsigned int);

  /** Set/Get how many sampled points the online ("Jacobian") mode runs through a model at
   * once. Online mode cuts a patch around every point and pushes it through the network, so
   * evaluating points one at a time costs one forward and one backward pass each; a batch
   * amortises both over all of its points, which is what makes the mode affordable. The batch
   * also bounds the memory: it holds one patch and the model's activations for each of its
   * points, so lower it for large patches and raise it for small ones. Ignored in Static
   * mode, which interpolates a precomputed map instead of running the model.
   *
   * The default is not a compromise between speed and memory so much as a floor. A batch of a
   * few points makes tensors that are large enough for LibTorch to hand its ops to the whole
   * intra-op thread pool and far too small to pay for the barriers, which measured two orders
   * of magnitude slower than either a batch of one or a batch past a few dozen. Values well
   * below the default are best avoided whatever the memory. */
  itkSetMacro(BatchSize, unsigned int);
  itkGetConstMacro(BatchSize, unsigned int);

  /** Set/Get how often the moving feature map is recomputed, in derivative evaluations.
   * Zero or negative disables it, which is the default and the behaviour of every release
   * before this one.
   *
   * Static mode extracts the moving features once, from the moving image as acquired, and then
   * samples that frozen map at the transformed point. The network therefore never sees the
   * anatomy as it currently aligns. A refresh re-extracts the map with the transform of the
   * moment applied, which is what makes a hybrid between Static's cost and the online mode's
   * fidelity.
   *
   * The interval counts GetValueAndDerivative() calls, which is one per iteration for a
   * gradient optimizer -- the metric has no other notion of an iteration. A value-only
   * GetValue(), as a line search issues, does not count.
   *
   * The refreshed map keeps the moving image's grid and is still sampled at the transformed
   * point, matching the elastix component. That is what preserves the derivative: the moving
   * features have to be read at a parameter-dependent point for a gradient to exist at all,
   * and the only such point is the transformed one. It also means the transform is applied
   * once when building the map and once when reading it, so the refreshed features answer for
   * a point further along than the one being scored. The approximation is exact at the moment
   * of the refresh, when the map is fresh, and degrades as the transform moves away from it --
   * shorter intervals keep it closer.
   */
  itkSetMacro(FeaturesMapUpdateInterval, int);
  itkGetConstMacro(FeaturesMapUpdateInterval, int);

  /** Recompute the moving feature map with the current transform applied, reusing the PCA
   * basis fitted on the fixed image. Called on the schedule GetFeaturesMapUpdateInterval()
   * sets; call it directly to drive the refresh from a host framework's own iteration hook.
   * Does nothing outside Static mode, which is the only mode holding a precomputed map. */
  void
  UpdateMovingFeaturesMaps();

  /** Counts the evaluation and refreshes the moving feature map when one is due, then defers
   * to the base metric. */
  void
  GetValueAndDerivative(typename Superclass::MeasureType &    value,
                        typename Superclass::DerivativeType & derivative) const override;

  void
  Initialize() override;

protected:
  ImpactImageToImageMetricv4();
  ~ImpactImageToImageMetricv4() override = default;

  friend class ImpactImageToImageMetricv4GetValueAndDerivativeThreader<
    ThreadedImageRegionPartitioner<Superclass::VirtualImageDimension>,
    Superclass,
    Self>;
  friend class ImpactImageToImageMetricv4GetValueAndDerivativeThreader<ThreadedIndexedContainerPartitioner,
                                                                       Superclass,
                                                                       Self>;

  /** Vector-valued feature-map image type; the per-layer maps and their interpolators
   * live in the torch-dependent Internals. */
  using FeaturesImageType = VectorImage<float, FixedImageDimension>;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Opaque, torch-dependent state (feature maps, interpolators, PCA bases), defined in
   * the .hxx. The threader reaches the feature maps through it. */
  struct Internals;
  std::shared_ptr<Internals> m_Internals;

  /** Build the per-layer feature maps for an image (fct maps points through the moving
   * transform). Templated on the concrete FeaturesMap type (Internals::FeaturesMap) so
   * this declaration carries no torch types. Defined in the .hxx. */
  template <typename TFeaturesMap, typename TImage>
  std::vector<TFeaturesMap>
  GetFeaturesMaps(typename TImage::ConstPointer                                                image,
                  const std::vector<ModelConfiguration> &                                      modelsConfiguration,
                  std::function<typename TImage::PointType(const typename TImage::PointType &)> fct = nullptr);

private:
  std::vector<ModelConfiguration> m_FixedModelsConfiguration;
  std::vector<ModelConfiguration> m_MovingModelsConfiguration;

  std::vector<unsigned int> m_SubsetFeatures;
  std::vector<unsigned int> m_PCA;
  std::vector<float>        m_LayersWeight;
  std::vector<std::string>  m_Distance;
  int                       m_FeaturesMapUpdateInterval{ 0 };
  /** Derivative evaluations since Initialize(), the metric's stand-in for an iteration count.
   * Mutable because GetValueAndDerivative() is const, as the base metric declares it. */
  mutable unsigned long     m_CurrentIteration{ 0 };
  std::string               m_Mode;
  std::string               m_FeatureMapsPath;
  std::string               m_Device = "cpu";
  // Zero means "seed from the clock", which is what the per-work-unit generator does with it.
  // It must have a value even when the user never calls SetSeed(): it is read on every
  // evaluation, and an indeterminate one would make the metric unreproducible at random.
  unsigned int              m_Seed{ 0 };
  unsigned int              m_BatchSize{ 64 };

  std::vector<std::vector<unsigned int>> m_features_indexes;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImpactImageToImageMetricv4.hxx"
#endif


#endif

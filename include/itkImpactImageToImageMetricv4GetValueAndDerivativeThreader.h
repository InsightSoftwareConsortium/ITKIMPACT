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

#ifndef itkImpactImageToImageMetricv4GetValueAndDerivativeThreader_h
#define itkImpactImageToImageMetricv4GetValueAndDerivativeThreader_h

#include <itkImageToImageMetricv4GetValueAndDerivativeThreader.h>
#include "ImpactLoss.h"
#include <random>

namespace itk
{

/** \class ImpactImageToImageMetricv4GetValueAndDerivativeThreader
 * \brief Processes points for ImpactImageToImageMetricv4 \c
 * GetValueAndDerivative.
 *
 * \ingroup Impact
 */
template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
class ITK_TEMPLATE_EXPORT ImpactImageToImageMetricv4GetValueAndDerivativeThreader
  : public ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImpactImageToImageMetricv4GetValueAndDerivativeThreader);

  /** Standard class type aliases. */
  using Self = ImpactImageToImageMetricv4GetValueAndDerivativeThreader;
  using Superclass = ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkOverrideGetNameOfClassMacro(ImpactImageToImageMetricv4GetValueAndDerivativeThreader);

  itkNewMacro(Self);

  using typename Superclass::DomainType;
  using typename Superclass::AssociateType;

  using ImageToImageMetricv4Type = typename Superclass::ImageToImageMetricv4Type;
  using typename Superclass::VirtualImageType;
  using typename Superclass::VirtualPointType;
  using typename Superclass::VirtualIndexType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::FixedImageGradientType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageGradientType;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;
  using typename Superclass::NumberOfParametersType;

protected:
  ImpactImageToImageMetricv4GetValueAndDerivativeThreader();
  ~ImpactImageToImageMetricv4GetValueAndDerivativeThreader() override = default;

  /** Allocate and initialize the per-work-unit loss accumulators. */
  void
  BeforeThreadedExecution() override;

  /** Reduce the per-work-unit loss accumulators and write the resulting value and
   *  derivative into the associated metric. */
  void
  AfterThreadedExecution() override;


  /** Compute the local, per-point contribution to the metric value and derivative. */
  bool
  ProcessPoint(const VirtualIndexType &        virtualIndex,
               const VirtualPointType &        virtualPoint,
               const FixedImagePointType &     mappedFixedPoint,
               const FixedImagePixelType &     fixedImageValue,
               const FixedImageGradientType &  mappedFixedImageGradient,
               const MovingImagePointType &    mappedMovingPoint,
               const MovingImagePixelType &    movingImageValue,
               const MovingImageGradientType & movingImageGradient,
               MeasureType &                   metricValueReturn,
               DerivativeType &                localDerivativeReturn,
               const ThreadIdType              threadId) const override;

  std::vector<unsigned int> GetSubsetOfFeatures(const std::vector<unsigned int> & features_index,
                      std::mt19937 &                    randomGenerator,
                      int n) const; 

  /**
   * \brief Per-work-unit accumulator of loss values and gradients, one loss object per
   * kept layer, for weighted multi-layer aggregation.
   *
   * \details Each work unit owns its own instance so parallel loss/gradient accumulation
   * is thread-safe; operator+= reduces one instance into another.
   */
  struct LossPerThreadStruct
  {
    std::vector<std::unique_ptr<itk::Impact::Loss>> m_losses;
    std::vector<float>                             m_layersWeight;
    SizeValueType                                  m_numberOfPixelsCounted;
    int                                            m_nb_parameters;
    std::mt19937                                   m_randomGenerator;

    void
    init(std::vector<std::string> distance_name, std::vector<float> layersWeight, unsigned int seed, unsigned int nb_parameters)
    {
      if (seed > 0)
      {
        this->m_randomGenerator = std::mt19937(seed);
      }
      else
      {
        this->m_randomGenerator = std::mt19937(time(nullptr));
      }
      this->m_layersWeight = layersWeight;
      for (std::string name : distance_name)
      {
        m_losses.push_back(itk::Impact::LossFactory::Instance().Create(name));
      }
      this->m_nb_parameters = nb_parameters;
      this->m_numberOfPixelsCounted = 0;
      for (int l = 0; l < this->m_layersWeight.size(); ++l)
      {
        this->m_losses[l]->SetNumberOfParameters(nb_parameters);
      }
    }

    void
    reset()
    {
      this->m_numberOfPixelsCounted = 0;
      for (std::unique_ptr<itk::Impact::Loss> & loss : m_losses)
      {
        loss->reset();
      }
    }

    double
    GetValue()
    {
      MeasureType value = MeasureType{};
      for (int l = 0; l < this->m_layersWeight.size(); ++l)
      {
        value +=
          this->m_layersWeight[l] * this->m_losses[l]->GetValue(static_cast<double>(this->m_numberOfPixelsCounted));
      }
      return value;
    }

    DerivativeType
    GetDerivative()
    {
      DerivativeType derivative = DerivativeType(this->m_nb_parameters);
      derivative.Fill(DerivativeValueType{});
      for (int l = 0; l < this->m_layersWeight.size(); ++l)
      {
        torch::Tensor d = this->m_layersWeight[l] *
                          this->m_losses[l]->GetDerivative(static_cast<double>(this->m_numberOfPixelsCounted));
        for (int i = 0; i < d.size(0); ++i)
        {
          derivative[i] += d[i].item<float>();
        }
      }
      return derivative;
    }

    LossPerThreadStruct &
    operator+=(const LossPerThreadStruct & other)
    {
      const auto * lossPerThreadStructOther = dynamic_cast<const LossPerThreadStruct *>(&other);
      if (lossPerThreadStructOther)
      {
        m_numberOfPixelsCounted += lossPerThreadStructOther->m_numberOfPixelsCounted;
        for (int i = 0; i < lossPerThreadStructOther->m_losses.size(); ++i)
        {
          *m_losses[i] += *lossPerThreadStructOther->m_losses[i];
        }
      }
      return *this;
    }
  };

  /** Cache-line-padded accumulator to avoid false sharing between work units. */
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT, LossPerThreadStruct, PaddedLossPerThreadStruct);

  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT, PaddedLossPerThreadStruct, AlignedLossPerThreadStruct);

  /** A point of the virtual domain that maps inside both images, resolved into each of them.
   * Online mode evaluates points in batches, so it gathers them before running any model. */
  struct Sample
  {
    VirtualIndexType     m_VirtualIndex;
    VirtualPointType     m_VirtualPoint;
    FixedImagePointType  m_FixedPoint;
    MovingImagePointType m_MovingPoint;
  };

  /** The plane a patch is cut on, as a matrix whose COLUMNS are the patch axes expressed in
   * the image's own frame. */
  using PatchPlaneType = Matrix<double,
                                ImageToImageMetricv4Type::FixedImageDimension,
                                ImageToImageMetricv4Type::FixedImageDimension>;

  /** Walk this work unit's share of the domain. Online mode walks it here so it can run the
   * models on a batch of points at a time; Static mode is left to the framework's own
   * per-point walk, which ends in ProcessPoint(). */
  void
  ThreadedExecution(const DomainType & domain, const ThreadIdType threadId) override;

  /** Gather the points of the domain that map inside both images. Overloaded on the two
   * domain types the v4 partitioners hand out: a region of the virtual image when the metric
   * samples densely, a range of indices into the sampled point set when it does not. */
  std::vector<Sample>
  CollectSamples(const ImageRegion<ImageToImageMetricv4Type::VirtualImageDimension> & region) const;
  std::vector<Sample>
  CollectSamples(const ThreadedIndexedContainerPartitioner::DomainType & indexRange) const;

  /** Resolve one virtual point into both image spaces; false if it falls outside either. */
  bool
  ResolveSample(const VirtualIndexType & virtualIndex, const VirtualPointType & virtualPoint, Sample & sample) const;

  /** The plane model \p modelIndex cuts its patch on at \p virtualIndex. Identity for a model
   * of the image's own dimension; a rotation for a lower-dimensional one. */
  PatchPlaneType
  PatchPlane(const VirtualIndexType & virtualIndex, size_t modelIndex, unsigned int modelDimension) const;

  /** Sample model \p modelIndex's intensity patch around one point in both images, and the
   * moving interpolator's spatial gradient at every patch voxel, into row \p row of the batch
   * tensors ([B, nVox] for the values, [B, nVox, MovingDim] for the gradient). */
  void
  SamplePatch(const Sample &  sample,
              size_t          modelIndex,
              int64_t         row,
              torch::Tensor & fixedPatches,
              torch::Tensor & movingPatches,
              torch::Tensor & movingGradients,
              bool            computeDerivative) const;

  /** Run the models on one batch of points, accumulating the layer losses and, when a
   * derivative is requested, their gradient. */
  void
  EvaluateJacobianBatch(typename std::vector<Sample>::const_iterator first,
                        typename std::vector<Sample>::const_iterator last,
                        const ThreadIdType                           threadId,
                        LossPerThreadStruct &                        loss) const;

  /** One accumulator per work unit, allocated in BeforeThreadedExecution. */
  mutable std::unique_ptr<AlignedLossPerThreadStruct[]> m_LossThreadStruct{ nullptr };

  /** Pre-cast associate metric, kept to avoid a dynamic_cast in tight loops. */
  TImpactMetric * m_ImpactAssociate{};

  /** Base seed for the per-point plane a lower-dimensional model is sampled on. Fixed on the
   * first evaluation and never redrawn: two evaluations of the same parameters have to sample
   * the same planes, or the metric is not a function of them and its derivative means nothing.
   * A Seed of zero still means "vary from run to run", it just varies once per metric rather
   * than once per evaluation. Shared by every work unit, so the value cannot depend on how the
   * domain was partitioned. */
  unsigned int m_PlaneSeed{ 0 };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImpactImageToImageMetricv4GetValueAndDerivativeThreader.hxx"
#endif

#endif

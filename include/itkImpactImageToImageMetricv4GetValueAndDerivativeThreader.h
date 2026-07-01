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

  /** One accumulator per work unit, allocated in BeforeThreadedExecution. */
  mutable std::unique_ptr<AlignedLossPerThreadStruct[]> m_LossThreadStruct{ nullptr };

  /** Pre-cast associate metric, kept to avoid a dynamic_cast in tight loops. */
  TImpactMetric * m_ImpactAssociate{};
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImpactImageToImageMetricv4GetValueAndDerivativeThreader.hxx"
#endif

#endif

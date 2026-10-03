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
#ifndef itkImpactLossNormalization_h
#define itkImpactLossNormalization_h

// Torch-free, so the metric and registration headers that hold one stay parsable by castxml.

#include <cmath>
#include <cstddef>
#include <vector>

namespace itk
{
namespace Impact
{

/** \class LossNormalization
 * \ingroup Impact
 * \brief Divides each compared layer's loss by its value when a level starts, so every layer starts at 1.
 *
 * Feature models answer on scales orders of magnitude apart (MIND in [0, 1] over 12 channels, a
 * segmentation decoder over 32 channels with activations of several units), so without it a layer's share of
 * the similarity is set by its range, and LayersWeight weighs incomparable quantities.
 *
 * The factor of a layer is latched the first time the layer's value is handed over after Reset(), and kept
 * until the next Reset(). Its owner resets it when a level starts: a metric at Initialize(), a registration
 * filter when its stage starts. It must not live in the loss objects, which a metric threader builds afresh
 * at every evaluation: a factor latched there divided every evaluation by its own value, which pinned the
 * reported value at 1 and turned the gradient into that of the logarithm. A layer whose starting value is 0
 * (already matched) or not finite keeps the factor 1.
 */
class LossNormalization
{
public:
  /** Forget every latched factor: the next value handed over for a layer sets its factor. */
  void
  Reset()
  {
    m_Factor.clear();
  }

  bool
  IsLatched(std::size_t layer) const
  {
    return layer < m_Factor.size() && m_Factor[layer] > 0.0;
  }

  /** The factor of `layer`, latched from `value` if the layer has none yet. */
  double
  Latch(std::size_t layer, double value)
  {
    if (layer >= m_Factor.size())
    {
      m_Factor.resize(layer + 1, 0.0);
    }
    if (m_Factor[layer] <= 0.0)
    {
      m_Factor[layer] = (value > 0.0 && std::isfinite(value)) ? 1.0 / value : 1.0;
    }
    return m_Factor[layer];
  }

  /** The latched factor of `layer`, 1 if none is latched. */
  double
  Factor(std::size_t layer) const
  {
    return this->IsLatched(layer) ? m_Factor[layer] : 1.0;
  }

private:
  std::vector<double> m_Factor;
};

} // namespace Impact
} // namespace itk

#endif // itkImpactLossNormalization_h

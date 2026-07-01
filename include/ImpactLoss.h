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

/**
 * \file ImpactLoss.h
 *
 * Differentiable loss functions for the IMPACT registration metric. Each loss derives
 * from `Loss` and registers itself with `LossFactory` for runtime instantiation by name.
 * Supports both static and Jacobian-based backpropagation modes.
 */

#ifndef _ImpactLoss_h
#define _ImpactLoss_h

#include <torch/torch.h>
#include <cmath>
#include <iostream>

namespace itk::Impact
{

/**
 * \class Loss
 * \ingroup Impact
 * \brief Abstract base class for losses operating on extracted feature maps.
 *
 * Accumulates the loss value and its derivative. Supports two modes:
 *   - Static: derivative updated manually from precomputed jacobians
 *   - Jacobian: direct backpropagation of gradients
 *
 * Subclasses must implement updateValue() and updateValueAndGetGradientModulator().
 */
class Loss
{
private:
  mutable double m_normalization = 0;

protected:
  double        m_value;
  torch::Tensor m_derivative;
  bool          m_initialized = false;
  int           m_nb_parameters;

public:
  Loss(bool isLossNormalized)
  {
    if (!isLossNormalized)
    {
      this->m_normalization = 1.0;
    }
  }

  void
  SetNumberOfParameters(int numberOfParameters)
  {
    this->m_nb_parameters = numberOfParameters;
  }
  void
  reset()
  {
    this->m_initialized = false;
  }

  virtual void
  initialize(torch::Tensor & output)
  {
    // Lazy init of internal buffers, sized from the output tensor and parameter count.
    if (!this->m_initialized)
    {
      this->m_value = 0;
      this->m_derivative = torch::zeros({ this->m_nb_parameters }, output.options());
      this->m_initialized = true;
    }
  }

  virtual void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) = 0;
  virtual void
  updateValueAndDerivativeInStaticMode(torch::Tensor & fixedOutput,
                                       torch::Tensor & movingOutput,
                                       torch::Tensor & jacobian,
                                       torch::Tensor & nonZeroJacobianIndices)
  {
    this->m_derivative.index_add_(
      0,
      nonZeroJacobianIndices.flatten(),
      (this->updateValueAndGetGradientModulator(fixedOutput, movingOutput).unsqueeze(-1) * jacobian).sum(1).flatten());
  }
  virtual torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) = 0;

  /** Non-mutating differentiable similarity used by autograd optimizers such as
   * ImpactFineRegistration (warp -> features -> forwardValue -> backward). Must stay
   * numerically consistent with updateValue(). The default throws so an un-ported loss
   * is rejected explicitly rather than silently mis-optimized. */
  virtual torch::Tensor
  forwardValue(const torch::Tensor & fixedOutput, const torch::Tensor & movingOutput) const
  {
    (void)fixedOutput;
    (void)movingOutput;
    throw std::runtime_error("forwardValue() is not implemented for this loss");
  }

  void
  updateDerivativeInJacobianMode(torch::Tensor & jacobian, torch::Tensor & nonZeroJacobianIndices)
  {
    this->m_derivative.index_add_(0, nonZeroJacobianIndices.flatten(), jacobian.flatten());
  }

  virtual double
  GetValue(double N) const
  {
    if (this->m_normalization == 0)
    {
      this->m_normalization = 1 / (this->m_value / N);
    }
    return this->m_normalization * this->m_value / N;
  }

  virtual torch::Tensor
  GetDerivative(double N) const
  {
    return this->m_normalization * this->m_derivative.to(torch::kCPU) / N;
  }

  virtual ~Loss() = default;

  virtual Loss &
  operator+=(const Loss & other)
  {
    if (!this->m_initialized && other.m_initialized)
    {
      this->m_value = other.m_value;
      this->m_derivative = other.m_derivative;
      this->m_initialized = true;
    }
    else if (other.m_initialized)
    {
      this->m_value += other.m_value;
      this->m_derivative += other.m_derivative;
    }
    return *this;
  }
};

/**
 * \class LossFactory
 * \ingroup Impact
 * \brief Singleton factory to register and create Loss instances by string name.
 *
 * Instantiates losses by name from configuration (e.g. "L1", "L2", "NCC").
 */
class LossFactory
{
public:
  using CreatorFunc = std::function<std::unique_ptr<Loss>()>;

  static LossFactory &
  Instance()
  {
    static LossFactory instance;
    return instance;
  }

  void
  RegisterLoss(const std::string & name, CreatorFunc creator)
  {
    factoryMap[name] = creator;
  }

  std::unique_ptr<Loss>
  Create(const std::string & name)
  {
    auto it = factoryMap.find(name);
    if (it != factoryMap.end())
    {
      return it->second();
    }
    throw std::runtime_error("Error: Unknown loss function " + name);
  }

private:
  std::unordered_map<std::string, CreatorFunc> factoryMap;
};

template <typename T>
class RegisterLoss
{
public:
  RegisterLoss(const std::string & name)
  {
    LossFactory::Instance().RegisterLoss(name, []() { return std::make_unique<T>(); });
  }
};

/**
 * \class L1
 * \ingroup Impact
 * \brief L1 loss over feature vectors: mean absolute difference.
 */
class L1 : public Loss
{
public:
  L1()
    : Loss(true)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    this->m_value += (fixedOutput - movingOutput).abs().mean(1).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor diffOutput = fixedOutput - movingOutput;
    this->m_value += diffOutput.abs().mean(1).sum().item<double>();
    return -torch::sign(diffOutput) / fixedOutput.size(1);
  }

  torch::Tensor
  forwardValue(const torch::Tensor & fixedOutput, const torch::Tensor & movingOutput) const override
  {
    // Sum (not mean) over channels so the similarity scales with the channel count and its
    // balance against the diffusion regularizer matches the reference (see L2::forwardValue).
    return (fixedOutput - movingOutput).abs().sum(1).mean();
  }
};

inline RegisterLoss<L1> L1_reg("L1");

/**
 * \class L2
 * \ingroup Impact
 * \brief Mean Squared Error (L2) over feature vectors.
 */
class L2 : public Loss
{
public:
  L2()
    : Loss(true)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    this->m_value += (fixedOutput - movingOutput).pow(2).mean(1).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor diffOutput = fixedOutput - movingOutput;
    this->m_value += diffOutput.pow(2).mean(1).sum().item<double>();
    return -2 * diffOutput / fixedOutput.size(1);
  }

  torch::Tensor
  forwardValue(const torch::Tensor & fixedOutput, const torch::Tensor & movingOutput) const override
  {
    // Sum (not mean) over channels, i.e. an SSD over feature vectors, so the similarity scales
    // with the channel count exactly like ConvexAdam's `(mov-fix)^2.mean(1)*C`. A per-channel
    // mean makes the similarity ~C times smaller, over-regularizing the Adam refinement.
    return (fixedOutput - movingOutput).pow(2).sum(1).mean();
  }
};


inline RegisterLoss<L2> MSE_reg("L2");

/**
 * \class Dice
 * \ingroup Impact
 * \brief Soft Dice loss over feature vectors.
 *
 * Operates on raw (non-thresholded) activations, does not mutate its inputs, and
 * is NaN-guarded for the degenerate empty/empty case (Dice forced to 1, gradient
 * forced to 0).
 */
class Dice : public Loss
{
public:
  Dice()
    : Loss(false)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor intersectionSum = (fixedOutput * movingOutput).sum(1);
    torch::Tensor unionSum = (fixedOutput + movingOutput).sum(1);

    // Degenerate case union == 0 (no structure): make the denominator safe to avoid
    // forming 0/0 (eagerly evaluated -> NaN) before the value is masked.
    torch::Tensor isEmpty = (unionSum == 0);
    torch::Tensor unionSumSafe = unionSum + isEmpty.to(unionSum.scalar_type());

    torch::Tensor dice = 2.0 * intersectionSum / unionSumSafe;
    dice.masked_fill_(isEmpty, 1.0); // empty/empty => Dice = 1
    this->m_value -= dice.sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor intersectionSum = (fixedOutput * movingOutput).sum(1);
    torch::Tensor unionSum = (fixedOutput + movingOutput).sum(1);

    torch::Tensor isEmpty = (unionSum == 0);
    torch::Tensor unionSumSafe = unionSum + isEmpty.to(unionSum.scalar_type());

    torch::Tensor dice = 2.0 * intersectionSum / unionSumSafe;
    dice.masked_fill_(isEmpty, 1.0);
    this->m_value -= dice.sum().item<double>();

    torch::Tensor grad = -2.0 * (fixedOutput * unionSumSafe.unsqueeze(-1) - intersectionSum.unsqueeze(-1)) /
                         (unionSumSafe * unionSumSafe).unsqueeze(-1);
    grad.masked_fill_(isEmpty.unsqueeze(-1), 0.0); // empty/empty => gradient = 0
    return grad;
  }

  torch::Tensor
  forwardValue(const torch::Tensor & fixedOutput, const torch::Tensor & movingOutput) const override
  {
    torch::Tensor intersectionSum = (fixedOutput * movingOutput).sum(1);
    torch::Tensor unionSum = (fixedOutput + movingOutput).sum(1);
    torch::Tensor isEmpty = (unionSum == 0);
    torch::Tensor unionSumSafe = unionSum + isEmpty.to(unionSum.scalar_type());
    torch::Tensor dice = 2.0 * intersectionSum / unionSumSafe;
    dice = dice.masked_fill(isEmpty, 1.0); // empty/empty => Dice = 1 (non-mutating)
    return -dice.mean();
  }
};


inline RegisterLoss<Dice> Dice_reg("Dice");

/**
 * \class L1Cosine
 * \ingroup Impact
 * \brief Combined cosine similarity and exponential L1 loss (penalizes both direction
 * and magnitude).
 */
class L1Cosine : public Loss
{
private:
  double lambda;

public:
  L1Cosine()
    : Loss(false)
  {
    this->lambda = 0.1;
  }

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor dot_product = (fixedOutput * movingOutput).sum(1);
    torch::Tensor norm_fixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor norm_moving = torch::norm(movingOutput, 2, 1);
    torch::Tensor cosine = dot_product / (norm_fixed * norm_moving);
    torch::Tensor expL1 = torch::exp(-this->lambda * (fixedOutput - movingOutput).abs());
    this->m_value -= (cosine.unsqueeze(-1) * expL1).mean(1).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor diffOutput = fixedOutput - movingOutput;
    torch::Tensor dot_product = (fixedOutput * movingOutput).sum(1);
    torch::Tensor norm_fixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor norm_moving = torch::norm(movingOutput, 2, 1);
    torch::Tensor v = (norm_fixed * norm_moving);

    torch::Tensor cosine = dot_product / (v);
    torch::Tensor expL1 = torch::exp(-this->lambda * (fixedOutput - movingOutput).abs());

    torch::Tensor dCossine = -(fixedOutput / v.unsqueeze(-1) -
                               (dot_product.unsqueeze(-1) * movingOutput) / (v * norm_moving.pow(2)).unsqueeze(-1));
    torch::Tensor dexpL1 = -torch::sign(diffOutput) * expL1 / fixedOutput.size(1);
    this->m_value -= (cosine.unsqueeze(-1) * expL1).mean(1).sum().item<double>();
    return dCossine * dexpL1 + cosine.unsqueeze(-1) * dexpL1;
  }

  torch::Tensor
  forwardValue(const torch::Tensor & fixedOutput, const torch::Tensor & movingOutput) const override
  {
    torch::Tensor dotProduct = (fixedOutput * movingOutput).sum(1);
    torch::Tensor normFixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor normMoving = torch::norm(movingOutput, 2, 1);
    torch::Tensor cosine = dotProduct / (normFixed * normMoving);
    torch::Tensor expL1 = torch::exp(-this->lambda * (fixedOutput - movingOutput).abs());
    return -(cosine.unsqueeze(-1) * expL1).mean(1).mean();
  }
};

inline RegisterLoss<L1Cosine> L1Cosine_reg(
  "L1Cosine");

/**
 * \class Cosine
 * \ingroup Impact
 * \brief Cosine similarity loss (negative mean cosine between vectors).
 */
class Cosine : public Loss
{
public:
  Cosine()
    : Loss(false)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor dot_product = (fixedOutput * movingOutput).sum(1);
    torch::Tensor norm_fixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor norm_moving = torch::norm(movingOutput, 2, 1);
    this->m_value -= (dot_product / (norm_fixed * norm_moving)).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor dot_product = (fixedOutput * movingOutput).sum(1);
    torch::Tensor norm_fixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor norm_moving = torch::norm(movingOutput, 2, 1);
    torch::Tensor v = (norm_fixed * norm_moving);
    this->m_value -= (dot_product / v).sum().item<double>();
    // d(-cosine)/d(movingOutput_c) = -( f_c/v - (f.m) m_c / (v |m|^2) ). The cross term
    // uses the full dot product f.m, not the per-channel f_c m_c.
    return -(fixedOutput / v.unsqueeze(-1) -
             (dot_product.unsqueeze(-1) * movingOutput) / (v * norm_moving.pow(2)).unsqueeze(-1));
  }

  torch::Tensor
  forwardValue(const torch::Tensor & fixedOutput, const torch::Tensor & movingOutput) const override
  {
    torch::Tensor dotProduct = (fixedOutput * movingOutput).sum(1);
    torch::Tensor normFixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor normMoving = torch::norm(movingOutput, 2, 1);
    return -(dotProduct / (normFixed * normMoving)).mean();
  }
};

inline RegisterLoss<Cosine> Cosine_reg("Cosine");

/**
 * \class DotProduct
 * \ingroup Impact
 * \brief Negative dot product loss (simple similarity).
 */
class DotProduct : public Loss
{
public:
  DotProduct()
    : Loss(false)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    this->m_value -= (fixedOutput * movingOutput).sum(1).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    this->m_value -= (fixedOutput * movingOutput).sum(1).sum().item<double>();
    return -fixedOutput;
  }

  torch::Tensor
  forwardValue(const torch::Tensor & fixedOutput, const torch::Tensor & movingOutput) const override
  {
    return -(fixedOutput * movingOutput).sum(1).mean();
  }
};


inline RegisterLoss<DotProduct> DotProduct_reg(
  "DotProduct");

/**
 * \class NCC
 * \ingroup Impact
 * \brief Normalized Cross Correlation loss over feature vectors.
 *
 * Computes NCC between fixed and moving features across batches; in static mode the
 * derivative is accumulated with full Jacobian tracking.
 */
class NCC : public Loss
{
private:
  torch::Tensor m_sff, m_smm, m_sfm, m_sf, m_sm;
  torch::Tensor m_sfdm, m_smdm, m_sdm;

public:
  NCC()
    : Loss(false)
  {}

  void
  initialize(torch::Tensor & output) override
  {
    if (!this->m_initialized)
    {
      this->m_sff = torch::zeros({ output.size(1) }, output.options());
      this->m_smm = torch::zeros({ output.size(1) }, output.options());
      this->m_sfm = torch::zeros({ output.size(1) }, output.options());
      this->m_sf = torch::zeros({ output.size(1) }, output.options());
      this->m_sm = torch::zeros({ output.size(1) }, output.options());
      this->m_initialized = true;
    }
  }

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    this->m_sff += (fixedOutput * fixedOutput).sum(0);
    this->m_smm += (movingOutput * movingOutput).sum(0);
    this->m_sfm += (fixedOutput * movingOutput).sum(0);
    this->m_sf += fixedOutput.sum(0);
    this->m_sm += movingOutput.sum(0);
  }

  void
  updateValueAndDerivativeInStaticMode(torch::Tensor & fixedOutput,
                                       torch::Tensor & movingOutput,
                                       torch::Tensor & jacobian,
                                       torch::Tensor & nonZeroJacobianIndices) override
  {
    // Accumulate first-order statistics and Jacobian-weighted sums:
    // sfdm = sum(fixed * dM), smdm = sum(moving * dM), sdm = sum(dM).
    if (!this->m_initialized)
    {
      this->m_sfdm = torch::zeros({ fixedOutput.size(1), this->m_nb_parameters }, fixedOutput.options());
      this->m_smdm = torch::zeros({ fixedOutput.size(1), this->m_nb_parameters }, fixedOutput.options());
      this->m_sdm = torch::zeros({ fixedOutput.size(1), this->m_nb_parameters }, fixedOutput.options());
    }
    this->updateValue(fixedOutput, movingOutput);
    this->m_sfdm.index_add_(
      1, nonZeroJacobianIndices.flatten(), (fixedOutput.unsqueeze(-1) * jacobian).permute({ 1, 0, 2 }).flatten(1, 2));
    this->m_smdm.index_add_(
      1, nonZeroJacobianIndices.flatten(), (movingOutput.unsqueeze(-1) * jacobian).permute({ 1, 0, 2 }).flatten(1, 2));
    this->m_sdm.index_add_(1, nonZeroJacobianIndices.flatten(), (jacobian).permute({ 1, 0, 2 }).flatten(1, 2));
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    if (!this->m_initialized)
    {
      this->m_derivative = torch::zeros({ this->m_nb_parameters }, fixedOutput.options());
    }
    this->initialize(fixedOutput);

    const double  N = fixedOutput.size(0);
    torch::Tensor sff = (fixedOutput * fixedOutput).sum(0);
    torch::Tensor smm = (movingOutput * movingOutput).sum(0);
    torch::Tensor sfm = (fixedOutput * movingOutput).sum(0);
    torch::Tensor sf = fixedOutput.sum(0);
    torch::Tensor sm = movingOutput.sum(0);

    this->m_sff += sff;
    this->m_smm += smm;
    this->m_sfm += sfm;
    this->m_sf += sf;
    this->m_sm += sm;

    torch::Tensor u = sfm - (sf * sm / N);
    torch::Tensor v = torch::sqrt(sff - sf * sf / N) * torch::sqrt(smm - sm * sm / N);

    torch::Tensor u_p = fixedOutput - sf.unsqueeze(0) / N;
    return -((u_p - u.unsqueeze(0) * (movingOutput - sm.unsqueeze(0) / N) / (smm - sm * sm / N).unsqueeze(0)) /
             v.unsqueeze(0)) /
           fixedOutput.size(1);
  }

  torch::Tensor
  forwardValue(const torch::Tensor & fixedOutput, const torch::Tensor & movingOutput) const override
  {
    // Global NCC over the whole batch (N samples), mean over channels; matches GetValue's
    // closed form (the Adam optimizer passes all voxels as a single batch).
    const double  N = fixedOutput.size(0);
    torch::Tensor sff = (fixedOutput * fixedOutput).sum(0);
    torch::Tensor smm = (movingOutput * movingOutput).sum(0);
    torch::Tensor sfm = (fixedOutput * movingOutput).sum(0);
    torch::Tensor sf = fixedOutput.sum(0);
    torch::Tensor sm = movingOutput.sum(0);
    torch::Tensor u = sfm - sf * sm / N;
    torch::Tensor v = torch::sqrt(sff - sf * sf / N) * torch::sqrt(smm - sm * sm / N);
    return -(u / v).mean();
  }

  double
  GetValue(double N) const override
  {
    // NCC loss from accumulated statistics: mean over channels of -NCC.
    if (N <= 0)
      return 0.0;
    torch::Tensor u = this->m_sfm - (this->m_sf * this->m_sm / N);
    torch::Tensor v =
      torch::sqrt(this->m_sff - this->m_sf * this->m_sf / N) * torch::sqrt(this->m_smm - this->m_sm * this->m_sm / N);
    return -(u / v).mean().item<double>();
  }

  torch::Tensor
  GetDerivative(double N) const override
  {
    if (this->m_derivative.defined())
    {
      return this->m_derivative.to(torch::kCPU);
    }

    torch::Tensor u = this->m_sfm - (this->m_sf * this->m_sm / N);
    torch::Tensor v =
      torch::sqrt(this->m_sff - this->m_sf * this->m_sf / N) * torch::sqrt(this->m_smm - this->m_sm * this->m_sm / N);
    torch::Tensor u_p = this->m_sfdm - this->m_sf.unsqueeze(-1) * this->m_sdm / N;
    return -((u_p - u.unsqueeze(-1) * (this->m_smdm - this->m_sm.unsqueeze(-1) * this->m_sdm / N) /
                      (this->m_smm - this->m_sm * this->m_sm / N).unsqueeze(-1)) /
             v.unsqueeze(-1))
              .mean(0)
              .to(torch::kCPU);
  }

  NCC &
  operator+=(const Loss & other) override
  {
    const auto * nccOther = dynamic_cast<const NCC *>(&other);
    // A work unit that processed no point never ran initialize(): its accumulators are
    // undefined and contribute nothing to the reduction (mirrors Loss::operator+=).
    if (nccOther == nullptr || !nccOther->m_initialized)
    {
      return *this;
    }
    if (!this->m_initialized)
    {
      // This side is empty: adopt the other work unit's statistics.
      this->m_sff = nccOther->m_sff;
      this->m_smm = nccOther->m_smm;
      this->m_sfm = nccOther->m_sfm;
      this->m_sf = nccOther->m_sf;
      this->m_sm = nccOther->m_sm;
      this->m_sfdm = nccOther->m_sfdm;
      this->m_smdm = nccOther->m_smdm;
      this->m_sdm = nccOther->m_sdm;
      this->m_derivative = nccOther->m_derivative;
      this->m_initialized = true;
      return *this;
    }
    this->m_sff += nccOther->m_sff;
    this->m_smm += nccOther->m_smm;
    this->m_sfm += nccOther->m_sfm;
    this->m_sf += nccOther->m_sf;
    this->m_sm += nccOther->m_sm;
    if (this->m_sfdm.defined() && nccOther->m_sfdm.defined())
    {
      this->m_sfdm += nccOther->m_sfdm;
      this->m_smdm += nccOther->m_smdm;
      this->m_sdm += nccOther->m_sdm;
    }
    if (this->m_derivative.defined() && nccOther->m_derivative.defined())
    {
      this->m_derivative += nccOther->m_derivative;
    }
    return *this;
  }
};

inline RegisterLoss<NCC> NCC_reg("NCC");

} // namespace itk::Impact

#endif // _ImpactLoss_h

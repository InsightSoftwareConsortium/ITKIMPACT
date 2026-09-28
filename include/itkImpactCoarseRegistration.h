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
#ifndef itkImpactCoarseRegistration_h
#define itkImpactCoarseRegistration_h

// Torch-free public header (castxml-safe; all LibTorch lives in the .hxx, included only when
// ITK_MANUAL_INSTANTIATION is undefined), like itkImpactFineRegistration.h.

#include <itkImageSource.h>
#include <itkImage.h>
#include <itkVector.h>
#include <itkDisplacementFieldTransform.h>
#include <itkImpactModelConfiguration.h>
#include <itkImpactLossNormalization.h>

#include <string>
#include <vector>

namespace itk
{

/** \class ImpactCoarseRegistration
 *
 * \brief Coarse, discrete-convex displacement-field initializer (stage 1 of the pipeline).
 *
 * Stage 1 of a two-stage coarse->fine pipeline: it produces a robust low-resolution field
 * that warm-starts the sub-voxel itk::ImpactFineRegistration filter:
 *
 *   fixed + moving -> ImpactCoarseRegistration -> initial field
 *                  -> ImpactFineRegistration (SetInitialDisplacementField) -> refined field
 *
 * Follows the ConvexAdam strategy (Siebert/Hansen/Heinrich): build a discrete cost
 * volume over a dense displacement search window on a coarse grid, then run a coupled-convex
 * global regularization (an increasing-coupling argmin/smoothing schedule) to obtain a smooth
 * coarse field, finally upsampled to full resolution. All heavy computation uses LibTorch. The
 * cost is on raw intensities by default, or on deep features of any configured IMPACT
 * TorchScript model (not tied to MIND specifically).
 *
 * The output is a geometry-correct itk displacement field on the fixed grid (physical
 * millimetres, ITK x,y,z, fixed->moving), sharing the ImpactFineRegistration convention
 * (itkImpactTorchRegistrationHelpers.h), so it feeds directly to that filter as a warm start.
 *
 * \note Supports 2D and 3D images.
 *
 * \ingroup Impact
 */
template <typename TFixedImage, typename TMovingImage = TFixedImage>
class ITK_TEMPLATE_EXPORT ImpactCoarseRegistration
  : public ImageSource<Image<Vector<float, TFixedImage::ImageDimension>, TFixedImage::ImageDimension>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImpactCoarseRegistration);

  static constexpr unsigned int ImageDimension = TFixedImage::ImageDimension;

  // Float displacement field (ITK-idiomatic precision; its ImageSource base is wrapped, so the
  // filter is exposable to Python), matching ImpactFineRegistration.
  using VectorType = Vector<float, ImageDimension>;
  using DisplacementFieldType = Image<VectorType, ImageDimension>;

  using Self = ImpactCoarseRegistration;
  using Superclass = ImageSource<DisplacementFieldType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(ImpactCoarseRegistration);

  using FixedImageType = TFixedImage;
  using MovingImageType = TMovingImage;
  using DisplacementFieldTransformType = DisplacementFieldTransform<float, ImageDimension>;

  /** Set/Get the fixed image (defines the output field domain). */
  void
  SetFixedImage(const FixedImageType * image);
  itkGetConstObjectMacro(FixedImage, FixedImageType);

  /** Set/Get the moving image. */
  void
  SetMovingImage(const MovingImageType * image);
  itkGetConstObjectMacro(MovingImage, MovingImageType);

  /** Optional masks (a voxel is in where the mask is not 0), each on its image's grid; absent, the whole image
   * counts, at no cost. A cell's features are averaged over its masked voxels, and its cost at a coarse voxel for a
   * candidate is averaged over the cell window weighed by the fixed cell's share of the fixed mask times the share of
   * the moving mask in the moving cell the candidate shifts it to (moved like the features). A candidate that leaves
   * the window no weight costs as much as the worst candidate with some; a coarse voxel no candidate gives any weight
   * has no data cost and follows its neighbours through the coupling. */
  using MaskImageType = Image<unsigned char, ImageDimension>;
  itkSetConstObjectMacro(FixedMask, MaskImageType);
  itkGetConstObjectMacro(FixedMask, MaskImageType);
  itkSetConstObjectMacro(MovingMask, MaskImageType);
  itkGetConstObjectMacro(MovingMask, MaskImageType);

  /** \name Optional IMPACT feature configuration. With no model the cost volume is built on
   * raw intensities; with model(s) it is built on their (concatenated) feature channels. */
  /** @{ */
  itkSetMacro(FixedModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkGetConstReferenceMacro(FixedModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkSetMacro(MovingModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkGetConstReferenceMacro(MovingModelsConfiguration, std::vector<ImpactModelConfiguration>);
  void
  SetModelsConfiguration(const std::vector<ImpactModelConfiguration> & configuration)
  {
    this->SetFixedModelsConfiguration(configuration);
    this->SetMovingModelsConfiguration(configuration);
  }
  void
  AddModelConfiguration(const ImpactModelConfiguration & configuration)
  {
    m_FixedModelsConfiguration.push_back(configuration);
    m_MovingModelsConfiguration.push_back(configuration);
    this->Modified();
  }
  /** Per kept layer, the number of its channels the cost compares, drawn at random once from the seeded
   * generator (0 or a missing entry = all), as ImpactFineRegistration draws them at every iteration. */
  itkSetMacro(SubsetFeatures, std::vector<unsigned int>);
  itkGetConstReferenceMacro(SubsetFeatures, std::vector<unsigned int>);
  /** Per kept layer, the number of principal components its channels are reduced to, fitted on the fixed
   * features (0 or a missing entry = all), as ImpactFineRegistration::SetPCA. */
  itkSetMacro(PCA, std::vector<unsigned int>);
  itkGetConstReferenceMacro(PCA, std::vector<unsigned int>);
  /** Per kept layer, the distance its cost compares with, as ImpactFineRegistration::SetDistance: L1,
   * L2 (the default, ConvexAdam's squared differences), Cosine, L1Cosine, Dice, NCC or LNCC, a missing
   * entry repeating the last one. The cost at a coarse voxel is the distance over the cell window around
   * it -- the two 3^Dim box passes ConvexAdam smooths its cost with, a triangular 5^Dim window of coarse
   * voxels: a point-wise distance is averaged over it, and NCC and LNCC both correlate each channel over
   * it (neither the whole image nor LNCCKernel: one window per coarse voxel and candidate). Unused on
   * intensities, which keep the squared differences. */
  itkSetMacro(Distance, std::vector<std::string>);
  itkGetConstReferenceMacro(Distance, std::vector<std::string>);
  /** Weight of each kept feature layer in the cost volume, one entry per layer kept across the
   * models, as ImpactFineRegistration::SetLayersWeight (a missing entry weighs 1). The cost sums
   * the layers' distances, so without a weight a layer with larger features or more channels
   * outweighs the others, and the coupling schedule, whose coefficients are absolute, regularises
   * it too little. Non-negative; unused on intensities. */
  itkSetMacro(LayersWeight, std::vector<float>);
  itkGetConstReferenceMacro(LayersWeight, std::vector<float>);
  /** Divide each layer's cost by its value at zero displacement, so every layer starts at 1 and
   * LayersWeight weighs comparable quantities (see Impact::LossNormalization). Default on; unused
   * while BalanceLosses is on. */
  itkSetMacro(NormalizeLosses, bool);
  itkGetConstMacro(NormalizeLosses, bool);
  itkBooleanMacro(NormalizeLosses);
  /** Balance the layers on how much each moves the argmin: layer l's spread S_l, the mean over the
   * coarse cells of max_d C_l(x, d) - min_d C_l(x, d) over the candidate box, is measured on the first
   * cost volume (fixed onto moving), and the layer is weighed by (sum_k S_k / L) / S_l times its
   * LayersWeight, the sum and L running over the layers LayersWeight keeps with a spread above 0 (a
   * layer of spread 0 moves nothing and adds nothing). Every layer then spreads alike, and their total
   * spread is the raw one, the scale the coupling schedule's absolute coefficients are calibrated on --
   * where NormalizeLosses divides a layer with a small cost at zero displacement into a spread that
   * swamps the coupling. Takes the place of NormalizeLosses. Default off. */
  itkSetMacro(BalanceLosses, bool);
  itkGetConstMacro(BalanceLosses, bool);
  itkBooleanMacro(BalanceLosses);
  /** Per kept layer, the spread S_l the last run measured (see BalanceLosses), before any weight; empty
   * while BalanceLosses is off. */
  const std::vector<double> &
  GetLayerSpreads() const
  {
    return m_LayerSpreads;
  }
  /** @} */

  /** Set/Get the torch device ("cpu", "cuda", "cuda:0", ...). */
  itkSetMacro(Device, std::string);
  itkGetConstReferenceMacro(Device, std::string);

  itkSetMacro(Seed, unsigned int);
  itkGetConstMacro(Seed, unsigned int);

  /** \name Coarse search parameters.
   * Counted in voxels of the fixed image's finest axis, s_min = min_a s_a, and derived per axis, so that
   * the cells and the capture range are (nearly) isotropic in millimetres; on an isotropic image every
   * axis takes the numbers as they are. */
  /** @{ */
  /** Cell size of the coarse grid the cost volume runs on, in voxels of the finest axis: along axis a a
   * cell spans gs_a = max(1, round(GridSpacing * s_min / s_a)) voxels (avg-pool kernel and stride), a
   * cell of about GridSpacing * s_min mm. The cost smoothing, the NCC/LNCC window and the displacement
   * smoothing are counted in cells. Default 4. */
  itkSetMacro(GridSpacing, unsigned int);
  itkGetConstMacro(GridSpacing, unsigned int);
  /** Displacement search half-width, in cells of the finest axis: the capture range is R =
   * DisplacementHalfWidth * GridSpacing * s_min mm each way, which axis a covers with hw_a = ceil(R /
   * (gs_a * s_a)) cells, so the candidates are the box prod_a (2 hw_a + 1). The coupling penalty
   * weighs a move of one cell along axis a by (gs_a * s_a / (GridSpacing * s_min))^2, so that a
   * distance in millimetres costs the same along every axis. Default 3. */
  itkSetMacro(DisplacementHalfWidth, unsigned int);
  itkGetConstMacro(DisplacementHalfWidth, unsigned int);
  /** Also solve the backward (moving->fixed) coarse problem and symmetrize the two fields
   * toward mutual inverses (ConvexAdam-style), for a more diffeomorphic coarse initialization.
   * Default on, matching the original ConvexAdam (convex_adam_pt ic=True): with it off the coarse
   * field is under-regularised and folds. */
  itkSetMacro(InverseConsistency, bool);
  itkGetConstMacro(InverseConsistency, bool);
  itkBooleanMacro(InverseConsistency);
  /** @} */

  /** The output displacement field (== GetOutput()), fixed grid, millimetres, x,y,z. */
  DisplacementFieldType *
  GetDisplacementField();
  /** The output field wrapped in a ready-to-use itk::DisplacementFieldTransform. */
  DisplacementFieldTransformType *
  GetDisplacementFieldTransform();

protected:
  ImpactCoarseRegistration();
  ~ImpactCoarseRegistration() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  void
  GenerateOutputInformation() override;

  void
  GenerateData() override;

private:
  typename FixedImageType::ConstPointer  m_FixedImage{ nullptr };
  typename MovingImageType::ConstPointer m_MovingImage{ nullptr };
  typename MaskImageType::ConstPointer   m_FixedMask{ nullptr };
  typename MaskImageType::ConstPointer   m_MovingMask{ nullptr };

  std::vector<ImpactModelConfiguration> m_FixedModelsConfiguration;
  std::vector<ImpactModelConfiguration> m_MovingModelsConfiguration;
  std::vector<unsigned int>             m_SubsetFeatures;
  std::vector<unsigned int>             m_PCA;
  std::vector<std::string>              m_Distance;
  std::vector<float>                    m_LayersWeight;
  bool                                  m_NormalizeLosses{ true };
  bool                                  m_BalanceLosses{ false };
  std::vector<double>                   m_LayerSpreads;

  std::string  m_Device{ "cpu" };
  unsigned int m_Seed{ 0 };
  unsigned int m_GridSpacing{ 4 };
  unsigned int m_DisplacementHalfWidth{ 3 };
  bool         m_InverseConsistency{ true };

  typename DisplacementFieldTransformType::Pointer m_DisplacementFieldTransform{ nullptr };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImpactCoarseRegistration.hxx"
#endif

#endif // itkImpactCoarseRegistration_h

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
#include <itkModelConfiguration.h>

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
 * Follows the ConvexAdam strategy (Siebert/Hansen/Heinrich): build a discrete SSD cost
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

  /** \name Optional IMPACT feature configuration. With no model the cost volume is built on
   * raw intensities; with model(s) it is built on their (concatenated) feature channels. */
  /** @{ */
  itkSetMacro(FixedModelsConfiguration, std::vector<ModelConfiguration>);
  itkGetConstReferenceMacro(FixedModelsConfiguration, std::vector<ModelConfiguration>);
  itkSetMacro(MovingModelsConfiguration, std::vector<ModelConfiguration>);
  itkGetConstReferenceMacro(MovingModelsConfiguration, std::vector<ModelConfiguration>);
  void
  SetModelsConfiguration(const std::vector<ModelConfiguration> & configuration)
  {
    this->SetFixedModelsConfiguration(configuration);
    this->SetMovingModelsConfiguration(configuration);
  }
  void
  AddModelConfiguration(const ModelConfiguration & configuration)
  {
    m_FixedModelsConfiguration.push_back(configuration);
    m_MovingModelsConfiguration.push_back(configuration);
    this->Modified();
  }
  itkSetMacro(SubsetFeatures, std::vector<unsigned int>);
  itkGetConstReferenceMacro(SubsetFeatures, std::vector<unsigned int>);
  /** @} */

  /** Set/Get the torch device ("cpu", "cuda", "cuda:0", ...). */
  itkSetMacro(Device, std::string);
  itkGetConstReferenceMacro(Device, std::string);

  itkSetMacro(Seed, unsigned int);
  itkGetConstMacro(Seed, unsigned int);

  /** \name Coarse search parameters. */
  /** @{ */
  /** Coarse-grid downsample factor (avg-pool stride); the cost volume runs on the
   * image-size / GridSpacing grid. Default 4. */
  itkSetMacro(GridSpacing, unsigned int);
  itkGetConstMacro(GridSpacing, unsigned int);
  /** Displacement search half-width, in coarse-grid voxels: the candidate set is the dense cube
   * [-hw, hw]^Dim, so the captured range is +/- DisplacementHalfWidth * GridSpacing full-resolution
   * voxels. Default 3. */
  itkSetMacro(DisplacementHalfWidth, unsigned int);
  itkGetConstMacro(DisplacementHalfWidth, unsigned int);
  /** Also solve the backward (moving->fixed) coarse problem and symmetrize the two fields
   * toward mutual inverses (ConvexAdam-style), for a more diffeomorphic coarse initialization.
   * Default off. */
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

  std::vector<ModelConfiguration> m_FixedModelsConfiguration;
  std::vector<ModelConfiguration> m_MovingModelsConfiguration;
  std::vector<unsigned int>       m_SubsetFeatures;

  std::string  m_Device{ "cpu" };
  unsigned int m_Seed{ 0 };
  unsigned int m_GridSpacing{ 4 };
  unsigned int m_DisplacementHalfWidth{ 3 };
  bool         m_InverseConsistency{ false };

  typename DisplacementFieldTransformType::Pointer m_DisplacementFieldTransform{ nullptr };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImpactCoarseRegistration.hxx"
#endif

#endif // itkImpactCoarseRegistration_h

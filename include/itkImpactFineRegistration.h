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
#ifndef itkImpactFineRegistration_h
#define itkImpactFineRegistration_h

// Torch-free public header so castxml can parse it and it is exposable to Python (WrapITK),
// like itkModelConfiguration.h and itkImpactImageToImageMetricv4.h. All torch state (the
// displacement-field leaf tensor, the grid_sample warp, the IMPACT feature loss and the
// torch::optim::Adam loop) lives in the .hxx, included only when ITK_MANUAL_INSTANTIATION
// is undefined.

#include <itkImageSource.h>
#include <itkImage.h>
#include <itkVector.h>
#include <itkDisplacementFieldTransform.h>
#include <itkModelConfiguration.h>

#include <string>
#include <vector>

namespace itk
{

/** \class ImpactFineRegistration
 *
 * \brief Fast dense displacement-field registration refined with a Torch-backed Adam
 * optimizer (stage 2), in the spirit of the ConvexAdam instance-optimization step.
 *
 * Self-contained ITK filter: takes a fixed and a moving image (plus an optional initial
 * displacement field as a warm start) and produces a dense displacement field mapping the
 * fixed domain onto the moving image (fixed \f$\to\f$ moving, physical millimetres, ITK
 * \f$x,y,z\f$ component order). The heavy computation runs entirely with LibTorch: the field
 * is a GPU-resident leaf tensor, the moving image is warped with \c grid_sample, a similarity
 * loss and a diffusion (smoothness) regularizer are evaluated, and \c torch::optim::Adam steps
 * the field in place. No per-voxel C++ loop and no per-iteration CPU/GPU round trip.
 *
 * The similarity term is pluggable: with no model configuration it compares raw intensities
 * (MSE); with IMPACT model configurations it compares deep features from the pretrained
 * TorchScript models (reusing the \c ModelConfiguration / loss vocabulary of
 * ImpactImageToImageMetricv4).
 *
 * ITK convention handling (axis reversal between ITK \f$x,y,z\f$ and the torch \f$z,y,x\f$
 * tensor layout, voxel\f$\to\f$millimetre scaling, rotation by the fixed-image direction
 * cosines) is internal; callers only see ITK images and a standard
 * itk::DisplacementFieldTransform.
 *
 * \ingroup Impact
 */
template <typename TFixedImage, typename TMovingImage = TFixedImage>
class ITK_TEMPLATE_EXPORT ImpactFineRegistration
  : public ImageSource<Image<Vector<float, TFixedImage::ImageDimension>, TFixedImage::ImageDimension>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImpactFineRegistration);

  static constexpr unsigned int ImageDimension = TFixedImage::ImageDimension;

  // Float displacement field: sub-voxel precision is unaffected, it is the ITK-idiomatic
  // displacement-field precision, and (unlike Vector<double>) its ImageSource base is wrapped,
  // so the filter is exposable to Python.
  using VectorType = Vector<float, ImageDimension>;
  using DisplacementFieldType = Image<VectorType, ImageDimension>;

  using Self = ImpactFineRegistration;
  using Superclass = ImageSource<DisplacementFieldType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(ImpactFineRegistration);

  using FixedImageType = TFixedImage;
  using MovingImageType = TMovingImage;
  using DisplacementFieldTransformType = DisplacementFieldTransform<float, ImageDimension>;
  using WarpedImageType = Image<float, ImageDimension>;

  /** Set/Get the fixed image. Its grid (size, spacing, origin, direction) defines the
   * domain of the output displacement field. */
  void
  SetFixedImage(const FixedImageType * image);
  itkGetConstObjectMacro(FixedImage, FixedImageType);

  /** Set/Get the moving image to be registered onto the fixed image. */
  void
  SetMovingImage(const MovingImageType * image);
  itkGetConstObjectMacro(MovingImage, MovingImageType);

  /** Set/Get an optional initial displacement field used as a warm start (e.g. the output
   * of an affine or a ConvexAdam-style discrete stage). Must be defined on the fixed grid. */
  itkSetObjectMacro(InitialDisplacementField, DisplacementFieldType);
  itkGetConstObjectMacro(InitialDisplacementField, DisplacementFieldType);

  /** \name IMPACT feature-loss configuration (mirrors ImpactImageToImageMetricv4).
   * Leaving the model configuration empty selects the intensity (MSE) similarity. */
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

  itkSetMacro(Distance, std::vector<std::string>);
  itkGetConstReferenceMacro(Distance, std::vector<std::string>);
  itkSetMacro(LayersWeight, std::vector<float>);
  itkGetConstReferenceMacro(LayersWeight, std::vector<float>);
  itkSetMacro(SubsetFeatures, std::vector<unsigned int>);
  itkGetConstReferenceMacro(SubsetFeatures, std::vector<unsigned int>);
  itkSetMacro(PCA, std::vector<unsigned int>);
  itkGetConstReferenceMacro(PCA, std::vector<unsigned int>);
  /** @} */

  /** Set/Get the torch device ("cpu", "cuda", "cuda:0", ...). */
  itkSetMacro(Device, std::string);
  itkGetConstReferenceMacro(Device, std::string);

  /** Set/Get the manual random seed used for any stochastic feature subsetting. */
  itkSetMacro(Seed, unsigned int);
  itkGetConstMacro(Seed, unsigned int);

  /** \name Adam / displacement-field optimization parameters. */
  /** @{ */
  /** Number of Adam iterations (default 80, as in ConvexAdam). */
  itkSetMacro(NumberOfIterations, unsigned int);
  itkGetConstMacro(NumberOfIterations, unsigned int);
  /** Adam learning rate (default 1.0). */
  itkSetMacro(LearningRate, double);
  itkGetConstMacro(LearningRate, double);
  itkSetMacro(Beta1, double);
  itkGetConstMacro(Beta1, double);
  itkSetMacro(Beta2, double);
  itkGetConstMacro(Beta2, double);
  itkSetMacro(Epsilon, double);
  itkGetConstMacro(Epsilon, double);
  /** Weight of the diffusion (squared spatial-gradient) regularizer (default 1.25). */
  itkSetMacro(RegularizationWeight, double);
  itkGetConstMacro(RegularizationWeight, double);
  /** Low-resolution control-grid shrink factor: the field is optimized at
   * image-size / GridShrinkFactor and upsampled to full resolution each iteration
   * (ConvexAdam-style) -- faster on large volumes and adds regularity. Default 1 (full resolution). */
  itkSetMacro(GridShrinkFactor, unsigned int);
  itkGetConstMacro(GridShrinkFactor, unsigned int);
  /** Number of 3x3x3 average-pool smoothing passes applied to the control grid each iteration
   * (B-spline-like control-point smoothing). Default 0. */
  itkSetMacro(ControlGridSmoothingIterations, unsigned int);
  itkGetConstMacro(ControlGridSmoothingIterations, unsigned int);
  /** In feature mode, re-extract the moving feature maps from the currently-warped moving image
   * every this many Adam iterations. Helps large deformations, where warping a precomputed feature
   * map diverges from the features of the warped image. <= 0 extracts once (disabled). Default -1. */
  itkSetMacro(FeatureMapUpdateInterval, int);
  itkGetConstMacro(FeatureMapUpdateInterval, int);
  /** @} */

  /** \name Outputs. */
  /** @{ */
  /** The output displacement field (== GetOutput()), on the fixed grid, in millimetres. */
  DisplacementFieldType *
  GetDisplacementField();

  /** The output field wrapped in a ready-to-use itk::DisplacementFieldTransform. */
  DisplacementFieldTransformType *
  GetDisplacementFieldTransform();

  /** The moving image warped onto the fixed grid by the final field (for inspection). */
  WarpedImageType *
  GetWarpedMovingImage();

  /** The total (similarity + regularization) loss recorded at each Adam iteration. */
  const std::vector<double> &
  GetMetricValuesPerIteration() const
  {
    return m_MetricValuesPerIteration;
  }
  /** @} */

protected:
  ImpactFineRegistration();
  ~ImpactFineRegistration() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Copy the fixed-image geometry onto the output displacement field. */
  void
  GenerateOutputInformation() override;

  /** Run the whole Torch-backed Adam refinement (defined in the .hxx). */
  void
  GenerateData() override;

private:
  typename FixedImageType::ConstPointer        m_FixedImage{ nullptr };
  typename MovingImageType::ConstPointer       m_MovingImage{ nullptr };
  typename DisplacementFieldType::Pointer      m_InitialDisplacementField{ nullptr };

  std::vector<ModelConfiguration> m_FixedModelsConfiguration;
  std::vector<ModelConfiguration> m_MovingModelsConfiguration;
  std::vector<std::string>        m_Distance;
  std::vector<float>              m_LayersWeight;
  std::vector<unsigned int>       m_SubsetFeatures;
  std::vector<unsigned int>       m_PCA;

  std::string  m_Device{ "cpu" };
  unsigned int m_Seed{ 0 };

  unsigned int m_NumberOfIterations{ 80 };
  double       m_LearningRate{ 1.0 };
  double       m_Beta1{ 0.9 };
  double       m_Beta2{ 0.999 };
  double       m_Epsilon{ 1e-8 };
  double       m_RegularizationWeight{ 1.25 };
  unsigned int m_GridShrinkFactor{ 1 };
  unsigned int m_ControlGridSmoothingIterations{ 0 };
  int          m_FeatureMapUpdateInterval{ -1 };

  // Auxiliary outputs (the primary displacement field is the ImageSource output 0).
  typename DisplacementFieldTransformType::Pointer m_DisplacementFieldTransform{ nullptr };
  typename WarpedImageType::Pointer                m_WarpedMovingImage{ nullptr };
  std::vector<double>                              m_MetricValuesPerIteration;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImpactFineRegistration.hxx"
#endif

#endif // itkImpactFineRegistration_h

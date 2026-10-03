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
// like itkImpactModelConfiguration.h and itkImpactImageToImageMetricv4.h. All torch state (the
// displacement-field leaf tensor, the grid_sample warp, the IMPACT feature loss and the
// torch::optim::Adam loop) lives in the .hxx, included only when ITK_MANUAL_INSTANTIATION
// is undefined.

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
 * TorchScript models (reusing the \c ImpactModelConfiguration / loss vocabulary of
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

  /** Optional masks (a voxel is in where the mask is not 0), each on its image's grid; absent, the whole image
   * counts, at no cost. As in the FireANTs engine, a voxel counts where the fixed mask and the moving mask warped by
   * the current field both hold (>= 0.5), on the grid each distance compares: every distance's terms are averaged
   * over those voxels only (NCC correlates them, LNCC averages its local terms there), in Static and Jacobian mode;
   * the sampled modes draw their points in the fixed mask and keep those the warped moving mask holds. */
  using MaskImageType = Image<unsigned char, ImageDimension>;
  itkSetConstObjectMacro(FixedMask, MaskImageType);
  itkGetConstObjectMacro(FixedMask, MaskImageType);
  itkSetConstObjectMacro(MovingMask, MaskImageType);
  itkGetConstObjectMacro(MovingMask, MaskImageType);

  /** Set/Get an optional initial displacement field used as a warm start (e.g. the output
   * of an affine or a ConvexAdam-style discrete stage). Must be defined on the fixed grid. */
  itkSetObjectMacro(InitialDisplacementField, DisplacementFieldType);
  itkGetConstObjectMacro(InitialDisplacementField, DisplacementFieldType);

  /** \name IMPACT feature-loss configuration (mirrors ImpactImageToImageMetricv4).
   * Leaving the model configuration empty selects the intensity (MSE) similarity. */
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

  itkSetMacro(Distance, std::vector<std::string>);
  itkGetConstReferenceMacro(Distance, std::vector<std::string>);
  /** Divide each layer's loss by its value at the stage's first iteration, so every layer starts at 1
   * and LayersWeight weighs comparable quantities (see Impact::LossNormalization). Default on. */
  itkSetMacro(NormalizeLosses, bool);
  itkGetConstMacro(NormalizeLosses, bool);
  itkBooleanMacro(NormalizeLosses);
  itkSetMacro(LayersWeight, std::vector<float>);
  itkGetConstReferenceMacro(LayersWeight, std::vector<float>);
  /** Per kept layer, the number of its channels the loss compares, drawn at random at every iteration from
   * the seeded generator (0 or a missing entry = all), as the metric's SubsetFeatures. */
  itkSetMacro(SubsetFeatures, std::vector<unsigned int>);
  itkGetConstReferenceMacro(SubsetFeatures, std::vector<unsigned int>);
  /** Side of the window the "LNCC" distance correlates each channel over, in voxels of the compared feature
   * map along its finest axis in millimetres; along each other axis the window takes the odd number of
   * voxels nearest the same length (the map's voxel side being the fixed image's times its pooling
   * factor), so it is (nearly) a cube in millimetres. Odd; default 5. */
  itkSetMacro(LNCCKernel, unsigned int);
  itkGetConstMacro(LNCCKernel, unsigned int);
  /** Share of the voxels the feature similarity reads at every iteration, drawn anew at random from the seeded
   * generator, as the metric draws its points (1 = every voxel, the default). In (0, 1]; point-wise distances and
   * NCC only (LNCC reads whole maps).
   * - Static mode: that share of each feature layer's voxels; only the drawn points are warped, so an iteration
   *   costs that share of a full one.
   * - Jacobian mode, elastix's scheme: that share of the fixed voxels, of which the points whose every model patch
   *   fits in the image are kept (elastix's SampleCheck). Each model runs on its patch (PatchSize, its receptive
   *   field: strictly positive on every model axis) around every point, cut in the fixed image and in the moving
   *   image as the field warps it -- a 2D model on a plane drawn for each point, as the metric's PatchPlane --, and
   *   its layers' centre voxels are compared. The whole images are never run through the network. */
  itkSetMacro(SamplingPercentage, double);
  itkGetConstMacro(SamplingPercentage, double);
  /** Jacobian mode with SamplingPercentage < 1: most patches one forward takes, as the metric's BatchSize, bounded
   * further by what the device holds (itkImpactBatchBudget.h); 0 (the default) = that bound alone, every point at
   * once on the CPU. A batch that runs out of device memory is replayed at half the size. */
  itkSetMacro(BatchSize, unsigned int);
  itkGetConstMacro(BatchSize, unsigned int);
  itkSetMacro(PCA, std::vector<unsigned int>);
  itkGetConstReferenceMacro(PCA, std::vector<unsigned int>);
  /** @} */

  /** Set/Get the torch device ("cpu", "cuda", "cuda:0", ...). */
  itkSetMacro(Device, std::string);
  itkGetConstReferenceMacro(Device, std::string);

  /** Set/Get the manual random seed used for any stochastic feature subsetting. */
  itkSetMacro(Seed, unsigned int);
  itkGetConstMacro(Seed, unsigned int);

  /** \name Adam / displacement-field optimization parameters.
   * The control grid, its step and its smoothing are counted in voxels of the fixed image's finest axis,
   * s_min = min_a s_a, and derived per axis, so that they are (nearly) isotropic in millimetres; on an
   * isotropic image every axis takes the numbers as they are. */
  /** @{ */
  /** Number of Adam iterations (default 80, as in ConvexAdam). */
  itkSetMacro(NumberOfIterations, unsigned int);
  itkGetConstMacro(NumberOfIterations, unsigned int);
  /** Adam learning rate, in units of s_min: the displacement is optimized in units of the finest voxel
   * side, so a step moves up to LearningRate * s_min mm along every axis (default 1.0). */
  itkSetMacro(LearningRate, double);
  itkGetConstMacro(LearningRate, double);
  itkSetMacro(Beta1, double);
  itkGetConstMacro(Beta1, double);
  itkSetMacro(Beta2, double);
  itkGetConstMacro(Beta2, double);
  itkSetMacro(Epsilon, double);
  itkGetConstMacro(Epsilon, double);
  /** Weight of the diffusion regularizer: the mean squared gradient of the (smoothed) displacement in mm
   * per mm, summed over the axes -- on an isotropic image the gradient in voxels per voxel, as ConvexAdam's
   * (default 1.25). */
  itkSetMacro(RegularizationWeight, double);
  itkGetConstMacro(RegularizationWeight, double);
  /** Low-resolution control-grid shrink factor, in voxels of the finest axis: along axis a the field is
   * optimized on a grid shrink_a = max(1, round(GridShrinkFactor * s_min / s_a)) times coarser than the
   * image, and upsampled to full resolution each iteration (ConvexAdam-style) -- faster on large volumes
   * and adds regularity. Default 1 (full resolution). */
  itkSetMacro(GridShrinkFactor, unsigned int);
  itkGetConstMacro(GridShrinkFactor, unsigned int);
  /** Number of 3x3x3 average-pool smoothing passes applied to the control grid each iteration
   * (B-spline-like control-point smoothing), in control cells, which GridShrinkFactor makes (nearly)
   * isotropic in millimetres. Default 0. */
  itkSetMacro(ControlGridSmoothingIterations, unsigned int);
  itkGetConstMacro(ControlGridSmoothingIterations, unsigned int);
  /** In feature mode, re-extract the moving feature maps from the currently-warped moving image
   * every this many Adam iterations. Helps large deformations, where warping a precomputed feature
   * map diverges from the features of the warped image. <= 0 extracts once (disabled). Default -1. */
  itkSetMacro(FeatureMapUpdateInterval, int);
  itkGetConstMacro(FeatureMapUpdateInterval, int);

  /** Feature-mode strategy, mirroring itk::ImpactImageToImageMetricv4's "Static"/"Jacobian" modes:
   * - "Static" (default): the moving features are extracted once and WARPED each iteration
   *   (grid_sample of a frozen feature map). Fast, and exact for a LOCAL descriptor (MIND) where
   *   F(warp(I)) == warp(F(I)).
   * - "Jacobian": every iteration warps the moving IMAGE by the field and RE-EXTRACTS the features
   *   through the network with autograd, so the loss is the true F(warp(I)) and its gradient carries
   *   the network term d(feature)/d(displacement) -- needed for a non-local backbone (SAM), whose
   *   warped feature map is not the feature of the warped image.
   * (The metric's "Jacobian" samples per-point patches; here, on the dense field, it re-extracts the
   * whole warped image -- same principle, differentiating the similarity through the model.) */
  itkSetMacro(Mode, std::string);
  itkGetConstReferenceMacro(Mode, std::string);
  /** "Jacobian" mode with a model of lower dimension than the image (a 2D model on a volume): number of
   * slices extracted per autograd chunk, to bound peak memory -- the per-voxel feature loss decomposes
   * over the slices, so chunk gradients accumulate into the field. The slices run along one image axis
   * drawn at random (from Seed) at every iteration, the fixed features of each axis extracted once, so
   * the network sees the anatomy in the three orientations over the iterations. 0 = whole volume in one
   * graph. Default 32. */
  itkSetMacro(FeatureChunkSize, unsigned int);
  itkGetConstMacro(FeatureChunkSize, unsigned int);
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
  typename FixedImageType::ConstPointer   m_FixedImage{ nullptr };
  typename MovingImageType::ConstPointer  m_MovingImage{ nullptr };
  typename MaskImageType::ConstPointer    m_FixedMask{ nullptr };
  typename MaskImageType::ConstPointer    m_MovingMask{ nullptr };
  typename DisplacementFieldType::Pointer m_InitialDisplacementField{ nullptr };

  std::vector<ImpactModelConfiguration> m_FixedModelsConfiguration;
  std::vector<ImpactModelConfiguration> m_MovingModelsConfiguration;
  std::vector<std::string>              m_Distance;
  std::vector<float>                    m_LayersWeight;
  std::vector<unsigned int>             m_SubsetFeatures;
  std::vector<unsigned int>             m_PCA;

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
  bool         m_NormalizeLosses{ true };
  unsigned int m_LNCCKernel{ 5 };
  double       m_SamplingPercentage{ 1.0 };
  unsigned int m_BatchSize{ 0 };
  std::string  m_Mode{ "Static" };
  unsigned int m_FeatureChunkSize{ 32 };

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

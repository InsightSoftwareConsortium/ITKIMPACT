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
#include <itkTransform.h>
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
 * TorchScript model (not tied to MIND specifically). With the authors' settings (one MIND layer, L2,
 * GridSpacing 6, DisplacementHalfWidth 4, inverse consistency) on an isotropic pair without masks, the field is
 * theirs: test/itkImpactConvexAdamParityTest.py compares the two.
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
   * Counted in voxels of the fixed image, the same number along every axis whatever its spacing, as ConvexAdam
   * counts them. */
  /** @{ */
  /** Cell size of the coarse grid the cost volume runs on, in voxels (avg-pool kernel and stride). The cost
   * smoothing, the NCC/LNCC window and the displacement smoothing are counted in cells. Default 4. */
  itkSetMacro(GridSpacing, unsigned int);
  itkGetConstMacro(GridSpacing, unsigned int);
  /** Displacement search half-width, in cells: DisplacementHalfWidth * GridSpacing voxels each way along every
   * axis, (2 DisplacementHalfWidth + 1)^Dim candidates. Default 3. */
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

  /** \name Region sampling: an out-of-core run.
   * With RegionSize > 0 the images are never read whole. One pass over boxes of the fixed grid, each RegionSize
   * voxels a side (whole cells of the coarse grid), reads each box of the fixed image and of the
   * moving image (brought onto the fixed grid through the identity, region by region) with its halo, extracts their
   * features and pools them to the coarse grid; the cost volume is then built by boxes of
   * that grid, each within CostBoxBytes, never whole, and the coupled-convex iterations and the inverse consistency
   * run on the pooled maps. The field is the whole-image run's to float precision. Where the pooled maps fit
   * ResidentFeatureBudget they stay on the device with the fields of the solve, which takes each of its steps on
   * the grid whole. Otherwise they stay in files of ScratchDirectory mapped to memory, and every step goes by the
   * boxes of the cost volume, each brought to the device with the cells around it the step reads: the cell window's
   * two for a cost, one for the smoothing of the chosen candidates, the displacement half-width (stretched by n /
   * (n - 1) on an axis of n cells) and one for the inverse consistency. Nothing the run then holds in host memory or
   * on the device grows with the image, save about a hundred bytes a box (the lists of the boxes): the coarse grid of
   * GetCoarseGridTransform() is a mapped file too. The images are read through their
   * own pipelines (SetRequestedRegion then Update: an itk::ImageFileReader with UseStreamingOn reads those regions
   * alone; an image in memory serves them from its buffer); the caller keeps its readers alive. Request a region of
   * GetOutput() (or stream it to a writer) rather than the whole field, and take the field on the coarse grid from
   * GetCoarseGridTransform(). A feature layer a power of two coarser than the image, no coarser than the coarse
   * cells, is computed by regions too, each region starting on its lattice. PCA fits its basis on the whole feature
   * map and a model resampling the image to its own voxel size does not line up with the boxes: refused. */
  /** @{ */
  /** Side of a box, in voxels; 0 (the default) reads the images whole. A model runs on a box with RegionHalo voxels
   * around it, whole, and once on a probe of 65 voxels a side, which tells where its layers' samples lie. */
  itkSetMacro(RegionSize, unsigned int);
  itkGetConstMacro(RegionSize, unsigned int);
  /** Voxels read around a box, on every axis, so that its features equal those of the whole image: the receptive
   * field of the feature model, in its input's voxels (0 for a point-wise model). */
  itkSetMacro(RegionHalo, unsigned int);
  itkGetConstMacro(RegionHalo, unsigned int);
  /** A transform from the fixed to the moving physical space the moving image and its mask are read through with
   * RegionSize (an affine pre-alignment): each of their regions is brought onto the fixed grid through it, by linear
   * interpolation (nearest neighbour for the mask), by a resampler that asks the moving image for that region alone.
   * Without one, the identity. */
  using TransformType = Transform<double, ImageDimension, ImageDimension>;
  itkSetConstObjectMacro(MovingTransform, TransformType);
  itkGetConstObjectMacro(MovingTransform, TransformType);
  /** Bytes a box of the cost volume may take (default 1 GiB): the coarse grid is tiled into boxes whose cost over
   * every candidate, with the margins the cell window and the candidates read, fits it. With the pooled maps in
   * mapped files it bounds what a step of the solve brings to the device: 8 bytes a cell and a candidate, of which
   * the other steps take a small part. */
  itkSetMacro(CostBoxBytes, SizeValueType);
  itkGetConstMacro(CostBoxBytes, SizeValueType);
  /** Most bytes the pooled feature maps of both images and the masks' shares may take to stay resident on the
   * device. 0 (the default): half the device memory free when the run starts, a quarter of the machine's on the
   * CPU, less what the fields of the solve take beside them. GetResidentFeatures() tells which route a run took. */
  itkSetMacro(ResidentFeatureBudget, SizeValueType);
  itkGetConstMacro(ResidentFeatureBudget, SizeValueType);
  itkGetConstMacro(ResidentFeatures, bool);
  /** Where the run keeps what grows with the image when the pooled maps are not resident: the maps, the masks'
   * shares, the fields of the solve and the coarse grid, in files mapped to memory that the system pages in and
   * out. Empty (the default): the system's temporary directory. A directory held in memory (a tmpfs) keeps them
   * there: give one on a disk for an image larger than the memory. */
  itkSetStringMacro(ScratchDirectory);
  itkGetStringMacro(ScratchDirectory);
  /** Bytes requested from the images and masks over the run. */
  itkGetConstMacro(BytesRead, SizeValueType);
  /** Whether the fixed (moving) image served its regions without buffering it whole. */
  itkGetConstMacro(FixedImageStreamed, bool);
  itkGetConstMacro(MovingImageStreamed, bool);
  /** The field on the coarse grid as a transform: one sample per cell at the cell's centre, padded by two cells of
   * replicated border, so that its linear interpolation is the map the dense field holds at every voxel. With
   * RegionSize > 0 it is solved without allocating the dense field. After a run by regions whose pooled maps were
   * not resident, the pixels of that field are a file of ScratchDirectory mapped to memory: the field's pixel
   * container holds the mapping, so they stay valid for as long as the field, or the transform, is referenced, past
   * the filter and its next run, and the file is removed when the last reference goes. */
  DisplacementFieldTransformType *
  GetCoarseGridTransform();
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

  /** With RegionSize > 0: the requested region of the dense field, interpolated from the coarse grid. */
  void
  DynamicThreadedGenerateData(const typename Superclass::OutputImageRegionType & region) override;

  /** The region-sampled run (RegionSize > 0): solves the coarse grid without reading an image whole. */
  void
  SolveByRegions();

  /** The coupled-convex solve both runs end in, on the two images' features pooled to the coarse grid ({1, C,
   * coarse...} a layer) and the masks' shares of its cells (undefined without a mask): sets the field on the coarse
   * grid and returns it, in voxels. `cost` builds the cost volume of one of `boxes`, which tile the grid. Without
   * `scratch` the maps are on the device and the solve returns its field there; with it they are mapped files, as
   * the fields of the solve then are in `scratch`, every step goes by `boxes`, and nothing is returned. The
   * tensors, the boxes and the scratch files are LibTorch-backed types this header does not name. */
  template <typename TTensor, typename TBox, typename TCost, typename TScratch>
  TTensor
  SolveOnCoarseGrid(const std::vector<TTensor> & fixedMaps,
                    const std::vector<TTensor> & movingMaps,
                    const TTensor &              fixedShare,
                    const TTensor &              movingShare,
                    const std::vector<TBox> &    boxes,
                    TCost &&                     cost,
                    TScratch *                   scratch);

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

  unsigned int                         m_RegionSize{ 0 };
  unsigned int                         m_RegionHalo{ 0 };
  typename TransformType::ConstPointer m_MovingTransform{ nullptr };
  SizeValueType                        m_CostBoxBytes{ static_cast<SizeValueType>(1) << 30 };
  SizeValueType                        m_ResidentFeatureBudget{ 0 };
  bool                                 m_ResidentFeatures{ false };
  std::string                          m_ScratchDirectory;
  SizeValueType                        m_BytesRead{ 0 };
  bool                                 m_FixedImageStreamed{ false };
  bool                                 m_MovingImageStreamed{ false };
  ModifiedTimeType                     m_SolvedTime{ 0 };

  typename DisplacementFieldTransformType::Pointer m_DisplacementFieldTransform{ nullptr };
  typename DisplacementFieldType::Pointer          m_CoarseGrid{ nullptr };
  typename DisplacementFieldTransformType::Pointer m_CoarseGridTransform{ nullptr };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImpactCoarseRegistration.hxx"
#endif

#endif // itkImpactCoarseRegistration_h

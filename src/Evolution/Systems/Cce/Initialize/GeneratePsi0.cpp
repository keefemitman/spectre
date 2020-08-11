// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/GeneratePsi0.hpp"

#include <array>
#include <boost/math/differentiation/finite_difference.hpp>
#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Transpose.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/NewmanPenrose.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace InitializeJ {
namespace detail {

void read_in_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        j_container,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        dr_j_container,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        r_container,
    const std::vector<std::string> files,
    const size_t l_max,
    const size_t target_idx,
    const double target_time) noexcept {
  Parallel::printf("reading\n");
  SpecWorldtubeH5BufferUpdater target_buffer_updater{files[target_idx]};
  const double target_radius = target_buffer_updater.get_extraction_radius();

  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      variables{number_of_angular_points};
  for(size_t i = 0; i < files.size(); ++i) {
    SpecWorldtubeH5BufferUpdater buffer_updater{files[i]};
    WorldtubeDataManager data_manager{
        std::make_unique<
            SpecWorldtubeH5BufferUpdater>(files[i]),
        l_max, 100,
        std::make_unique<
            intrp::BarycentricRationalSpanInterpolator>(10_st, 10_st)};

    const double ext_radius = buffer_updater.get_extraction_radius();
    const double corrected_time = (ext_radius - target_radius) + target_time;
    data_manager.populate_hypersurface_boundary_data(
        make_not_null(&variables), corrected_time);

    ComplexDataVector angular_view_j{
        get(*j_container).data().data()+
            get(get<Tags::BoundaryValue<Tags::BondiJ>>(
                variables)).size() * i,
        get(get<Tags::BoundaryValue<Tags::BondiJ>>(
            variables)).size()};
    angular_view_j =
        get(get<Tags::BoundaryValue<Tags::BondiJ>>(variables)).data();
    ComplexDataVector angular_view_dr_j{
        get(*dr_j_container).data().data()+
            get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
                variables)).size() * i,
        get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
            variables)).size()};
    angular_view_dr_j =
        get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(variables)).data();
    ComplexDataVector angular_view_r{
        get(*r_container).data().data()+
            get(get<Tags::BoundaryValue<Tags::BondiR>>(
                variables)).size() * i,
        get(get<Tags::BoundaryValue<Tags::BondiR>>(
            variables)).size()};
    angular_view_r =
        get(get<Tags::BoundaryValue<Tags::BondiR>>(variables)).data();
  }
}

void second_derivative_of_j_from_worldtubes(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const size_t l_max,
    const size_t target_idx) noexcept {
  Parallel::printf("dr_dr_j\n");
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(r).size() / number_of_angular_points;
  auto r_transpose = transpose(
      get(r).data(), number_of_angular_points, number_of_radial_points);
  auto dr_j_transpose = transpose(
      get(dr_j).data(), number_of_angular_points, number_of_radial_points);

  Parallel::printf("for loop\n");
  for(size_t i = 0; i < number_of_angular_points; ++i) {
    const DataVector r_real_part = real(get(r).data());
    const DataVector dr_j_real_part = real(get(dr_j).data());
    const DataVector r_imag_part = imag(get(r).data());
    const DataVector dr_j_imag_part = imag(get(dr_j).data());
    gsl::span<const double> span_r_real_part(
        r_real_part.data()
            + number_of_radial_points * i, number_of_radial_points);
    gsl::span<const double> span_dr_j_real_part(
        dr_j_real_part.data()
            + number_of_radial_points * i, number_of_radial_points);
    gsl::span<const double> span_r_imag_part(
        r_imag_part.data()
            + number_of_radial_points * i, number_of_radial_points);
    gsl::span<const double> span_dr_j_imag_part(
        dr_j_imag_part.data()
            + number_of_radial_points * i, number_of_radial_points);
    intrp::BarycentricRationalSpanInterpolator interpolator{3_st, 4_st};

    Parallel::printf("span stuff\n");
    auto interpolated_dr_j_real_part =
        [&span_r_real_part, &span_dr_j_real_part, &interpolator](const double r)
        noexcept {
            return interpolator.interpolate(
                span_r_real_part, span_dr_j_real_part, r);
        };
    auto interpolated_dr_j_imag_part =
        [&span_r_real_part, &span_dr_j_imag_part, &interpolator](const double r)
        noexcept {
            return interpolator.interpolate(
                span_r_real_part, span_dr_j_imag_part, r);
        };

    Parallel::printf("real\n");
    auto real_dr_dr_j = boost::math::differentiation::
        finite_difference_derivative(interpolated_dr_j_real_part,
            r_real_part.data()[target_idx + number_of_angular_points * i]);
    Parallel::printf("imag\n");
    auto imag_dr_dr_j = boost::math::differentiation::
        finite_difference_derivative(interpolated_dr_j_imag_part,
            r_real_part.data()[target_idx + number_of_angular_points * i]);
    Parallel::printf("combine\n");
    get(*dr_dr_j).data()[i] = std::complex(real_dr_dr_j, imag_dr_dr_j);
  }
}
}  // namespace detail

GeneratePsi0::GeneratePsi0(
    std::vector<std::string> files,
    const size_t target_idx,
    const double target_time) noexcept
    : files_{files},
      target_idx_{target_idx},
      target_time_{target_time} {}

std::unique_ptr<InitializeJ> GeneratePsi0::get_clone() const noexcept {
  return std::make_unique<GeneratePsi0>(*this);
}

void GeneratePsi0::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points) const noexcept {
  Parallel::printf("starting\n");
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Scalar<SpinWeighted<ComplexDataVector, 2>> j_container{
      files_.size()*number_of_angular_points};
  Scalar<SpinWeighted<ComplexDataVector, 2>> dr_j_container{
      files_.size()*number_of_angular_points};
  Scalar<SpinWeighted<ComplexDataVector, 0>> r_container{
      files_.size()*number_of_angular_points};
  detail::read_in_worldtube_data(make_not_null(&j_container),
      make_not_null(&dr_j_container), make_not_null(&r_container),
      files_, l_max, target_idx_, target_time_);

  // compute dr_dr_j
  Scalar<SpinWeighted<ComplexDataVector, 2>> dr_dr_j_at_radius{
      number_of_angular_points};
  detail::second_derivative_of_j_from_worldtubes(
      make_not_null(&dr_dr_j_at_radius),
      dr_j_container, r_container, l_max, target_idx_);

  Parallel::printf("psi_0 variables\n");
  // acquire variables for psi_0
  size_t start_idx = number_of_angular_points * target_idx_;
  Scalar<SpinWeighted<ComplexDataVector, 2>> j_at_radius;
  get(j_at_radius)
      .set_data_ref(
          get(j_container).data().data()
              + start_idx, number_of_angular_points);
  Scalar<SpinWeighted<ComplexDataVector, 2>> dr_j_at_radius;
  get(dr_j_at_radius)
      .set_data_ref(
          get(dr_j_container).data().data()
              + start_idx, number_of_angular_points);
  Scalar<SpinWeighted<ComplexDataVector, 0>> r_at_radius;
  get(r_at_radius)
      .set_data_ref(
          get(r_container).data().data()
              + start_idx, number_of_angular_points);
  Scalar<SpinWeighted<ComplexDataVector, 0>> k_at_radius{
      sqrt(1.0 + get(j_at_radius).data() * conj(get(j_at_radius).data()))};

  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_j_at_radius{
      get(r_at_radius).data() * get(dr_j_at_radius).data()};
  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_dy_j_at_radius{
      get(r_at_radius).data() * get(dr_dr_j_at_radius).data()};

  Parallel::printf("compute psi_0\n");
  // compute psi_0
  Scalar<SpinWeighted<ComplexDataVector, 0>> one_minus_y{
      number_of_angular_points};
  get(one_minus_y).data() = std::complex<double>(2.0,0.0);
  Scalar<SpinWeighted<ComplexDataVector, 2>> psi_0{
      number_of_angular_points};
  VolumeWeyl<Tags::Psi0>::apply(make_not_null(&psi_0),
                                j_at_radius,
                                dy_j_at_radius,
                                dy_dy_j_at_radius,
                                k_at_radius,
                                r_at_radius,
                                one_minus_y);
  Parallel::printf("print psi_0\n");
  for(int i = 0; i < get(psi_0).data().size(); ++i) {
    Parallel::printf(std::to_string(real(get(psi_0).data()[i]))+"\n");
  }
}

void GeneratePsi0::pup(PUP::er& p) noexcept {
  p | files_;
  p | target_idx_;
  p | target_time_;
}

/// \cond
PUP::able::PUP_ID GeneratePsi0::my_PUP_ID = 0;
/// \endcond
} // namespace InitializeJ
} // namespace Cce

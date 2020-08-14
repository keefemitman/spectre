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
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
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

  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(r).size() / number_of_angular_points;

  auto r_transpose = transpose(get(r).data(),
      number_of_angular_points, number_of_radial_points);
  auto dr_j_transpose = transpose(get(dr_j).data(),
      number_of_angular_points, number_of_radial_points);

  for(size_t i = 0; i < number_of_angular_points; ++i) {
    const DataVector r_real_part = real(r_transpose);
    const DataVector dr_j_real_part = real(dr_j_transpose);
    const DataVector dr_j_imag_part = imag(dr_j_transpose);
    gsl::span<const double> span_r_real_part(
        r_real_part.data()
            + number_of_radial_points * i, number_of_radial_points);
    gsl::span<const double> span_dr_j_real_part(
        dr_j_real_part.data()
            + number_of_radial_points * i, number_of_radial_points);
    gsl::span<const double> span_dr_j_imag_part(
        dr_j_imag_part.data()
            + number_of_radial_points * i, number_of_radial_points);
    intrp::BarycentricRationalSpanInterpolator interpolator{3_st, 4_st};

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

    auto real_dr_dr_j = boost::math::differentiation::
        finite_difference_derivative(interpolated_dr_j_real_part,
            r_real_part.data()[target_idx + number_of_radial_points * i]);
    auto imag_dr_dr_j = boost::math::differentiation::
        finite_difference_derivative(interpolated_dr_j_imag_part,
            r_real_part.data()[target_idx + number_of_radial_points * i]);
    get(*dr_dr_j).data()[i] = std::complex(real_dr_dr_j, imag_dr_dr_j);
  }
}

void radial_evolve_psi0_condition(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> volume_j_id,
    const SpinWeighted<ComplexDataVector, 2>& boundary_j,
    const SpinWeighted<ComplexDataVector, 2>& boundary_dr_j,
    const SpinWeighted<ComplexDataVector, 0>& boundary_psi_0,
    const SpinWeighted<ComplexDataVector, 0>& r, const size_t l_max,
    const size_t number_of_radial_points) noexcept {
  // use the maximum to measure the scale for the vector quantities
  const double j_scale = max(abs(boundary_j.data()));
  const double dy_j_scale = max(abs(0.5 * boundary_dr_j.data() * r.data()));
  // set initial step size according to the first couple of steps in section
  // II.4 of Solving Ordinary Differential equations by Hairer, Norsett, and
  // Wanner
  double initial_radial_step = 1.0e-6;
  if (j_scale > 1.0e-5 and dy_j_scale > 1.0e-5) {
    initial_radial_step = 0.01 * j_scale / dy_j_scale;
  }

  const auto psi_0_condition_system =
      [](const std::array<ComplexDataVector, 2>& bondi_j_and_i,
         std::array<ComplexDataVector, 2>& dy_j_and_dy_i,
         std::array<ComplexDataVector, 0>& psi_0
         const double y) noexcept {
        dy_j_and_dy_i[0] = bondi_j_and_i[1];
        const auto& bondi_j = bondi_j_and_i[0];
        const auto& bondi_i = bondi_j_and_i[1];
        const auto& bondi_psi_0 = psi_0[0];
        dy_j_and_dy_i[1] =
            0.5 *
            (conj(bondi_psi_0) * square(bondi_j)
            / (2.0 + conj(bondi_j) * bondi_j +
               2.0 * sqrt(1.0 + conj(bondi_j) * bondi_j)) +
            bondi_psi_0) +
            -0.0625 *
            (square(conj(bondi_i) * bondi_j) + square(conj(bondi_j) * bondi_i) -
             2.0 * bondi_i * conj(bondi_i) * (2.0 + bondi_j * conj(bondi_j))) *
            (4.0 * bondi_j + bondi_i * (1.0 - y)) /
            (1.0 + bondi_j * conj(bondi_j));
      };

  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<
          boost::numeric::odeint::runge_kutta_dopri5<
              std::array<ComplexDataVector, 2>>>>
      dense_stepper = boost::numeric::odeint::make_dense_output(
          1.0e-14, 1.0e-14,
          boost::numeric::odeint::runge_kutta_dopri5<
              std::array<ComplexDataVector, 2>>{});
  dense_stepper.initialize(
      std::array<ComplexDataVector, 2>{
          {boundary_j.data(), boundary_dr_j.data() * r.data()}},
      std::array<ComplexDataVector, 0>{
          {boundary_psi_0.data()}},
      -1.0, initial_radial_step);
  auto state_buffer =
      std::array<ComplexDataVector, 2>{{ComplexDataVector{boundary_j.size()},
                                        ComplexDataVector{boundary_j.size()}}};

  std::pair<double, double> step_range =
      dense_stepper.do_step(psi_0_condition_system);
  const auto& y_collocation =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
                                       number_of_radial_points);
  for (size_t y_collocation_point = 0;
       y_collocation_point < number_of_radial_points; ++y_collocation_point) {
    while(step_range.second < y_collocation[y_collocation_point]) {
      step_range = dense_stepper.do_step(psi_0_condition_system);
    }
    if (step_range.second < y_collocation[y_collocation_point] or
        step_range.first > y_collocation[y_collocation_point]) {
      ERROR(
          "Psi 0 radial integration failed. The current y value is "
          "incompatible with the required Gauss-Lobatto point.");
    }
    dense_stepper.calc_state(y_collocation[y_collocation_point], state_buffer);
    ComplexDataVector angular_view{
        volume_j_id->data().data() +
            y_collocation_point *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    angular_view = state_buffer[0];
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
      get(r_at_radius).data() * get(r_at_radius).data()
          * get(dr_dr_j_at_radius).data()};

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

  //Parallel::printf("%s / %s: %s \n",
  //    "index","number of indices","Psi_0+dr_dr_j");
  //for(int i = 0; i < get(psi_0).data().size(); ++i) {
  //  Parallel::printf(
  //      "%d / %d: %e + %e i \n",
  //      i, get(psi_0).data().size()-1,
  //      real(get(psi_0).data()[i]+get(dr_dr_j_at_radius).data()[i]),
  //      imag(get(psi_0).data()[i]+get(dr_dr_j_at_radius).data()[i]));
  //}

  detail::radial_evolve_psi_0_condition(
      make_not_null(&get(*j)), get(j_at_radius),
      get(dr_j_at_radius), get(r), l_max, number_of_radial_points);
  const SpinWeighted<ComplexDataVector, 2> j_at_scri_view;
  make_const_view(make_not_null(&j_at_scri_view), get(*j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  const double final_angular_coordinate_deviation =
      detail::adjust_angular_coordinates_for_j(
          j, cartesian_cauchy_coordinates, angular_cauchy_coordinates,
          j_at_scri_view, l_max, angular_coordinate_tolerance_, max_iterations_,
          true);

  bool require_convergence_ = false;
  double angular_coordinate_tolerance_ = 1.0e-14;
  size_t max_iterations_ = 1000;
  if (final_angular_coordinate_deviation > angular_coordinate_tolerance_ and
      require_convergence_) {
    ERROR(
        "Initial data iterative angular solve did not reach "
        "target tolerance "
        << angular_coordinate_tolerance_ << ".\n"
        << "Exited after " << max_iterations_
        << " iterations, achieving final\n"
           "maximum over collocation points deviation of J from target of "
        << final_angular_coordinate_deviation);
  } else if (final_angular_coordinate_deviation >
             angular_coordinate_tolerance_) {
    Parallel::printf(
        "Warning: iterative angular solve did not reach "
        "target tolerance %e.\n"
        "Exited after %zu iterations, achieving final maximum over "
        "collocation points deviation of J from target of %e\n"
        "Proceeding with evolution using the partial result from partial "
        "angular solve.",
        angular_coordinate_tolerance_, max_iterations_,
        final_angular_coordinate_deviation);
  }

  Parallel::printf("Finished Running!\n");
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

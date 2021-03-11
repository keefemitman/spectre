// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"

#include <cstddef>
#include <cmath>
#include <memory>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "NumericalAlgorithms/LinearOperators/IndefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace Cce::InitializeJ {
namespace detail {

double adjust_angular_coordinates_for_omega(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const SpinWeighted<ComplexDataVector, 0>& target_omega, const size_t l_max,
    const double tolerance, const size_t max_steps,
    const bool adjust_volume_gauge) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_phi_points =
      Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max);
  const size_t number_of_theta_points =
      Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max);

  Parallel::printf("\n");
  Parallel::printf("Operating with (Tolerance, Max Steps): (%e, %zu)\n",
                   tolerance, max_steps);

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
    get<0>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.theta;
    get<1>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.phi;
  }

  get<0>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      cos(get<1>(*angular_cauchy_coordinates));
  get<1>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      sin(get<1>(*angular_cauchy_coordinates));
  get<2>(*cartesian_cauchy_coordinates) =
      cos(get<0>(*angular_cauchy_coordinates));

  Variables<tmpl::list<
      // gauge Jacobians
      ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // gauge conformal factor
      ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // Integral Input
      ::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // Integral Result
      ::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // Interpolated target omega
      ::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                           std::integral_constant<int, 0>>>>
    computation_buffers{number_of_angular_points};

  auto& gauge_c =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          computation_buffers);
  auto& gauge_d =
      get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& gauge_omega =
      get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& integral_input =
      get<::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& integral_result =
      get<::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& interpolated_target_omega =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  // Initialize
  interpolated_target_omega.data() = target_omega.data();

  // Update angular coordinates so they are formatted correctly,
  // i.e., phi in [-pi, pi]
  GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                            cartesian_cauchy_coordinates);

  double max_error = 1.0;
  size_t number_of_steps = 0;
  while (true) {
    // Compute the Integrand
    // Note that there is no factor of sin(\theta) because this is
    // removed to convert the integration to the angular domain.
    get(integral_input).data() =
        pow(interpolated_target_omega.data(), 2.0);

    // Obtain theta via Integration
    const Mesh<2> Mesh2d{
        {{number_of_phi_points, number_of_theta_points}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss}}};
    indefinite_integral(make_not_null(&get(integral_result).data()),
                        get(integral_input).data(),
                        Mesh2d,
                        1);

    // Update the angular coordinates
    for(size_t i = 0; i < get<0>(*angular_cauchy_coordinates).size(); ++i) {
      get<0>(*angular_cauchy_coordinates)[i] =
          acos(-real(get(integral_result).data()[i]) + 1.0);
    }

    // Update the Cartesian coordinates
    get<0>(*cartesian_cauchy_coordinates) =
        sin(get<0>(*angular_cauchy_coordinates)) *
        cos(get<1>(*angular_cauchy_coordinates));
    get<1>(*cartesian_cauchy_coordinates) =
        sin(get<0>(*angular_cauchy_coordinates)) *
        sin(get<1>(*angular_cauchy_coordinates));
    get<2>(*cartesian_cauchy_coordinates) =
        cos(get<0>(*angular_cauchy_coordinates));

    // Update angular coordinates so they are formatted correctly,
    // i.e., phi in [-pi, pi]
    GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                            cartesian_cauchy_coordinates);

    // Update Jacobian factors
    GaugeUpdateJacobianFromCoordinates<
        Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                            make_not_null(&gauge_d),
                                            angular_cauchy_coordinates,
                                            *cartesian_cauchy_coordinates,
                                            l_max);

    // Compute omega from new angular coordinates
    get(gauge_omega).data() =
        0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                   get(gauge_c).data() * conj(get(gauge_c).data()));

    // Interpolate target omega onto new angular coordinates
    Spectral::Swsh::SwshInterpolator iteration_interpolator{
        get<0>(*angular_cauchy_coordinates),
        get<1>(*angular_cauchy_coordinates), l_max};

    iteration_interpolator.interpolate(
        make_not_null(&interpolated_target_omega),
        target_omega);

    max_error = max(abs(get(gauge_omega).data()
                        - interpolated_target_omega.data()));
    Parallel::printf("Debug Integral solve: %e\n", max_error);
    ++number_of_steps;
    if (max_error > 2.0e+0) {
      ERROR(
          "Iterative solve for surface coordinates of initial data failed. The "
          "strain is too large to be fully eliminated by a well-behaved "
          "alteration of the spherical mesh. For this data, please use an "
          "alternative initial data generator such as "
          "`InitializeJConformalFactor`.\n");
    }
    if (max_error < tolerance) {
      Parallel::printf("Tolerance Reached!\n");
      break;
    }
    if (number_of_steps > max_steps) {
      Parallel::printf("Max Number of Steps Exceeded...\n");
      break;
    }
  }

  // Use the finalied coordinates to update J
  GaugeAdjustInitialJ::apply(volume_j, gauge_c, gauge_d, gauge_omega,
                             *angular_cauchy_coordinates, l_max);

  Parallel::printf("Integral solve: %e\n", max_error);
  return max_error;
}
}  // namespace detail


std::unique_ptr<InitializeJ> ConformalFactor::get_clone() const
    noexcept {
  return std::make_unique<ConformalFactor>();
}

void ConformalFactor::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*boundary_dr_j*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*r*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta, const size_t l_max,
    const size_t number_of_radial_points) const noexcept {
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  Scalar<SpinWeighted<ComplexDataVector, 2>> first_angular_view_j{};
  get(first_angular_view_j).data().set_data_ref(get(*j).data().data(),
                                           get(boundary_j).size());
  get(first_angular_view_j) = get(boundary_j);
  detail::adjust_angular_coordinates_for_omega(
      make_not_null(&first_angular_view_j), cartesian_cauchy_coordinates,
      angular_cauchy_coordinates, exp(2.0 * get(beta)), l_max, 1.0e-10, 100_st,
      true);
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,
        get(boundary_j).size()};
    // auto is acceptable here as these two values are only used once in the
    // below computation. `auto` causes an expression template to be
    // generated, rather than allocating.
    const auto one_minus_y_coefficient = 0.5 * get(first_angular_view_j).data();
    angular_view_j = one_minus_y_collocation[i] * one_minus_y_coefficient;
  }
}

void ConformalFactor::pup(PUP::er& /*p*/) noexcept {}

/// \cond
PUP::able::PUP_ID ConformalFactor::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce::InitializeJ

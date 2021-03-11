// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"

#include <cstddef>
#include <cmath>
#include <memory>
#include <type_traits>

#include <iostream>
#include <fstream>
#include <string>

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

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
    get<0>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.theta;
    get<1>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.phi;
  }

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

  // Compute the Integrand
  auto& integral_input =
      get<::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  // Note that there is no factor of sin(\theta) because this is
  // removed to convert the integration to the angular domain.
  get(integral_input).data() =
      pow(target_omega.data(), 2.0);

  // Obtain theta via Integration (loop over the phis)
  auto& integral_result =
      get<::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  const Mesh<2> Mesh2d{
      {{number_of_phi_points, number_of_theta_points}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss}}};
  indefinite_integral(make_not_null(&get(integral_result).data()),
                      get(integral_input).data(),
                      Mesh2d,
                      1);

  // this is just for debugging
  Parallel::printf("\n");
  Parallel::printf("Omega Min Max: %e %e\n",
      min(real(target_omega.data())),
      max(real(target_omega.data())));
  Parallel::printf("Integrand Min Max: %e %e\n",
      min(real(get(integral_input).data())),
      max(real(get(integral_input).data())));
  Parallel::printf("Integral Min Max: %e %e\n",
      min(real(get(integral_result).data())),
      max(real(get(integral_result).data())));
  Parallel::printf("Result Min Max: %e %e\n",
      min(-real(get(integral_result).data()) + 1.0),
      max(-real(get(integral_result).data()) + 1.0));
  Parallel::printf("ArcCos Min Max: %e %e\n",
      acos(max(-real(get(integral_result).data()) + 1.0)),
      acos(min(-real(get(integral_result).data()) + 1.0)));

  for (size_t i = 0; i < get<0>(*angular_cauchy_coordinates).size(); ++i) {
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

  // Use the new coordinates to create new gauge_c, gauge_d
  GaugeUpdateJacobianFromCoordinates<
      Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                          make_not_null(&gauge_d),
                                          angular_cauchy_coordinates,
                                          *cartesian_cauchy_coordinates,
                                          l_max);

  // Compute omega and check the error
  get(gauge_omega).data() =
      0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                 get(gauge_c).data() * conj(get(gauge_c).data()));

  double max_error = 1.0;
  max_error = max(abs(get(gauge_omega).data() - target_omega.data()));

  // then run
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

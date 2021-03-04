// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"

#include <cstddef>
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
      // cartesian coordinates
      ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // eth of cartesian coordinates
      ::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      // eth of gauge-transformed cartesian coordinates
      ::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<8, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      // gauge Jacobians
      ::Tags::SpinWeighted<::Tags::TempScalar<12, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<13, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // gauge Jacobians on next iteration
      ::Tags::SpinWeighted<::Tags::TempScalar<14, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<15, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // gauge conformal factor
      ::Tags::SpinWeighted<::Tags::TempScalar<16, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // cartesian coordinates steps
      ::Tags::SpinWeighted<::Tags::TempScalar<17, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<18, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<19, ComplexDataVector>,
                           std::integral_constant<int, 0>>>>
      computation_buffers{number_of_angular_points};

  auto& x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));

  x.data() =
      std::complex<double>(1.0, 0.0) * get<0>(*cartesian_cauchy_coordinates);
  y.data() =
      std::complex<double>(1.0, 0.0) * get<1>(*cartesian_cauchy_coordinates);
  z.data() =
      std::complex<double>(1.0, 0.0) * get<2>(*cartesian_cauchy_coordinates);

  auto& eth_x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));

  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Eth>>(l_max, 1, make_not_null(&eth_x),
                                             make_not_null(&eth_y),
                                             make_not_null(&eth_z), x, y, z);

  auto& evolution_gauge_eth_x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& evolution_gauge_eth_y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& evolution_gauge_eth_z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<8, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));

  auto& gauge_c =
      get<::Tags::SpinWeighted<::Tags::TempScalar<12, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          computation_buffers);
  auto& gauge_d =
      get<::Tags::SpinWeighted<::Tags::TempScalar<13, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& next_gauge_c =
      get<::Tags::SpinWeighted<::Tags::TempScalar<14, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          computation_buffers);
  auto& next_gauge_d =
      get<::Tags::SpinWeighted<::Tags::TempScalar<15, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& gauge_omega =
      get<::Tags::SpinWeighted<::Tags::TempScalar<16, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& x_step =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<17, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& y_step =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<18, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& z_step =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<19, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));

  double max_error = 1.0;
  size_t number_of_steps = 0;
  while (true) {
    GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                            cartesian_cauchy_coordinates);

    Spectral::Swsh::SwshInterpolator iteration_interpolator{
      get<0>(*angular_cauchy_coordinates),
      get<1>(*angular_cauchy_coordinates), l_max};

    GaugeUpdateJacobianFromCoordinates<
        Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                            make_not_null(&gauge_d),
                                            angular_cauchy_coordinates,
                                            *cartesian_cauchy_coordinates,
                                            l_max);

    get(gauge_omega).data() =
        0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                   get(gauge_c).data() * conj(get(gauge_c).data()));

    // check completion conditions
    max_error = max(abs(get(gauge_omega).data() - target_omega.data()));
    Parallel::printf("Debug iterative solve: %e\n", max_error);
    ++number_of_steps;
    if (max_error > 5.0e-3) {
      ERROR(
          "Iterative solve for surface coordinates of initial data failed. The "
          "strain is too large to be fully eliminated by a well-behaved "
          "alteration of the spherical mesh. For this data, please use an "
          "alternative initial data generator such as "
          "`InitializeJConformalFactor`.");
    }
    if (max_error < tolerance or number_of_steps > max_steps) {
      break;
    }
    // The alteration in each of the spin-weighted Jacobian factors determined
    // by linearizing the system in small J
    get(next_gauge_c).data() =
        -(target_omega.data() - get(gauge_omega).data()) * get(gauge_c).data();
    get(next_gauge_d).data() = get(gauge_omega).data() *
                               (target_omega.data() - get(gauge_omega).data()) /
                               get(gauge_d).data();

    iteration_interpolator.interpolate(make_not_null(&evolution_gauge_eth_x),
                                       eth_x);
    iteration_interpolator.interpolate(make_not_null(&evolution_gauge_eth_y),
                                       eth_y);
    iteration_interpolator.interpolate(make_not_null(&evolution_gauge_eth_z),
                                       eth_z);

    evolution_gauge_eth_x =
        0.5 * ((get(next_gauge_c)) * conj(evolution_gauge_eth_x) +
               conj((get(next_gauge_d))) * evolution_gauge_eth_x);
    evolution_gauge_eth_y =
        0.5 * ((get(next_gauge_c)) * conj(evolution_gauge_eth_y) +
               conj((get(next_gauge_d))) * evolution_gauge_eth_y);
    evolution_gauge_eth_z =
        0.5 * ((get(next_gauge_c)) * conj(evolution_gauge_eth_z) +
               conj((get(next_gauge_d))) * evolution_gauge_eth_z);

    // here we attempt to just update the current value according to the
    // alteration suggested. Ideally that's the `dominant` part of the needed
    // alteration
    Spectral::Swsh::angular_derivatives<tmpl::list<
        Spectral::Swsh::Tags::InverseEth, Spectral::Swsh::Tags::InverseEth,
        Spectral::Swsh::Tags::InverseEth>>(
        l_max, 1, make_not_null(&x_step), make_not_null(&y_step),
        make_not_null(&z_step), evolution_gauge_eth_x, evolution_gauge_eth_y,
        evolution_gauge_eth_z);

    get<0>(*cartesian_cauchy_coordinates) += real(x_step.data());
    get<1>(*cartesian_cauchy_coordinates) += real(y_step.data());
    get<2>(*cartesian_cauchy_coordinates) += real(z_step.data());
  }

  GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                          cartesian_cauchy_coordinates);

  if (adjust_volume_gauge) {
    GaugeUpdateJacobianFromCoordinates<
        Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                            make_not_null(&gauge_d),
                                            angular_cauchy_coordinates,
                                            *cartesian_cauchy_coordinates,
                                            l_max);

    get(gauge_omega).data() =
        0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                   get(gauge_c).data() * conj(get(gauge_c).data()));

    GaugeAdjustInitialJ::apply(volume_j, gauge_c, gauge_d, gauge_omega,
                               *angular_cauchy_coordinates, l_max);
  }
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

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJCoordinatesForVolumeValue.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ScriPlusValues.hpp"
#include "Evolution/Systems/Cce/System.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Given initial boundary data for \f$J\f$ and \f$\partial_r J\f$,
 * computes the initial hypersurface quantities \f$J\f$ and gauge values.
 *
 * \details This action is to be called after boundary data has been received,
 * but before the time-stepping evolution loop. So, it should be either late in
 * an initialization phase or early (before a `Actions::Goto` loop or similar)
 * in the `Evolve` phase.
 *
 * Internally, this dispatches to the call function of
 * `Tags::InitializeJ`, which designates a hypersurface initial data generator
 * chosen by input file options, `InitializeGauge`, and
 * `InitializeScriPlusValue<Tags::InertialRetardedTime>` to perform the
 * computations. Refer to the documentation for those mutators for mathematical
 * details.
 */
template <typename RunStage>
struct InitializeFirstHypersurface;

template <>
struct InitializeFirstHypersurface<InitializationRun> {
  using const_global_cache_tags = tmpl::list<Tags::InitializeJ>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate_apply<InitializeJ::InitializeJ::mutate_tags,
                     InitializeJ::InitializeJ::argument_tags>(
        db::get<Tags::InitializeJ>(box), make_not_null(&box));
    db::mutate_apply<InitializeScriPlusValue<Tags::InertialRetardedTime>>(
        make_not_null(&box),
        db::get<::Tags::TimeStepId>(box).substep_time().value());
    return {std::move(box)};
  }
};

template <>
struct InitializeFirstHypersurface<MainRun> {
  using inbox_tags = tmpl::list<ReceiveTags::JHypersurfaceData>;

  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<tmpl::list<InboxTags...>,
                                     Cce::ReceiveTags::JHypersurfaceData>> =
          nullptr>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& initialization_j_hypersurface_data =
        tuples::get<ReceiveTags::JHypersurfaceData>(inboxes)[0_st];
    const size_t l_max = get<Spectral::Swsh::Tags::LMaxBase>(box);

    const size_t initialization_l_max =
        get<Tags::LMax<InitializationRun>>(initialization_j_hypersurface_data);
    const size_t initialization_number_of_radial_points =
        get<Tags::NumberOfRadialPoints<InitializationRun>>(
            initialization_j_hypersurface_data);

    // resample and copy the initialization data over to the new variables.
    // need to resample:
    // - J ( volume quantitiy - resample each shell then resample along radial
    // stripes)
    const size_t initialization_number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(initialization_l_max);
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    SpinWeighted<ComplexDataVector, 2> shell_interpolated_j{
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
        initialization_number_of_radial_points};
    SpinWeighted<ComplexDataVector, 2> target_buffer_view;
    const SpinWeighted<ComplexDataVector, 2> source_buffer_view;
    const auto& initialization_j =
        get(get<Tags::BondiJ>(initialization_j_hypersurface_data));
    for (size_t i = 0; i < initialization_number_of_radial_points; ++i) {
      make_const_view(make_not_null(&source_buffer_view), initialization_j,
                      i * initialization_number_of_angular_points,
                      initialization_number_of_angular_points);
      target_buffer_view.set_data_ref(
          shell_interpolated_j.data().data() + i * number_of_angular_points,
          number_of_angular_points);
      Spectral::Swsh::resample(make_not_null(&target_buffer_view),
                               source_buffer_view, l_max, initialization_l_max);
    }

    // Use matrix application to obtain the interpolated data in the new volume
    // data structures
    const size_t number_of_radial_points =
        get<Spectral::Swsh::Tags::NumberOfRadialPointsBase>(box);
    const auto radial_interpolation_matrix =
        Spectral::interpolation_matrix<Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto>(
            initialization_number_of_radial_points,
            Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points));
    const auto mesh = Spectral::Swsh::swsh_volume_mesh_for_radial_operations(
        l_max, number_of_radial_points);
    const auto empty_matrix = Matrix{};
    std::array<std::reference_wrapper<const Matrix>, 3> matrix_array{
        {std::ref(empty_matrix), std::ref(empty_matrix),
         std::ref(radial_interpolation_matrix)}};
    db::mutate<Tags::BondiJ, Tags::CauchyCartesianCoords,
               Tags::InertialRetardedTime>(
        make_not_null(&box),
        [&matrix_array, &mesh, &shell_interpolated_j, &initialization_j,
         &initialization_j_hypersurface_data](
            const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                bondi_j,
            const gsl::not_null<tnsr::i<DataVector, 3>*>
                cartesian_cauchy_coordinates,
            const gsl::not_null<Scalar<DataVector>*>
                inertial_retarded_time) noexcept {
          get(*bondi_j).data() = initialization_j.data();
          *cartesian_cauchy_coordinates =
              tuples::get<Tags::CauchyCartesianCoords>(
                  initialization_j_hypersurface_data);
          *inertial_retarded_time = tuples::get<Tags::InertialRetardedTime>(
              initialization_j_hypersurface_data);
          // Debugging resampling
          // apply_matrices(make_not_null(&get(*bondi_j).data()), matrix_array,
          // shell_interpolated_j.data(), mesh.extents());
        });

    db::mutate_apply<GaugeUpdateAngularFromCartesian<Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>>(make_not_null(&box));
    // db::mutate_apply<InitializeJ::InitializeJ::mutate_tags,
    //                  InitializeJ::InitializeJ::argument_tags>(
    //     InitializeJ::InitializeJCoordinatesForVolumeValue{1.0e-15, 1000_st},
    //     /*InitializeJ::InverseCubic(),*/ make_not_null(&box));
    // db::mutate<Tags::InertialRetardedTime>(
    // make_not_null(&box),
    // [&initialization_j_hypersurface_data](
    // const gsl::not_null<Scalar<DataVector>*> retarded_time) noexcept {
    // get(*retarded_time) = blaze::max(get(get<Tags::InertialRetardedTime>(
    // initialization_j_hypersurface_data)));
    // });
    return {std::move(box)};
  }

  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<tmpl::list<InboxTags...>,
                                         Cce::ReceiveTags::JHypersurfaceData>> =
          nullptr>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "required tag `Cce::ReceiveTags::JHypersurfaceData` is not in the "
        "inbox, so this action should not be executing");
  }

  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<tmpl::list<InboxTags...>,
                                     Cce::ReceiveTags::JHypersurfaceData>> =
          nullptr>
  static bool is_ready(const db::DataBox<DbTags>& /*box*/,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    Parallel::printf("checking readiness : %d\n",
                     tuples::get<ReceiveTags::JHypersurfaceData>(inboxes).count(
                         0_st) == 1_st);
    return tuples::get<ReceiveTags::JHypersurfaceData>(inboxes).count(0_st) ==
           1_st;
  }

  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex,
      Requires<not tmpl::list_contains_v<tmpl::list<InboxTags...>,
                                         Cce::ReceiveTags::JHypersurfaceData>> =
          nullptr>
  static bool is_ready(const db::DataBox<DbTags>& /*box*/,
                       const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    Parallel::printf("not ready, appropriate tag not present\n");
    return false;
  }
};
}  // namespace Actions
}  // namespace Cce

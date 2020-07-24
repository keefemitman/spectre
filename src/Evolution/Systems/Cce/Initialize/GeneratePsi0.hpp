// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/Initialize/GeneratePsi0.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
class ComplexModalVector;
/// \endcond

namespace Cce {
namespace InitializeJ {

struct GeneratePsi0 : InitializeJ {
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Generate Psi0 from J"};

  //WRAPPED_PUPable_decl_template(GeneratePsi0);  // NOLINT
  explicit GeneratePsi0(CkMigrateMessage* /*unused*/) noexcept {}

  GeneratePsi0() = default;

  std::unique_ptr<InitializeJ> get_clone() const noexcept override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexModalVector, 2>>& bondi_j,
      const Scalar<SpinWeighted<ComplexModalVector, 2>>& dr_bondi_j,
      const Scalar<SpinWeighted<ComplexModalVector, 0>>& bondi_r,
      const size_t l_max,
      const size_t number_of_radial_points) const noexcept;

  void pup(PUP::er& /*p*/) noexcept override;
};
}  // namespace InitializeJ
}  // namespace Cce

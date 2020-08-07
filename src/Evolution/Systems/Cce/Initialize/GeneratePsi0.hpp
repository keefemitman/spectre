// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

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
/// \endcond

namespace Cce {
namespace InitializeJ {

struct GeneratePsi0 : InitializeJ {
  struct Files {
    using type = std::vector<std::string>;
    static constexpr OptionString help = {
        "Input worldtube files from Cauchy evolution"};
    static type default_value() noexcept {
        return std::vector<std::string>{""}; }
  };

  struct TargetIndex {
    using type = size_t;
    static constexpr OptionString help = {
        "Index of file in files with target extraction radius"};
    static type default_value() noexcept { return 0.0; }
  };

  struct TargetTime {
    using type = double;
    static constexpr OptionString help = {
        "Initial time for CCE"};
    static type default_value() noexcept { return 0.0; }
  };

  using options = tmpl::list<Files, TargetIndex, TargetTime>;
  static constexpr OptionString help = {
      "Generate Psi0 from J, DrJ, and R"};

  WRAPPED_PUPable_decl_template(GeneratePsi0);  // NOLINT
  explicit GeneratePsi0(CkMigrateMessage* /*unused*/) noexcept {}

  GeneratePsi0(std::vector<std::string> files,
               size_t target_idx,
               double target_time) noexcept;

  GeneratePsi0() = default;

  std::unique_ptr<InitializeJ> get_clone() const noexcept override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points) const noexcept override;

  void pup(PUP::er& /*p*/) noexcept override;

  private:
   std::vector<std::string> files_{""};
   size_t target_idx_ = 0.0;
   double target_time_ = 0.0;
};
}  // namespace InitializeJ
}  // namespace Cce

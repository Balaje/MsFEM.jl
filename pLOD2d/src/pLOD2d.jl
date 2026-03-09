__precompile__()

module pLOD2d

include("./Triangulations.jl");
include("./LegendrePolynomials.jl");
include("./MultiscaleBasis.jl");
include("./StabilizedMultiscaleBasis.jl");
include("./AdditionalCorrections.jl");

using pLOD2d.Triangulations: elements_in_coarse_scale_patch, get_interior
using pLOD2d.LegendrePolynomials: poly_exps, reference_rectangular_matrix, assemble_rectangular_matrix, assemble_legendre_mass_matrix
using pLOD2d.MultiscaleBasis: multiscale_bases, multiscale_lhs, multiscale_rhs
using pLOD2d.StabilizedMultiscaleBasis: stabilized_multiscale_bases
using pLOD2d.AdditionalCorrections: additional_correction_bases, stabilized_additional_correction_bases

export elements_in_coarse_scale_patch, get_interior
export poly_exps, reference_rectangular_matrix, assemble_rectangular_matrix, assemble_legendre_mass_matrix
export multiscale_bases, multiscale_lhs, multiscale_rhs
export stabilized_multiscale_bases
export additional_correction_bases, stabilized_additional_correction_bases

end # module pLOD2d

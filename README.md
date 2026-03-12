# MsFEM.jl

This repository contains code implementing the high-order multiscale finite element method in one and two space dimensions. The goal is to provide a more maintainable, easy-to-run codebase for Cartesian meshes. This repository will eventually be merged into the main repository (MultiscaleFEM.jl). The code uses `Gridap.jl` to define the Cartesian model, FESpaces and the bilinear/linear forms. Below are the instructions for running the code. The same steps can be applied to both 1D and 2D codes. First `cd` into the project directory `(pLOD1d or pLOD2d)`, open the Julian prompt and instantiate the environments:

```julia
using Pkg
Pkg.instantiate()
```

A helper function `parse_command_line()` written using ArgParse.jl is provided to accept the discretization parameters from the command line. The following Julia snippet is used to extract the command line arguments:

```julia
parsed_args = parse_command_line()

n = parsed_args["fine_scale"]
N = parsed_args["coarse_scale"]
p = parsed_args["order"]
l = parsed_args["patch_radius"]
j = parsed_args["correction_level"]
```

Do 

```shell
julia --project=. [SCRIPT] --help
``` 

to get the help environment. Once done setting up, there are three main scripts in the folders:

- `main.jl`: Illustrates how to setup the multiscale problem, defintions of the discrete models, the finite element spaces and the bilinear/linear forms. The code computes the three types of multiscale basis functions available in literature: 1) The pLOD multiscale bases discussed in [(Maier, R. 2021, SIAM Journal on Numerical Analysis 59(2), 1067-1089)](https://epubs.siam.org/doi/10.1137/20M1364321) , 2) The stabilized pLOD basis by [(Dong, Z., Hauck, M., & Maier, R. (2023), SIAM Journal on Numerical Analysis, 61(4), 1918–1937)](https://epubs.siam.org/doi/10.1137/22M153392X) and, 3) The eho-LOD method with additional bases on the kernel space [(Kalyanaraman, B., Krumbiegel, F., Maier, R., & Wang, S. (2025), arXiv [Math.NA])](https://arxiv.org/abs/2510.09514).
- `poisson.jl`: Solution to the Poisson problem using the pLOD and spLOD methods.
- `wave_equation.jl`: Solution to the wave equation using pLOD, spLOD and eho-LOD methods.

## References

1. Maier, R. (2021). A high-order approach to elliptic multiscale problems with general unstructured coefficients. SIAM Journal on Numerical Analysis, 59(2), 1067-1089. 
2. Dong, Z., Hauck, M., & Maier, R. (2023). An improved high-order method for elliptic multiscale problems. SIAM Journal on Numerical Analysis, 61(4), 1918-1937.
3. Kalyanaraman, B., Krumbiegel, F., Maier, R., & Wang, S. (2025). Optimal higher-order convergence rates for parabolic multiscale problems. arXiv preprint arXiv:2510.09514.
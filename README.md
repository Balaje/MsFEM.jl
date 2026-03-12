# MsFEM.jl

This repository contains code implementing the high-order multiscale finite element method in one and two space dimensions. The goal is to provide a more maintainable, easy-to-run codebase for Cartesian meshes. This repository will eventually be merged into the main repository ([MultiscaleFEM.jl](https://github.com/Balaje/MultiScaleFEM.jl)). The code uses `Gridap.jl` to define the Cartesian model, FESpaces and the bilinear/linear forms. Below are the instructions for running the code. The same steps can be applied to both 1D and 2D codes. First `cd` into the project directory `(pLOD1d or pLOD2d)`, open the Julian prompt and instantiate the environments:

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

- `main.jl`: Illustrates how to setup the multiscale problem, defintions of the discrete models, the finite element spaces and the bilinear/linear forms. The code computes the three types of multiscale basis functions available in literature: 1) The p-LOD multiscale bases discussed in [(Maier, R. 2021, SIAM Journal on Numerical Analysis 59(2), 1067-1089)](https://epubs.siam.org/doi/10.1137/20M1364321) , 2) The stabilized p-LOD (sp-LOD) basis by [(Dong, Z., Hauck, M., & Maier, R. (2023), SIAM Journal on Numerical Analysis, 61(4), 1918–1937)](https://epubs.siam.org/doi/10.1137/22M153392X) and, 3) The eho-LOD method with additional bases on the kernel space [(Kalyanaraman, B., Krumbiegel, F., Maier, R., & Wang, S. (2025), arXiv [Math.NA])](https://arxiv.org/abs/2510.09514).
- `poisson.jl`: Solution to the Poisson problem using the p-LOD and sp-LOD methods.
- `wave_equation.jl`: Solution to the wave equation using p-LOD, sp-LOD and eho-LOD methods.

## Explanation

### pLOD vs spLOD

Let us consider the example of solving the following 1D BVP

$$
\begin{align*}
  -\frac{d}{dx}\left( A(x) \frac{du}{dx}  \right) &= f(x), \quad x \in (0,1),\\
  u &= 0, \quad x \in \\{0, 1\\},
\end{align*}
$$

and assume that $A(x)$ oscillates at the scale $\varepsilon \ll 1$. We are interested to solve the multiscale problem using the pLOD and spLOD methods discussed in [Maier, R. 2021, SIAM Journal on Numerical Analysis 59(2), 1067-1089](https://epubs.siam.org/doi/10.1137/20M1364321) and [Dong, Z., Hauck, M., & Maier, R. (2023), SIAM Journal on Numerical Analysis, 61(4), 1918–1937](https://epubs.siam.org/doi/10.1137/22M153392X), respectively. Run 

```shell
cd pLOD1d/
julia --project=. poisson.jl -n 2048 -N 128 -p 1 -l 2
julia --project=. poisson.jl -n 2048 -N 128 -p 1 -l 128
```

Along with some progress bars, we obtain the lines:

```shell
2048     128     1       2       0.03262799256083895     0.14161089692859816     5.484084832881531e-6    0.0016891869071974756
2048     128     1       128     4.7286214920445696e-11          2.6309649981017746e-8   4.728610825337349e-11   2.630965187866267e-8
```

Here we obtain:

$n$ | $N$ | $p$ | $l$ | $L^2$-Error obtained using pLOD | Energy Error obtained using pLOD | $L^2$-Error obtained using spLOD | Energy Error obtained using spLOD |
--- | --- | --- | --- | --- | --- | --- | --- |
2048  |   128  |   1  |    2  |     0.03262799256083895  |   0.14161089692859816  |   5.484084832881531e-6  |  0.0016891869071974756  |
2048  |   128  |   1  |     128  |   4.7286214920445696e-11    |      2.6309649981017746e-8  |   4.728610825337349e-11 |  2.630965187866267e-8 |

As proposed by [Dong, Z., Hauck, M., & Maier, R. (2023), SIAM Journal on Numerical Analysis, 61(4), 1918–1937](https://epubs.siam.org/doi/10.1137/22M153392X), we see that the solution obtained using the sp-LOD method is orders of magnitude better than the p-LOD method when patch radius $l=2$, and they are equal (close to machine precision) when $l=\infty$.

### sp-LOD vs eho-LOD

Consider the 1D initial boundary value problem

$$
\begin{align*}
  \frac{\partial u}{\partial t} - \frac{\partial}{\partial x}\left( A(x) \frac{\partial u}{\partial x}  \right) &= f(x, t), \quad x \in (0,1), \\; t > 0,\\
  u &= 0, \quad x \in \\{0, 1\\}, \\; t > 0,\\
  u(x,0) &= 0, \quad x \in (0,1),\\
  \frac{\partial u}{\partial t}(x,0) &= 0, \quad x \in (0,1),
\end{align*}
$$

and assume that $A(x)$ oscillates at the scale $\varepsilon \ll 1$. We are interested in demonstrating the effects of the additional correction bases of the eho-LOD method proposed in [Kalyanaraman, B., Krumbiegel, F., Maier, R., & Wang, S. (2025), arXiv [Math.NA]](https://arxiv.org/abs/2510.09514). Run the following commands:

```shell
julia --project=. wave_equation.jl -n 2048 -N 16 -p 1 -l 5 -j 0
julia --project=. wave_equation.jl -n 2048 -N 16 -p 1 -l 5 -j 1
julia --project=. wave_equation.jl -n 2048 -N 16 -p 1 -l 16 -j 0
julia --project=. wave_equation.jl -n 2048 -N 16 -p 1 -l 16 -j 1
```

Again, we obtain the following errors:

```shell
2048     16      1       5       0       6.270890391951073e-7    4.860719871904017e-5    3.527773277785003e-7    2.9554075552267894e-5
2048     16      1       5       1       1.8047187891948405e-7   2.5893105224818887e-5   2.106973276766412e-8    2.274600850145292e-6
2048     16      1       16      0       3.548515608556209e-7    2.9649971531781995e-5   3.5485156085457956e-7   2.9649971531519415e-5
2048     16      1       16      1       1.9632782723751118e-8   2.0755202874592446e-6   1.9632782710665524e-8   2.0755202873066648e-6
```

$n$ | $N$ | $p$ | $l$ | $j$ | $L^2$-Error obtained using pLOD | Energy Error obtained using pLOD | $L^2$-Error obtained using spLOD | Energy Error obtained using spLOD |
--- | --- | --- | --- | --- | --- | --- | --- | --- |
2048  |   16  |    1   |    5   |    0  |   6.270890391951073e-7  |  4.860719871904017e-5    |   3.527773277785003e-7   |  2.9554075552267894e-5 |
2048  |   16  |    1   |    5   |    1  |   1.8047187891948405e-7 |  2.5893105224818887e-5   |   2.106973276766412e-8   |  2.274600850145292e-6 |
2048  |   16  |    1   |    16  |    0  |   3.548515608556209e-7  |   2.9649971531781995e-5  |   3.5485156085457956e-7  |  2.9649971531519415e-5 |
2048  |   16  |    1   |    16  |    1  |   1.9632782723751118e-8 |   2.0755202874592446e-6  |   1.9632782710665524e-8  |  2.0755202873066648e-6  |

As we see, the one-level $(j=1)$ additional correction bases from the eho-LOD method pushes the error obtained in the p-LOD/sp-LOD method down by an order of magnitude.

## References

1. Maier, R. (2021). A high-order approach to elliptic multiscale problems with general unstructured coefficients. SIAM Journal on Numerical Analysis, 59(2), 1067-1089. 
2. Dong, Z., Hauck, M., & Maier, R. (2023). An improved high-order method for elliptic multiscale problems. SIAM Journal on Numerical Analysis, 61(4), 1918-1937.
3. Kalyanaraman, B., Krumbiegel, F., Maier, R., & Wang, S. (2025). Optimal higher-order convergence rates for parabolic multiscale problems. arXiv preprint arXiv:2510.09514.

# heat_flow_into_sphere
FEniCSx implementation of some numerical methods for the heat flow into the sphere.

## Governing Equations
The governing equations in a general form can be stated as:

$$
\begin{aligned}
\partial_t d + \gamma  (I - d \otimes d)  q  & = 0 , \\
\vert d \vert & = 1 ,
\end{aligned}
$$


The governing equations are equipped with an elastic energy term, as well as a coupling with a magnetic field,

$$
E = E_{\mathrm{ela}} + E_{ H},
\qquad E_{ H} = - \frac{\chi_{\perp}}{2} \vert  d \times  H \vert^2 - \frac{\chi_{\Vert}}{2} ( d \cdot  H)^2 \, .
$$

As elastic energy, we consider the Oseen-Frank energy given by

$$
E_{\mathrm{ela}} = \int_{\Omega} \frac{{K}_1}{2} \vert \nabla  d \vert ^2 + \frac{{K}_2}{2} (\nabla \cdot   d)^2 + \frac{{K}_3}{2} \vert \nabla \times  d \vert ^2 + \frac{{K}_4}{2} ( d \cdot \nabla \times  d)^2 + \frac{{K}_5}{2} \vert  d \times \nabla \times  d \vert ^2 \mathrm{dx} .
$$

Accordingly, the variational derivative $q$ is defined by

$$
\begin{aligned}
\int_{\Omega} q \cdot \phi \mathrm{dx} \mathrm{d} \tau = & \int_0^T \int_{\Omega} {K}_1 \nabla  d : \nabla \phi + {K}_2 (\nabla \cdot   d) (\nabla \cdot   \phi) + {K}_3 (\nabla \times  d) \cdot (\nabla \times  \phi) \mathrm{dx} \\
& + \int_{\Omega} {K}_4 ( d \cdot \nabla \times  d) ( d \cdot \nabla \times  \phi + \phi \cdot \nabla \times  d) \mathrm{dx} \\ 
& + \int_{\Omega} {K}_5 ( d \times \nabla \times  d ) \cdot ( \phi \times \nabla \times  d  +  d \times \nabla \times \phi) \mathrm{dx} .
\end{aligned}
$$

In the case of the Dirichlet energy, 

$$
E = E_{\mathrm{ela}} = \int_{\Omega} \frac{1}{2} \vert \nabla  d \vert ^2 ,
$$


the variational derivative can be simplified,

$$
\int_{\Omega} q \cdot \phi \mathrm{dx} \mathrm{d} \tau = 
\int_0^T \int_{\Omega} \nabla  d : \nabla \phi \mathrm{dx},
\qquad \implies \qquad q = - \Delta d ,
$$

and so the partial differential equation (also referred to as harmonic heat flow into the sphere),

$$
\begin{aligned}
\partial_t d - \gamma  (I - d \otimes d)  \Delta d  & = 0 , \\
\vert d \vert & = 1 .
\end{aligned}
$$

Equivalently this is often written as

$$
\begin{aligned}
\partial_t d - \gamma \Delta d  & = \gamma \vert \nabla d \vert^2 d , \\
\vert d \vert & = 1 ,
\end{aligned}
$$

or

$$
\begin{aligned}
\partial_t d + \gamma  d \times (d \times \Delta d)  & = 0 , \\
\vert d \vert & = 1 .
\end{aligned}
$$

## Numerical Methods

Currently the following numerical schemes are available (key : explanation).
- nonlin_cg : Numerical Method in [[1]](#1) (setting the velocity to zero in time and space). Implemented using a monolithic Newton solver.
- linear_cg : Algorithm 1 in [[3]](#3) (setting the velocity to zero in time and space). Linear projection method.
- linear_dg : Algorithm 2 in [[3]](#3) (setting the velocity to zero in time and space). Linear projection method based on the DG method.
- fp_decoupled : Numerical Method in [[1]](#1) (setting the velocity to zero in time and space) with an iterative Picard-type linearization, see [[2]](#2).
- fp_coupled : Numerical Method in [[1]](#1) (setting the velocity to zero in time and space) with an iterative Picard-type linearization similar to fp_decoupled.


They can be selected by specifying the command line key "-m" or "--mod".

## Numerical Settings

- spiral : spiral domain with a known stationary solution, see [[3,5]](#5).
- smooth : smooth initial condition on a unit square, see [[4]](#4).
- annihilation : two line defects in a unit-cube, see [[1]](#1).

They can be selected by specifying the command line key "-e" or "--exp".

## Getting Started and Usage

All arguments to run simulations are given via the command line input. To see the options run the following command in the package directory:

```
python -m sim.run -h
```

Another usage example with several arguments:

```
python -m sim.run -m linear_cg -e spiral -vtx 1 -dt 0.01 -sid "spiral-experiment" -T 0.05
```

Presets for several simulations are given in the folder 'sim/sim_presets/' usually in the form of a bash or python file. Examples for usage:
```
sim/sim_presets/unit.sh
```

Physical parameters of the governing equations and the energy term can be changed via command line arguments, i.e.
```
python -m sim.run -m linear_cg -e spiral -gamma 2.0 -K1 1.0 -K2 0.5 -K3 0.0 -K4 0.1 -K5 0.1 -chi_vert -1.0 -chi_perp -0.5
```

Simulations can also be run from an existing config file:
```
python -m sim.runconfig "output/unit1/config.json"
```

## Requirements

All requirements can be found in the file requirements.txt and can be installed via pip by

```
pip install -r requirements.txt
```

or via conda by

```
conda create --name my-env-name --file requirements.txt -c conda-forge
```

## Notes

- Some experiments need a predefined mesh in .msh or .xdmf format in the folder 'input/meshes'. These are either provided or there are files to generate meshes in the folder 'input'.
- Results are automatically named and saved in the folder 'output/'

## References to relevant publications

<a id="1">[1]</a> 
Lasarzik, R., & Reiter, M. E. V. (2023). Analysis and numerical approximation of energy-variational solutions to the Ericksen-Leslie equations. Acta Appl. Math., 184, 44. https://doi.org/10.1007/s10440-023-00563-9

<a id="1">[2]</a> 
Reiter, M. E. V. (2023). Decoupling and Linearization of a Liquid Crystal Model fulfilling a Unit Norm Constraint. Proceedings of ECMI 2023.

<a id="1">[3]</a> 
Reiter, M. E. V. (2025). Projection Methods in the Context of Nematic Crystal Flow. https://arxiv.org/abs/2502.08571 & https://www.doi.org/10.1093/imanum/drag013 

<!-- <a id="1">[4]</a> 
Becker, R., Feng, X., & Prohl, A. (2008). Finite element approximations of the Ericksen-Leslie model for nematic liquid crystal flow. SIAM Journal on Numerical Analysis, 46(4), 1704-1731. doi:10.1137/07068254X

<a id="1">[5]</a> 
Badia, S., Guillen-González, F., & Gutiérrez-Santacreu, J. (2011). Finite element approximation of nematic liquid crystal flows using a saddle-point structure. J. Comput. Phys., 230(4), 1686-1706. -->

## Authors

* **Maximilian E. V. Reiter**, https://orcid.org/0000-0001-9137-7978

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
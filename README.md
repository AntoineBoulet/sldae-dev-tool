# SLDAE DEV TOOL
Set of routines in C to calculate and interpolate automatically SLDAE parameters from quasi-particle properties [[Phys. Rev. A **106**, 013306 (2022)](https://doi.org/10.1103/PhysRevA.106.013306)].
The code is integrated with the [W-SLDA Toolkit](https://wslda.fizyka.pw.edu.pl/) which solve self-consistent equations of mathematical problems which have structure formally equivalent to Bogoliubov-de Gennes equations.


#### Requirement
 - [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/)
 - [GNU Compiler Collection (GCC)](https://gcc.gnu.org/)

#### Usage
 > gcc sldae-dev-tool.c -lm -lgsl -lgslcblas

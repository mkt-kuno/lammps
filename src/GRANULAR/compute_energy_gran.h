/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/*~ This compute gathers energy terms. The full listing of energy terms
  which can be written out is as follows (name of LAMMPS input in 
  parentheses):
  
  Energy dissipated by friction ('friction')

  Energy dissipated by local damping ('local_damping')
  Energy dissipated by viscous damping ('viscous_damping')
  Total energy dissipated by damping ('damping')

  Energy added/removed by boundary work ('boundary')

  Rotational kinetic energy ('rotational_kinetic')
  Translational kinetic energy ('translational_kinetic')
  Total kinetic energy ('kinetic')

  Normal spring contribution to strain energy ('normal_strain')
  Shear spring contribution to strain energy ('shear_strain')
  Total strain energy ('strain')

  spin spring contribution to spin energy ('spin_energy')
  only if D_spin function is called

  [KH - 12 March 2014]   "spin_energy" was added [MO - 15 November 2014]
*/

#ifdef COMPUTE_CLASS

ComputeStyle(energy/gran,ComputeEnergyGran)

#else

#ifndef LMP_COMPUTE_ENERGY_GRAN_H
#define LMP_COMPUTE_ENERGY_GRAN_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeEnergyGran : public Compute {
 public:
  ComputeEnergyGran(class LAMMPS *, int, char **);
  ~ComputeEnergyGran();
  void init();
  void compute_vector();
  void compute_peratom();
  double memory_usage();

  int pairenergy; //~ 0 if energy not tracked in pairstyle; else 1

 private:
  int nmax, dim;
  int length_enum; //~ The length of an enumeration
  int wallactive; //~ The number of walls that are active
  int wallcheck; //~ 1 if the check for walls has been done
  int dampactive[2]; //~ The IDs of fix damp/local and fix viscous
  int dampcheck[2]; //~ 1 if the check for local/viscous damping done
  int *inputs;
  double *evector, **earray;
  double kinetic[3];
  class Fix *deffix;

  //~ List of helper functions to obtain and prepare the data
  double pair_extract(const char *);
  double kinetic_extract(int);
  double damping_extract(const char *, int);
  void add_fix_energy_boundary();
  inline double peratom_rke(double, double, double, double);
  inline double peratom_tke(double, double, double);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

W: More than one compute energy/gran

It is not efficient to use compute energy/gran more than once.

*/

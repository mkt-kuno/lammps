/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(viscous,FixViscous)

#else

#ifndef LMP_FIX_VISCOUS_H
#define LMP_FIX_VISCOUS_H

#include "fix.h"

namespace LAMMPS_NS {

class FixViscous : public Fix {
 public:
  FixViscous(class LAMMPS *, int, char **);
  virtual ~FixViscous();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  void *extract(const char *, int &); //~ To fetch energy [KH - 9 April 2014]
  void write_restart(FILE *); //~ To store energy in restart file
  void restart(char *);

 protected:
  double *gamma;
  int ilevel_respa;

  int energy_calc; //~ 0 if energy not computed; else 1 [KH - 9 April 2014]
  double energy_dissip; //~ Energy dissipated by damping
  double dissipated_energy(double); //~ Function to calculate dissipation
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/

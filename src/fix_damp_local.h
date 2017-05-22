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

FixStyle(damp/local,FixDampLocal)

#else

#ifndef LMP_FIX_DAMP_LOCAL_H
#define LMP_FIX_DAMP_LOCAL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixDampLocal : public Fix {
 public:
  FixDampLocal(class LAMMPS *, int, char **);
  virtual ~FixDampLocal();
  int setmask();
  void init();
  void setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  int modify_param(int, char **); // added in so as to be able to reset parameters
  void *extract(const char *, int &); //~ To fetch energy [KH - 9 April 2014]
  void write_restart(FILE *); //~ To store energy in restart file
  void restart(char *);

 protected:
  double alpha;
  int ilevel_respa;

  double signofnum(double);

  int energy_calc; //~ 0 if energy not computed; else 1 [KH - 9 April 2014]
  double energy_dissip; //~ Energy dissipated by damping
  double dissipated_energy(double, int, double *, double *); //~ Function to calculate dissipation
};

}

#endif
#endif

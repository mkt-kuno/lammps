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

FixStyle(energy/boundary,FixEnergyBoundary)

#else

#ifndef LMP_FIX_ENERGY_BOUNDARY_H
#define LMP_FIX_ENERGY_BOUNDARY_H

#include "fix.h"

namespace LAMMPS_NS {

class FixEnergyBoundary : public Fix {
 public:
  FixEnergyBoundary(class LAMMPS *, int, char **);
  ~FixEnergyBoundary();
  int setmask();
  void setup(int);
  void end_of_step();
  void *extract(const char *, int &);
  void write_restart(FILE *);
  void restart(char *);

 protected:
  int sfound, pb, wallactive;
  int wiggle,wtranslate,wscontrol; //added [MO - 22 Aug 2015] 
  double boundary_work, deltawv, deltawd;
  double oldierates[6];
  class Compute *stressatom;
  class Fix *deffix;
};

}

#endif
#endif

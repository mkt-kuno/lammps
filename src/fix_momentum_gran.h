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

/*~ Added this fix to LAMMPS as fix_momentum does something strange
  for zeroing rotations with granular materials. This fix zeroes the 
  angular velocities immediately before the force computation and
  zeroes the torques immediately after the force computation, but
  before final_integrate is run [KH - 2 July 2012]*/

#ifdef FIX_CLASS

FixStyle(momentum/gran,FixMomentumGran)

#else

#ifndef LMP_FIX_MOMENTUM_GRAN_H
#define LMP_FIX_MOMENTUM_GRAN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixMomentumGran : public Fix {
 public:
  FixMomentumGran(class LAMMPS *, int, char **);
  int setmask();
  void pre_force(int);
  void post_force(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: FixMomentumGran requires that atoms have torque

The post-force function requires the atoms to store torque.

*/

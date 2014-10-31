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

/*~ This has been added to apply a drag force to the particles. The file
  is based on fix_gravity.

  The input syntax is the same as for fix_gravity, except two extra
  parameters are required: the hydraulic gradient and the computation
  method. Thus the order is as follows:

  fix ID group fluiddrag {hydraulic gradient} {computation_method} style magnitude args

  The computation method is an integer with two possible values:
  0 - subtract the velocity of the largest particle from the velocities of all
  particles in the system
  1 - do not perform this velocity adjustment

  [KH - 7 December 2012]*/

#ifdef FIX_CLASS

FixStyle(fluiddrag,FixFluidDrag)

#else

#ifndef LMP_FIX_FLUIDDRAG_H
#define LMP_FIX_FLUIDDRAG_H

#include "fix.h"

namespace LAMMPS_NS {

class FixFluidDrag : public Fix {
 public:
  FixFluidDrag(class LAMMPS *, int, char **);
  ~FixFluidDrag();
  int setmask();
  void init();
  void post_force(int);
  void post_force_respa(int, int, int);
  void end_of_step();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

 private:
  int style;
  double hydgrad;
  double phi,theta,phigrad,thetagrad;
  double xdir,ydir,zdir;
  double xgrav,ygrav,zgrav;
  double degree2rad,PI;
  int nlevels_respa;
  int time_origin;

  int me; //~ The processor ID
  double specificweight,pgradient,totalpvolume,totaldeff,overallmaxrad;
  double **fddata; //~ For gathering data
  int compmethod; //~ The computational method as described above
  int procandtag[2]; //~ The proc and tag of the largest atom
  int numcols; //~ The number of data columns to store
  int dimension; //~ 2 or 3 dimensional simulation
};

}

#endif
#endif

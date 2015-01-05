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

#ifdef COMPUTE_CLASS

ComputeStyle(laguerre/atom,ComputeLaguerre)

#else

#ifndef LMP_COMPUTE_LAGUERRE_H
#define LMP_COMPUTE_LAGUERRE_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeLaguerre : public Compute {
 public:
  ComputeLaguerre(class LAMMPS *, int, char **);
  ~ComputeLaguerre();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int nmax;
  double **voro;
  double xleft,xright,yleft,yright,zleft,zright; // coordinates of the container for the Laguerre diagram
  int memflag,verflag,volflag; //flags that are set to 1 if the membrane areas, vertices or Laguerre volumes are to be output
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

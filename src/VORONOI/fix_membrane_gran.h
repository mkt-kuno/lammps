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

FixStyle(membrane/gran,FixMembraneGran)

#else

#ifndef LMP_FIX_MEMBRANE_GRAN_H
#define LMP_FIX_MEMBRANE_GRAN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixMembraneGran : public Fix {
 public:
  FixMembraneGran(class LAMMPS *, int, char **);
  virtual ~FixMembraneGran();
  int setmask();
  void init();
  void setup(int);
  virtual void post_force(int);
  virtual void post_force_respa(int, int, int);

  double memory_usage();
  int size_restart(int);
  int maxsize_restart();
  void reset_dt();
  int modify_param(int, char **);
  double compute_vector(int);

 protected:
  double dt;
  //double targetf,gain;
  //int ftvarying; // 1 if ftarget set through a variable
  char *pstr;
  int pvar;
  int nlevels_respa;
  int time_origin;
  double xleft,xright,yleft,yright,zleft,zright; // coordinates of the container for the Laguerre diagram
  double pressurex,pressurey,pressurez; // the mebrane pressure for the x, y, z sides (inwards positive)
  int xflag,yflag,zflag; // flags that are 1 if a membrane exists in the x,y,z sides respectively
  double distx,disty,distz; // how far from the xleft xright etc a Laguerre diagram edge can be - useful
  // for parallelisation where some grains might appear to be at the edge because they are close to the
  // edge of their cells instead of the sample edges

  int shearupdate;

  void ev_tally_membrane(int, double, double, double, double, double, double, double);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix membrane/gran requires atom style sphere

Self-explanatory.

E: Cannot use wall in periodic dimension

Self-explanatory.

E: Cannot wiggle and shear fix membrane/gran

Cannot specify both options at the same time.

E: Invalid wiggle direction for fix membrane/gran

Self-explanatory.

E: Invalid shear direction for fix membrane/gran

Self-explanatory.

E: Fix membrane/gran is incompatible with Pair style

Must use a granular pair style to define the parameters needed for
this fix.

*/

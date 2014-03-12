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

#ifdef FIX_CLASS

FixStyle(wall/gran,FixWallGran)

#else

#ifndef LMP_FIX_WALL_GRAN_H
#define LMP_FIX_WALL_GRAN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWallGran : public Fix {
 public:
  FixWallGran(class LAMMPS *, int, char **);
  virtual ~FixWallGran();
  int setmask();
  void init();
  void setup(int);
  virtual void post_force(int);
  virtual void post_force_respa(int, int, int);

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  void set_arrays(int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  int pack_restart(int, double *);
  void unpack_restart(int, int);
  int size_restart(int);
  int maxsize_restart();
  void reset_dt();
  int modify_param(int, char **);
  double compute_vector(int);
  void *extract(const char *, int &); //~ [KH - 20 February 2014]
  void write_restart(FILE *); //~ [KH - 20 February 2014]
  void restart(char *); //~ [KH - 20 February 2014]

 protected:
  int wallstyle,pairstyle,wiggle,wshear,axis,dampflag;
  int wtranslate,wscontrol; //flags for wall movement and wall stress control respectively
  double kn,kt,gamman,gammat,xmu;
  double lo,hi,cylradius;
  double loINI,hiINI; // for wiggle only
  double velwall[3],fwall[3],fwall_all[3];
  double amplitude,period,omega,vshear;
  double dt;
  double targetf,gain;
  char *fstr;
  int fvar;
  int ftvarying; // 1 if ftarget set through a variable
  int nlevels_respa;
  int time_origin;
  int numshearquants; //~ The number of shear quantities [KH - 30 October 2013]
  int *rolling,*model_type; //~ Quantities for rolling resistance model [KH - 30 October 2013]
  double *rolling_delta;
  int lastwarning[2]; //~ Used to control frequencies at which warnings about failures to calculate contact stiffnesses are output in the rolling resistance model [KH - 6 November 2013]

  //~ Add quantities for tracing global energy [KH - 20 February 2014]
  int pairenergy, *trace_energy;;
  double dissipfriction, normalstrain, shearstrain;

  /*~ Used for accessing fix old_omega when rolling resistance model
    is active [KH - 30 October 2013]*/
  class Fix *deffix;
  
  int *touch;
  double **shear;
  int shearupdate;

  void hooke(double, double, double, double, double *,
             double *, double *, double *, double *, double, double, int);
  void hooke_history(double, double, double, double, double *,
                     double *, double *, double *, double *, double, double,
                     double *, int);
  void hertz_history(double, double, double, double, double *,
                     double *, double *, double *, double *, double, double,
                     double *, int);
  void shm_history(double, double, double, double, double *,
		   double *, double *, double *, double *, double, double,
		   double *, int); //~ [KH - 30 October 2013]
  void move_wall();
  void velscontrol();
  void ev_tally_wall(int, double, double, double, double, double, double, double);
  void rolling_resistance(int, int, double, double, double, double, double, double, double, double, double *, double *); //~ Added this function [KH - 30 October 2013]
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix wall/gran requires atom style sphere

Self-explanatory.

E: Cannot use wall in periodic dimension

Self-explanatory.

E: Cannot wiggle and shear fix wall/gran

Cannot specify both options at the same time.

E: Invalid wiggle direction for fix wall/gran

Self-explanatory.

E: Invalid shear direction for fix wall/gran

Self-explanatory.

E: Fix wall/gran is incompatible with Pair style

Must use a granular pair style to define the parameters needed for
this fix.

*/

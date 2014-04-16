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

FixStyle(multistress,FixMultistress)

#else

#ifndef LMP_FIX_MULTISTRESS_H
#define LMP_FIX_MULTISTRESS_H

#include "fix.h"

namespace LAMMPS_NS {

  class FixMultistress : public Fix {
    friend class FixCrushing;

  public:
    FixMultistress(class LAMMPS *, int, char **);
    ~FixMultistress();
    int setmask();
    void init();
    void pre_exchange();
    void pre_force(int);
    void calc_ctrl_params();
    void create_fix();
    void end_of_step();
    void eval_fix_deform_params_basic();
    void eval_fix_deform_params_special();
    void eval_fix_deform_params_linkvolstress();
    void eval_fix_deform_params_constantpq();
    inline void linkvolstress_loop(double [], int, int, double);
    inline void constantpq_loop(double [], int, int);
    void update_fix_deform_params();
    void instability_test(int);
    double *param_export();
    double compute_vector(int);
    void lost_atom_check();
    void write_restart(FILE *);
    void restart(char *);

  private:
    int me; //~ Denotes the processor rank
    int dimension,triclinic;
    int numchar; //~ The number of strings to store in tnewarg
    int flip; //~ Flag corresponding to that of the same name in fix_deform
    int currstep; //~ The number of times fix_deform was updated
    int stabtestflag[6]; //~ Flag to begin the stability test
    int instabcheck; //~ 1 to run the check for instability, otherwise 0
    int scaleflag; //~ 0 for box units, 1 (the default) for lattice units
    int lvstressflag; //~ 1 if linkvolstress is active, otherwise 0
    int constanteflag; //~ Added for constant void ratio with fix crushing
    int constpflag; //~ 1 if constantp is active, otherwise 0
    int constqflag; //~ 1 if constantq is active, otherwise 0
    int maxrateflag; //~ 1 if a maximum strain rate is defined, otherwise 0
    int extrasteps; //~ The number of additional steps to run
    int ncyclicsteps; //~ Number of steps for which cyclic loading is active
    int iterateflag; //~ May be needed for conditional code branches
    double tolerance; //~ The percentage accuracy required for the stresses
    double initialmaxvel[6]; //~ The initial maximum permissible boundary speeds
    double maxrate[6]; //~ The optional user-defined maximum strain rate (x 6)
    double oldmeans[6]; //~ The mean stresses on the preceding timestep
    double *means; //~ The mean stresses on the current timestep
    double tallymeans[6]; //~ The total mean stresses from all processors
    double starget[6]; //~ The target stresses input by the user
    double initialvolume; //~ The initial volume bounded by the periodic space
    double pboundstart[6]; //~ The initial positions of the cell boundaries
    double meaneffectivestress; //~ The initial value of p'
    double deviatorstress; //~ The initial value of q
    double temprates[6]; //~ Strain rates prior to the imposition of maxrate restrictions

    int strflag[6]; //~ Flags to indicate which dimensions are servo-controlled
    int defflag[6]; //~ Flags to indicate which dimensions are strain-controlled
    int deltflag[6]; //~ Flags to indicate which dimensions use delta in the calculation of the strain rate
    int ictrlflag[6]; //~ Flags to indicate that cumulative error terms should start to be summed
    int cyclicflag[6]; //~ Flags to indicate whether or not cyclic stress control is active on a boundary: 0 if not active; 1 if cyclicmean; 2 if cyclicdeviator
    int linkvolstress[3]; //~ Flags to indicate that the volume is maintained constant and the stresses in two specified directions are kept equal
    int constantpq[3]; //~ Flags to indicate that either the mean effective stress or the deviator stress is maintained constant and the stresses in two specified directions are kept equal
    int constbflag[3]; //~ Flags to indicate that the b value is maintained constant for a boundary
    int constructorflag; //~ Flag to denote whether the constructor has been run on this timestep (1) or not (0)
    int cstressflag; //~ Flag defined for convenience to indicate whether cyclic stress loading is active
    int constbctrl; //~ Flag indicating whether constantb is active (1) or not (0)

    double erates[6]; //~ Engineering strain rates
    double Kp[6]; //~ The proportional gains
    double ti[6]; //~ The integral times
    double td[6]; //~ The derivative times
    double cumul[6]; //~ The accumulated integral terms

    double instability[6][4]; //~ Array used for checking for instability
    double cyclicparam[3][6]; //~ Array used to store cyclic loading parameters

    char *id_stress,*id_multistress;
    class Compute *tstress;
    class Fix *deffix;
  };
}

#endif
#endif

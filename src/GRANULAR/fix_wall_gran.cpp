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

/* ----------------------------------------------------------------------
   Contributing authors: Leo Silbert (SNL), Gary Grest (SNL),
                         Dan Bolintineanu (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix_wall_gran.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "modify.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "input.h"
#include "variable.h"
#include "error.h"
#include "math_extra.h" //~ These five header files needed for rolling resistance model [KH - 5 November 2013]
#include "fix_old_omega.h"
#include "math_special.h"
#include "comm.h"
#include <mpi.h>
//~ Added compute header files for energy tracing [KH - 20 February 2014]
#include "compute.h"
#include "compute_energy_gran.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

// XYZ PLANE need to be 0,1,2

enum{XPLANE=0,YPLANE=1,ZPLANE=2,ZCYLINDER,REGION};
enum{HOOKE,HOOKE_HISTORY,HERTZ_HISTORY,BONDED_HISTORY,SHM_HISTORY,CM_HISTORY,HMD_HISTORY,CMD_HISTORY}; //~ Added SHM_HISTORY option [KH - 30 October 2013] other three were added [MO - 30 November 2014]] 
enum{NONE,CONSTANT,EQUAL};

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixWallGran::FixWallGran(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), idregion(NULL), shearone(NULL), fix_rigid(NULL), mass_rigid(NULL)
{
  virial_flag = 1; // this fix can contribute to compute stress/atom
  if (narg < 4) error->all(FLERR,"Illegal fix wall/gran command");

  if (!atom->sphere_flag)
    error->all(FLERR,"Fix wall/gran requires atom style sphere");

  //~ Global information is saved to restart file [KH - 20 February 2014]
  restart_global = 1;
  restart_peratom = 1;
  
  create_attribute = 1;

  // set interaction style
  // disable bonded/history option for now

  if (strcmp(arg[3],"hooke") == 0) pairstyle = HOOKE;
  else if (strcmp(arg[3],"hooke/history") == 0) pairstyle = HOOKE_HISTORY;
  else if (strcmp(arg[3],"hertz/history") == 0) pairstyle = HERTZ_HISTORY;
  else if (strcmp(arg[3],"shm/history") == 0) pairstyle = SHM_HISTORY;
  else if (strcmp(arg[3],"HMD/history") == 0) pairstyle = HMD_HISTORY;
  else if (strcmp(arg[3],"CM/history") == 0) pairstyle = CM_HISTORY;
  else if (strcmp(arg[3],"CMD/history") == 0) pairstyle = CMD_HISTORY;
  //else if (strcmp(arg[3],"bonded/history") == 0) pairstyle = BONDED_HISTORY;
  else error->all(FLERR,"Invalid fix wall/gran interaction style");

  history = restart_peratom = 1;
  if (pairstyle == HOOKE) history = restart_peratom = 0;

  // wall/particle coefficients
  
  vector_flag = 1;
  size_vector = 6;  // increased from 5 to 6 [MO - 12 March 2015]
  global_freq = 1;

  // wall/particle coefficients

  //~ Make special allowances for shm pairstyle [KH - 30 October 2013]
  //~~ Added for CM, HMD and CMD pairstyle [MO - 05 January 2015]
  int iarg = 10;
  if (pairstyle == SHM_HISTORY) {
    Geq = force->numeric(FLERR,arg[4]);
    Poiseq = force->numeric(FLERR,arg[5]);
    xmu = force->numeric(FLERR,arg[6]);

    if (Geq < 0.0 || Poiseq < 0.0 || Poiseq > 0.5)
      error->all(FLERR,"Illegal shm pair parameter values in fix wall gran");

    kn = 4.0*Geq / (3.0*(1.0-Poiseq));
    kt = 4.0*Geq / (2.0-Poiseq);

    //~ Set dummy values for the remaining variables [KH - 9 January 2014]
    gamman = gammat = 0.0;
    dampflag = 0;
    iarg = 7; //~ Reduce number of args for SHM pairstyle [KH - 10 June 2014]
  } else if (pairstyle == CM_HISTORY) {
    Geq = force->numeric(FLERR,arg[4]);
    Poiseq = force->numeric(FLERR,arg[5]);
    xmu = force->numeric(FLERR,arg[6]);
    RMSf = force->numeric(FLERR,arg[7]);
    Hp = force->numeric(FLERR,arg[8]);
    Model = force->inumeric(FLERR,arg[9]);
    
    if (Geq < 0.0 || Poiseq < 0.0 || Poiseq > 0.5 || (Model != 0 && Model != 1))
      error->all(FLERR,"Illegal CM pair parameter values in fix wall gran");
    
    kn = 4.0*Geq / (3.0*(1.0-Poiseq));
    kt = 4.0*Geq / (2.0-Poiseq);
    
    //~ Set dummy values for the remaining variables [KH - 9 January 2014]
    gamman = gammat = 0.0;
    dampflag = 0;
  } else if (pairstyle == HMD_HISTORY) {
    Geq = force->numeric(FLERR,arg[4]);
    Poiseq = force->numeric(FLERR,arg[5]);
    xmu = force->numeric(FLERR,arg[6]);
    THETA1 = force->inumeric(FLERR,arg[7]);

    if (Geq < 0.0 || Poiseq < 0.0 || Poiseq > 0.5 || (THETA1 != 0 && THETA1 != 1))
      error->all(FLERR,"Illegal HMD pair parameter values in fix wall gran");

    kn = 4.0*Geq / (3.0*(1.0-Poiseq));
    kt = 4.0*Geq / (2.0-Poiseq);
    
    //~ Set dummy values for the remaining variables [KH - 9 January 2014]
    gamman = gammat = 0.0;
    dampflag = 0;
    iarg = 8; //~ Reduce number of args for HMD pairstyle [MO - 12 Sept 2015]
  } else if (pairstyle == CMD_HISTORY) {     
    Geq = force->numeric(FLERR,arg[4]);
    Poiseq = force->numeric(FLERR,arg[5]);
    xmu = force->numeric(FLERR,arg[6]);
    RMSf = force->numeric(FLERR,arg[7]);
    Hp = force->numeric(FLERR,arg[8]);
    Model = force->inumeric(FLERR,arg[9]);
    THETA1 = force->inumeric(FLERR,arg[10]);
    
    if (Geq < 0.0 || Poiseq < 0.0 || Poiseq > 0.5 || (Model != 0 && Model != 1) || (THETA1 != 0 && THETA1 != 1))
      error->all(FLERR,"Illegal CMD pair parameter values in fix wall gran");
    
    kn = 4.0*Geq / (3.0*(1.0-Poiseq));
    kt = 4.0*Geq / (2.0-Poiseq);
    
    //~ Set dummy values for the remaining variables [KH - 9 January 2014]
    gamman = gammat = 0.0;
    dampflag = 0;
    iarg = 11; //~ Change number of args for CMD pairstyle [MO - 12 Sep 2014]
  } else if (pairstyle == BONDED_HISTORY) {
    if (narg < 10) error->all(FLERR,"Illegal fix wall/gran command");

    E = force->numeric(FLERR,arg[4]);
    G = force->numeric(FLERR,arg[5]);
    SurfEnergy = force->numeric(FLERR,arg[6]);
    // Note: this doesn't get used, check w/ Jeremy?
    gamman = force->numeric(FLERR,arg[7]);

    xmu = force->numeric(FLERR,arg[8]);
    // pois = E/(2.0*G) - 1.0;
    // kn = 2.0*E/(3.0*(1.0+pois)*(1.0-pois));
    // gammat=0.5*gamman;

    iarg = 9;
  } else {
    kn = force->numeric(FLERR,arg[4]);
    if (strcmp(arg[5],"NULL") == 0) kt = kn * 2.0/7.0;
    else kt = force->numeric(FLERR,arg[5]);
    
    gamman = force->numeric(FLERR,arg[6]);
    if (strcmp(arg[7],"NULL") == 0) gammat = 0.5 * gamman;
    else gammat = force->numeric(FLERR,arg[7]);

    xmu = force->numeric(FLERR,arg[8]);
    int dampflag = force->inumeric(FLERR,arg[9]);
    if (dampflag == 0) gammat = 0.0;
  }
  
  if (kn < 0.0 || kt < 0.0 || gamman < 0.0 || gammat < 0.0 ||
      xmu < 0.0 || xmu > 10000.0 || dampflag < 0 || dampflag > 1)
    error->all(FLERR,"Illegal fix wall/gran command");

  // convert Kn and Kt from pressure units to force/distance^2 if Hertzian

  if (pairstyle == HERTZ_HISTORY || pairstyle == SHM_HISTORY) {
    kn /= force->nktv2p;
    kt /= force->nktv2p;
  }
  
  // wallstyle args

  idregion = NULL;

  if (strcmp(arg[iarg],"xplane") == 0) {
    if (narg < iarg+3) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = XPLANE;
    if (strcmp(arg[iarg+1],"NULL") == 0) lo = -BIG;
    else lo = force->numeric(FLERR,arg[iarg+1]);
    if (strcmp(arg[iarg+2],"NULL") == 0) hi = BIG;
    else hi = force->numeric(FLERR,arg[iarg+2]);
    iarg += 3;
  } else if (strcmp(arg[iarg],"yplane") == 0) {
    if (narg < iarg+3) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = YPLANE;
    if (strcmp(arg[iarg+1],"NULL") == 0) lo = -BIG;
    else lo = force->numeric(FLERR,arg[iarg+1]);
    if (strcmp(arg[iarg+2],"NULL") == 0) hi = BIG;
    else hi = force->numeric(FLERR,arg[iarg+2]);
    iarg += 3;
  } else if (strcmp(arg[iarg],"zplane") == 0) {
    if (narg < iarg+3) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = ZPLANE;
    if (strcmp(arg[iarg+1],"NULL") == 0) lo = -BIG;
    else lo = force->numeric(FLERR,arg[iarg+1]);
    if (strcmp(arg[iarg+2],"NULL") == 0) hi = BIG;
    else hi = force->numeric(FLERR,arg[iarg+2]);
    iarg += 3;
  } else if (strcmp(arg[iarg],"zcylinder") == 0) {
    if (narg < iarg+2) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = ZCYLINDER;
    lo = hi = 0.0;
    cylradius = force->numeric(FLERR,arg[iarg+3]);
    iarg += 2;
  } else if (strcmp(arg[iarg],"region") == 0) {
    if (narg < iarg+2) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = REGION;
    int n = strlen(arg[iarg+1]) + 1;
    idregion = new char[n];
    strcpy(idregion,arg[iarg+1]);
    iarg += 2;
  }

  // optional args

  wiggle = 0;
  wshear = 0;
  wtranslate = 0;
  wscontrol = 0;
  ftvarying = 0;
  fstr = NULL;
  vwall[0] = vwall[1] = vwall[2] = 0.0;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"wiggle") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix wall/gran command");
      if (strcmp(arg[iarg+1],"x") == 0) axis = 0;
      else if (strcmp(arg[iarg+1],"y") == 0) axis = 1;
      else if (strcmp(arg[iarg+1],"z") == 0) axis = 2;
      else error->all(FLERR,"Illegal fix wall/gran command");
      amplitude = force->numeric(FLERR,arg[iarg+2]);
      period = force->numeric(FLERR,arg[iarg+3]);
      if (strcmp(arg[iarg+4],"cos") == 0) wiggletype = 1;
      else if (strcmp(arg[iarg+4],"sin") == 0) wiggletype = 2;
      else error->all(FLERR,"Illegal fix wall/gran command");
      wiggle = 1;
      //loINI = lo; 
      //hiINI = hi; 
      iarg += 5;
    } else if (strcmp(arg[iarg],"shear") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix wall/gran command");
      if (strcmp(arg[iarg+1],"x") == 0) axis = 0;
      else if (strcmp(arg[iarg+1],"y") == 0) axis = 1;
      else if (strcmp(arg[iarg+1],"z") == 0) axis = 2;
      else error->all(FLERR,"Illegal fix wall/gran command");
      vshear = force->numeric(FLERR,arg[iarg+2]);
      wshear = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg],"translate") == 0) {
      wtranslate = 1;
      vwall[0] = force->numeric(FLERR,arg[iarg+1]);
      vwall[1] = force->numeric(FLERR,arg[iarg+2]);
      vwall[2] = force->numeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"stresscontrol") == 0) {
      wscontrol = 1;
      //wtranslate = 1; removed wtranslate [MO - 28 Aug 2015]
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        ftvarying = 1;
        int nn = strlen(&arg[iarg+1][2]) + 1;
        fstr = new char[nn];
        strcpy(fstr,&arg[iarg+1][2]);
      } else targetf = force->numeric(FLERR,arg[iarg+1]);
      gain = force->numeric(FLERR,arg[iarg+2]);
      if (strcmp(arg[iarg+2],"auto") == 0) error->all(FLERR,"Illegal fix wall/gran command - more coding needed");
      iarg += 3;
    } else error->all(FLERR,"Illegal fix wall/gran command");
  }

  if (wscontrol == 1 && (lo != -BIG && hi != BIG)) error->all(FLERR,"Cannot have both lo and hi walls with stresscontrol"); // put warning message for fix output too?
  if (wiggle == 1 && (lo != -BIG && hi != BIG)) error->all(FLERR,"Cannot have both lo and hi walls with wiggle"); // Added [MO - 27 Aug 2015]

  if (wallstyle == XPLANE && domain->xperiodic)
    error->all(FLERR,"Cannot use wall in periodic dimension");
  if (wallstyle == YPLANE && domain->yperiodic)
    error->all(FLERR,"Cannot use wall in periodic dimension");
  if (wallstyle == ZPLANE && domain->zperiodic)
    error->all(FLERR,"Cannot use wall in periodic dimension");
  if (wallstyle == ZCYLINDER && (domain->xperiodic || domain->yperiodic))
    error->all(FLERR,"Cannot use wall in periodic dimension");

  if (wiggle && wshear)
    error->all(FLERR,"Cannot wiggle and shear fix wall/gran");
  if (wiggle && wallstyle == ZCYLINDER && axis != 2)
    error->all(FLERR,"Invalid wiggle direction for fix wall/gran");
  if (wshear && wallstyle == XPLANE && axis == 0)
    error->all(FLERR,"Invalid shear direction for fix wall/gran");
  if (wshear && wallstyle == YPLANE && axis == 1)
    error->all(FLERR,"Invalid shear direction for fix wall/gran");
  if (wshear && wallstyle == ZPLANE && axis == 2)
    error->all(FLERR,"Invalid shear direction for fix wall/gran");
  if ((wtranslate || wscontrol) && (lo != -BIG && hi != BIG)) // added wscontrol [MO - 28 Aug 2015]
    error->all(FLERR,"Cannot specify both top and bottom walls and translate for fix wall/gran");
  if ((wtranslate || wscontrol) && wallstyle == ZCYLINDER) // added wscontrol [MO - 28 Aug 2015]
    error->all(FLERR,"Cannot use translate with cylinder fix wall/gran");
  if ((wtranslate || wscontrol) && (wiggle || wshear)) // added wscontrol [MO - 28 Aug 2015]
    error->all(FLERR,"Cannot translate and wiggle or shear fix wall/gran");
  if ((wiggle || wshear) && wallstyle == REGION)
    error->all(FLERR,"Cannot wiggle or shear with fix wall/gran/region");

  // setup oscillations

  if (wiggle) omega = 2.0*MY_PI / period;

  // perform initial allocation of atom-based arrays
  // register with Atom class

  if (pairstyle == BONDED_HISTORY) sheardim = 7;
  else sheardim = 3;

  //~ pair/gran/shm/history has 4 shear quantities
  if (pairstyle == SHM_HISTORY) sheardim++;

  //~ pair/gran/CM/history has 5 shear quantities [MO - 18 July 2014]
  if (pairstyle == CM_HISTORY) sheardim += 2;
  
  //~ pair/gran/HMD/history has 26 shear quantities [MO - 21 July 2014]
  if (pairstyle == HMD_HISTORY) sheardim += 23;

  //~ pair/gran/CMD/history has 26 shear quantities [MO - 30 November 2014]
  if (pairstyle == CMD_HISTORY) sheardim += 23;

  shearone = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  atom->add_callback(1);

  nmax = 0;
  mass_rigid = NULL;
  
  /*~ Adding a rolling resistance model causes the number of shear 
    history quantities to be increased by 15 [KH - 29 July 2014]*/
  // 20 quantities for Deresiewicz1954_spin model [MO - 30 November 2014]
  int dim = 1;
  Pair *pair;
  if (force->pair_match("gran/hooke/history",1)) 
    pair = force->pair_match("gran/hooke/history",1);
  else if (force->pair_match("gran/hertz/history",1))
    pair = force->pair_match("gran/hertz/history",1);
  else if (force->pair_match("gran/shm/history",1))
    pair = force->pair_match("gran/shm/history",1);
  else if (force->pair_match("gran/CM/history",1))    //[MO - 18 July 2014]
    pair = force->pair_match("gran/CM/history",1);
  else if (force->pair_match("gran/HMD/history",1))   //[MO - 21 July 2014]
    pair = force->pair_match("gran/HMD/history",1);
  else if (force->pair_match("gran/CMD/history",1))   //[MO - 30 November 2014]
    pair = force->pair_match("gran/CMD/history",1);
  else dim = 0; //~ Adding for other pairstyles [KH - 5 November 2013]

  if (dim) { // if(dim) {} was modified [KH&MO - 04 December 2014]
    rolling = (int *) pair->extract("rolling",dim);
    if (*rolling) sheardim += 15;

    // Option of D_spin was added [MO - 4 December 2014]
    D_spin = (int *) pair->extract("D_spin",dim);
    if (*D_spin) sheardim += 20;

    trace_energy = (int *) pair->extract("trace_energy",dim);
    if (*trace_energy) sheardim += 4;
  }

  // parameters of the particle [MO - 05 December 2014]
  xmu_p = (double *) pair->extract("xmu",dim);
  Geq_p = (double *) pair->extract("Geq",dim);
  Poiseq_p = (double *) pair->extract("Poiseq",dim);
  RMSf_p = (double *) pair->extract("RMSf",dim);
  Hp_p = (double *) pair->extract("Hp",dim);
   
  /*~ Use same method to obtain model_type, rolling_delta, kappa and
    post_limit_index from pairstyles. Also initialise two integers
    used to limit the numbers of warnings about failures to calculate 
    either of the contact stiffnesses in the rolling resistance model
    [KH - 29 July 2014]*/
  if (*rolling) {
    model_type = (int *) pair->extract("model_type",dim);
    rolling_delta = (double *) pair->extract("rolling_delta",dim);
    kappa = (double *) pair->extract("kappa",dim);
    post_limit_index = (double *) pair->extract("post_limit_index",dim);
    lastwarning[0] = lastwarning[1] = -1000000;
  }
  // D_switch == 1 or 0 mean D_spin is on or off. [MO - 05 December 2014]  
  if (*D_spin) D_switch = (int *) pair->extract("D_switch",dim);

  /*~ Note that there is no need to set up fix_old_omega as the pairstyle
    will do this automatically*/

  /*~ At present, not allowed to have cylindrical wall with rolling
    resistance model*/
  if (*rolling && wallstyle == ZCYLINDER)
    error->all(FLERR,"Not permitted to use rolling resistance with cylindrical walls");
  // Also for D_spin model
  if (*D_spin && wallstyle == ZCYLINDER)
    error->all(FLERR,"Not permitted to use D_spin resistance with cylindrical walls");

  // initialize shear history as if particle is not touching region
  // shearone will be NULL for wallstyle = REGION

  if (history && shearone) {
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++)
      for (int j = 0; j < sheardim; j++)
        shearone[i][j] = 0.0;
  }

  time_origin = update->ntimestep;

  /*~ Initialise the accumulated energy terms to zero. For the
    linear contact model, the shear strain is not calculated
    cumulatively, but it makes no difference to zero it here
    anyway [KH - 27 February 2014]*/
  dissipfriction = shearstrain = spinenergy = 0.0;
  // spin_energy was added. [MO - 30 November 2014]
}

/* ---------------------------------------------------------------------- */

FixWallGran::~FixWallGran()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);
  atom->delete_callback(id,1);

  // delete local storage

  delete [] idregion;
  memory->destroy(shearone);
  memory->destroy(mass_rigid);
}

/* ---------------------------------------------------------------------- */

int FixWallGran::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallGran::init()
{
  int i;

  dt = update->dt;

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // check for FixRigid so can extract rigid body masses
  fix_rigid = NULL;
  for (i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->rigid_flag) break;
  if (i < modify->nfix) fix_rigid = modify->fix[i];
}

/* ---------------------------------------------------------------------- */

void FixWallGran::setup(int vflag)
{
  // [MO - 13 AUGUST 2015] /////////////////////////////////////
  if (wscontrol) {
    //~ Firstly check whether a stress/atom compute already exists
    for (int q = 0; q < modify->ncompute; q++)
      if (strcmp(modify->compute[q]->style,"stress/atom") == 0) {
	stressatom = modify->compute[q];
	sfound = 1;
	break;
      }
   
    if (!sfound) { //~ Need to set up a new compute stress/atom  
      char **snewarg = new char*[7];
      snewarg[0] = (char *) "e_stress_comp";
      snewarg[1] = (char *) "all";
      snewarg[2] = (char *) "stress/atom";
      snewarg[3] = (char *) "NULL";
      snewarg[4] = (char *) "pair";
      snewarg[5] = (char *) "fix";
      snewarg[6] = (char *) "bond";
      
      modify->add_compute(7,snewarg);
      stressatom = modify->compute[modify->find_compute("e_stress_comp")];
     
      delete [] snewarg;
    }
  }
  ////////////////////////////////////////////////////////////////
  

  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }

  //~ Set pointers for fix old omega [KH - 30 October 2013]
  int accfix;
  if (*rolling) {
    accfix = modify->find_fix("pair_oldomega");
    if (accfix < 0) error->all(FLERR,"Fix ID for old_omega does not exist");
    deffix = modify->fix[accfix];
  }
}

/* ---------------------------------------------------------------------- */

void FixWallGran::post_force(int vflag)
{
  int i,j;
  double dx,dy,dz,del1,del2,delxy,delr,rsq,rwall,meff;

  // do not update shear history during setup

  shearupdate = 1;
  if (update->setupflag) shearupdate = 0;

  // if just reneighbored:
  // update rigid body masses for owned atoms if using FixRigid
  //   body[i] = which body atom I is in, -1 if none
  //   mass_body = mass of each rigid body

  if (neighbor->ago == 0 && fix_rigid) {
    int tmp;
    int *body = (int *) fix_rigid->extract("body",tmp);
    double *mass_body = (double *) fix_rigid->extract("masstotal",tmp);
    if (atom->nmax > nmax) {
      memory->destroy(mass_rigid);
      nmax = atom->nmax;
      memory->create(mass_rigid,nmax,"wall/gran:mass_rigid");
    }
    int nlocal = atom->nlocal;
    for (i = 0; i < nlocal; i++) {
      if (body[i] >= 0) mass_rigid[i] = mass_body[body[i]];
      else mass_rigid[i] = 0.0;
    }
  }

  // if wiggle or shear, set wall position and velocity accordingly
  // if wtranslate lo and hi track the wall position and vwall is set in the constructor

  if (wiggle) { 
    double arg = omega * (update->ntimestep - time_origin) * dt;
    if (wiggletype == 1) vwall[axis] = amplitude*omega*sin(arg); // same as before [MO - 09 May 2016]
    else                 vwall[axis] = amplitude*omega*cos(arg); // newly added [MO - 09 May 2016]
    if (shearupdate && wallstyle == axis) move_wall(); // move_wall will update hi & lo
  } 
  else if (wtranslate && shearupdate) move_wall(); // move_wall will update hi & lo
  else if (wshear) vwall[axis] = vshear;

  fwall[0] = fwall[1] = fwall[2] = 0.0; //per-processor force// fwall_all[0] = fwall_all[1] = fwall_all[2] = 0.0;
  wcoordnos[0] = 0.0; // coordination number of wall [MO - 12 March 2015]
  
  /*~ Ascertain whether or not energy tracing is active by checking
    for the presence of compute energy/gran. If so, check if the
    tracked terms include those calculated in this fix. The
    energy terms are updated only if necessary for efficiency.
    [KH - 20 February 2014]*/
  pairenergy = *trace_energy;
  if (!pairenergy)
    for (int q = 0; q < modify->ncompute; q++)
      if (strcmp(modify->compute[q]->style,"energy/gran") == 0) {
	pairenergy = ((ComputeEnergyGran *) modify->compute[q])->pairenergy;
	break;
      }
  
  //~ Initialise the non-accumulated strain energy terms to zero
  normalstrain = 0.0;

  
  //***********// This is for HMD and CMD models
  dissipfriction = 0.0; // TEMPORARY [MO - 25 Sep 2015]
  //***********


  if (pairstyle == HOOKE_HISTORY) shearstrain = 0.0;
  
  // loop over all my atoms
  // rsq = distance from wall
  // dx,dy,dz = signed distance from wall
  // for rotating cylinder, reset vwall based on particle position
  // skip atom if not close enough to wall
  //   if wall was set to NULL, it's skipped since lo/hi are infinity
  // compute force and torque on atom if close enough to wall
  //   via wall potential matched to pair potential
  // set shear if pair potential stores history

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  rwall = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      dx = dy = dz = 0.0;

      if (wallstyle == XPLANE) {
        del1 = x[i][0] - lo;
        del2 = hi - x[i][0];
        if (del1 < del2) dx = del1;
        else dx = -del2;
      } else if (wallstyle == YPLANE) {
        del1 = x[i][1] - lo;
        del2 = hi - x[i][1];
        if (del1 < del2) dy = del1;
        else dy = -del2;
      } else if (wallstyle == ZPLANE) {
        del1 = x[i][2] - lo;
        del2 = hi - x[i][2];
        if (del1 < del2) dz = del1;
        else dz = -del2;
      } else if (wallstyle == ZCYLINDER) {
        delxy = sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1]);
        delr = cylradius - delxy;
        if (delr > radius[i]) {
          dz = cylradius;
          rwall = 0.0;
        } else {
          dx = -delr/delxy * x[i][0];
          dy = -delr/delxy * x[i][1];
          // rwall = -2r_c if inside cylinder, 2r_c outside
          rwall = (delxy < cylradius) ? -2*cylradius : 2*cylradius;
          if (wshear && axis != 2) {
            vwall[0] += vshear * x[i][1]/delxy;
            vwall[1] += -vshear * x[i][0]/delxy;
            vwall[2] = 0.0;
          }
        }
      }

      rsq = dx*dx + dy*dy + dz*dz;

      if (rsq > radius[i]*radius[i]) {
        if (history)
          for (j = 0; j < sheardim; j++)
            shearone[i][j] = 0.0;

      } else {
	// meff = effective mass of sphere
	// if I is part of rigid body, use body mass

	meff = rmass[i];
        if (fix_rigid && mass_rigid[i] > 0.0) meff = mass_rigid[i];	
	  
	wcoordnos[0] += 1.0; // accumulate coordination number [MO - 12 March 2015]

        if (pairstyle == HOOKE)
          hooke(rsq,dx,dy,dz,vwall,v[i],f[i],omega[i],torque[i],
                radius[i],meff,i);
        else if (pairstyle == HOOKE_HISTORY)
          hooke_history(rsq,dx,dy,dz,vwall,v[i],f[i],omega[i],torque[i],
                        radius[i],meff,shearone[i],i);
        else if (pairstyle == HERTZ_HISTORY)
          hertz_history(rsq,dx,dy,dz,vwall,v[i],f[i],omega[i],torque[i],
                        radius[i],meff,shearone[i],i);
	else if (pairstyle == SHM_HISTORY) //~ [KH - 30 October 2013]
          shm_history(rsq,dx,dy,dz,vwall,v[i],f[i],omega[i],torque[i],
		      radius[i],meff,shearone[i],i);
	else if (pairstyle == CM_HISTORY) //~ [MO - 18 July 2014]
          CM_history(rsq,dx,dy,dz,vwall,v[i],f[i],omega[i],torque[i],
		     radius[i],meff,shearone[i],i);
	else if (pairstyle == HMD_HISTORY) //~ [MO - 21 July 2014]
          HMD_history(rsq,dx,dy,dz,vwall,v[i],f[i],omega[i],torque[i],
		      radius[i],meff,shearone[i],i);
	else if (pairstyle == CMD_HISTORY) //~ [MO - 30 November 2014]
          CMD_history(rsq,dx,dy,dz,vwall,v[i],f[i],omega[i],torque[i],
		      radius[i],meff,shearone[i],i);
        else if (pairstyle == BONDED_HISTORY)
          bonded_history(rsq,dx,dy,dz,vwall,rwall,v[i],f[i],
                         omega[i],torque[i],radius[i],meff,shearone[i]);
      }
    }
  } 
  if (wscontrol) { // velscontrol and move_wall are called here [MO - 28 Aug 2015]
    velscontrol(); 
    if (shearupdate) move_wall(); // move_wall will update hi & lo
  } 
}

/* ---------------------------------------------------------------------- */

void FixWallGran::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallGran::hooke(double rsq, double dx, double dy, double dz,
                        double *vwall, double *v,
                        double *f, double *omega, double *torque,
                        double radius, double meff, int i)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double damp,ccel,vtr1,vtr2,vtr3,vrel;
  double fn,fs,ft,fs1,fs2,fs3,fx,fy,fz,rinv,rsqinv;

  r = sqrt(rsq);
  rinv = 1.0/r;
  rsqinv = 1.0/rsq;

  // relative translational velocity

  vr1 = v[0] - vwall[0];
  vr2 = v[1] - vwall[1];
  vr3 = v[2] - vwall[2];

  // normal component

  vnnr = vr1*dx + vr2*dy + vr3*dz;
  vn1 = dx*vnnr * rsqinv;
  vn2 = dy*vnnr * rsqinv;
  vn3 = dz*vnnr * rsqinv;

  // tangential component

  vt1 = vr1 - vn1;
  vt2 = vr2 - vn2;
  vt3 = vr3 - vn3;

  // relative rotational velocity - removed
  //wr1 = radius*omega[0] * rinv; //radius should be substituted by r, so wr1==omega[0] etc

  // normal forces = Hookian contact + normal velocity damping

  damp = meff*gamman*vnnr*rsqinv;
  ccel = kn*(radius-r)*rinv - damp;

  // relative velocities

  vtr1 = vt1 - dz*omega[1]+dy*omega[2];
  vtr2 = vt2 - dx*omega[2]+dz*omega[0];
  vtr3 = vt3 - dy*omega[0]+dx*omega[1];
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);

  // force normalization

  fn = xmu * fabs(ccel*r);
  fs = meff*gammat*vrel;
  if (vrel != 0.0) ft = MIN(fn,fs) / vrel;
  else ft = 0.0;

  // tangential force due to tangential velocity damping

  fs1 = -ft*vtr1;
  fs2 = -ft*vtr2;
  fs3 = -ft*vtr3;

  // forces & torques

  fx = dx*ccel + fs1;
  fy = dy*ccel + fs2;
  fz = dz*ccel + fs3;

  f[0] += fx;
  f[1] += fy;
  f[2] += fz;

  torque[0] -= dy*fs3 - dz*fs2;
  torque[1] -= dz*fs1 - dx*fs3;
  torque[2] -= dx*fs2 - dy*fs1;

  fwall[0] += fx;
  fwall[1] += fy;
  fwall[2] += fz;
  if (evflag) ev_tally_wall(i,fx,fy,fz,dx,dy,dz,radius);

}

/* ---------------------------------------------------------------------- */

void FixWallGran::hooke_history(double rsq, double dx, double dy, double dz,
                                double *vwall, double *v,
                                double *f, double *omega, double *torque,
                                double radius, double meff, double *shear, int i)
{
  /*~ This function was modified extensively so that shear force
    is stored in the shear array rather than displacement
    [KH - 1 April 2015]*/

  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double damp,ccel,vtr1,vtr2,vtr3,vrel;
  double fs,fslim,fx,fy,fz;
  double shsqmag,shsqnew,rsht,shratio,polyhertz,rinv,rsqinv;
  double wspinx,wspiny,wspinz,shint0,shint1,shint2,omdel;

  r = sqrt(rsq);
  rinv = 1.0/r;
  rsqinv = 1.0/rsq;

  // relative translational velocity

  vr1 = v[0] - vwall[0];
  vr2 = v[1] - vwall[1];
  vr3 = v[2] - vwall[2];

  // normal component

  vnnr = vr1*dx + vr2*dy + vr3*dz;
  vn1 = dx*vnnr * rsqinv;
  vn2 = dy*vnnr * rsqinv;
  vn3 = dz*vnnr * rsqinv;

  // tangential component

  vt1 = vr1 - vn1;
  vt2 = vr2 - vn2;
  vt3 = vr3 - vn3;

  // relative rotational velocity - removed see above

  // normal forces = Hookian contact + normal velocity damping

  damp = meff*gamman*vnnr*rsqinv;
  ccel = kn*(radius-r)*rinv - damp;

  // relative velocities

  vtr1 = vt1 - dz*omega[1]+dy*omega[2];
  vtr2 = vt2 - dx*omega[2]+dz*omega[0];
  vtr3 = vt3 - dy*omega[0]+dx*omega[1];
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);

  // shear history effects
  //~ Note that shear now refers to shear force, not shear displacement

  shsqmag = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];

  // rotate shear forces onto new contact plane conserving length

  rsht = shear[0]*dx + shear[1]*dy + shear[2]*dz;
  rsht = rsht*rsqinv;
  if (shearupdate) {
    shear[0] -= rsht*dx;
    shear[1] -= rsht*dy;
    shear[2] -= rsht*dz;
    shsqnew = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];
    if (shsqnew!=0.0) {
      shratio=sqrt(shsqmag/shsqnew);
      shear[0] *= shratio; // conserve shear force length
      shear[1] *= shratio;
      shear[2] *= shratio;
    }
  }
 
  // then perform rotation for rigid-body SPIN
  omdel=omega[0]*dx+omega[1]*dy+omega[2]*dz;
  wspinx=0.5*rsqinv*dx*omdel;
  wspiny=0.5*rsqinv*dy*omdel;
  wspinz=0.5*rsqinv*dz*omdel;
  shint0 = shear[0];
  shint1 = shear[1];
  shint2 = shear[2];

  if (shearupdate) {
    shear[0]=shint0+shint1*(-wspinz*dt)+shint2*wspiny*dt;
    shear[1]=shint0*wspinz*dt+shint1+shint2*(-wspinx*dt);
    shear[2]=shint0*(-wspiny*dt)+shint1*wspinx*dt+shint2;
  }

  // tangential forces = shear + tangential velocity damping
  if (shearupdate) {
    shear[0] -= kt*vtr1*dt;//shear displacement =vtr*dt
    shear[1] -= kt*vtr2*dt;
    shear[2] -= kt*vtr3*dt;
  }

  // rescale frictional forces if needed

  fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  fslim = xmu * fabs(ccel*r);

  if (fs > fslim) {
    if (fs != 0.0) {
      if (shearupdate) {
	shear[0] *= fslim/fs;
	shear[1] *= fslim/fs;
	shear[2] *= fslim/fs;
      }
    } else shear[0] = shear[1] = shear[2] = 0.0;
  }

  // forces & torques

  fx = dx*ccel + shear[0];
  fy = dy*ccel + shear[1];
  fz = dz*ccel + shear[2];

  f[0] += fx;
  f[1] += fy;
  f[2] += fz;

  torque[0] -= dy*shear[2] - dz*shear[1];
  torque[1] -= dz*shear[0] - dx*shear[2];
  torque[2] -= dx*shear[1] - dy*shear[0];

  //~ Call function for rolling resistance model [KH - 30 October 2013]
  double db[3], localdM[3], globaldM[3]; //~ Pass by reference
  if (*rolling && shearupdate)
    rolling_resistance(i,sheardim,dx,dy,dz,r,radius,ccel,fslim,
		       kt,torque,shear,db,localdM,globaldM);

  //~ Add contributions to traced energy [KH - 20 February 2014]
  double aveshearforce, slipdisp, incdissipf, nstr, sstr;
  if (pairenergy) {
    /*~ Increment the friction energy only if the slip condition
      is invoked*/
    if (fs > fslim && fslim > 0.0) {
      slipdisp = (fs-fslim)/kt;
      aveshearforce = 0.5*(sqrt(shsqmag) + fslim);

      //~ slipdisp and aveshearforce are both positive
      incdissipf = aveshearforce*slipdisp;
      dissipfriction += incdissipf;
      if (*trace_energy) shear[3] += incdissipf;
    }

    /*~ Update the strain energy terms which don't need to be 
      calculated incrementally*/
    nstr = 0.5*kn*(radius-r)*(radius-r);
    sstr = 0.5*(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2])/kt;
    normalstrain += nstr;
    shearstrain += sstr;

    if (trace_energy) {
      shear[4] = nstr;
      shear[5] = sstr;
    }
  }

  fwall[0] += fx;
  fwall[1] += fy;
  fwall[2] += fz;
  if (evflag) ev_tally_wall(i,fx,fy,fz,dx,dy,dz,radius);
}

/* ---------------------------------------------------------------------- */

void FixWallGran::hertz_history(double rsq, double dx, double dy, double dz,
                                double *vwall, double rwall, double *v,
                                double *f, double *omega, double *torque,
                                double radius, double meff, double *shear, int i)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double damp,ccel,vtr1,vtr2,vtr3,vrel;
  double fn,fs,fs1,fs2,fs3,fx,fy,fz;
  double shrmag,rsht,polyhertz,rinv,rsqinv,oldshear[3];

  r = sqrt(rsq);
  rinv = 1.0/r;
  rsqinv = 1.0/rsq;

  // relative translational velocity

  vr1 = v[0] - vwall[0];
  vr2 = v[1] - vwall[1];
  vr3 = v[2] - vwall[2];

  // normal component

  vnnr = vr1*dx + vr2*dy + vr3*dz;
  vn1 = dx*vnnr / rsq;
  vn2 = dy*vnnr / rsq;
  vn3 = dz*vnnr / rsq;

  // tangential component

  vt1 = vr1 - vn1;
  vt2 = vr2 - vn2;
  vt3 = vr3 - vn3;

  // relative rotational velocity - removed

  // normal forces = Hertzian contact + normal velocity damping
  // rwall = 0 is flat wall case
  // rwall positive or negative is curved wall
  //   will break (as it should) if rwall is negative and
  //   its absolute value < radius of particle

  damp = meff*gamman*vnnr*rsqinv;
  ccel = kn*(radius-r)*rinv - damp;
  if (rwall == 0.0) polyhertz = sqrt((radius-r)*radius);
  else polyhertz = sqrt((radius-r)*radius*rwall/(rwall+radius));
  ccel *= polyhertz;

  // relative velocities

  vtr1 = vt1 - dz*omega[1]+dy*omega[2];
  vtr2 = vt2 - dx*omega[2]+dz*omega[0];
  vtr3 = vt3 - dy*omega[0]+dx*omega[1];
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);

  // shear history effects

  if (shearupdate) {
    //~ Store the old values of shear [KH - 20 February 2014]
    oldshear[0] = shear[0];
    oldshear[1] = shear[1];
    oldshear[2] = shear[2];

    shear[0] += vtr1*dt;
    shear[1] += vtr2*dt;
    shear[2] += vtr3*dt;
  }
  shrmag = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);

  // rotate shear displacements

  rsht = shear[0]*dx + shear[1]*dy + shear[2]*dz;
  rsht = rsht*rsqinv;
  if (shearupdate) {
    shear[0] -= rsht*dx;
    shear[1] -= rsht*dy;
    shear[2] -= rsht*dz;
  }

  // tangential forces = shear + tangential velocity damping

  fs1 = -polyhertz * (kt*shear[0] + meff*gammat*vtr1);
  fs2 = -polyhertz * (kt*shear[1] + meff*gammat*vtr2);
  fs3 = -polyhertz * (kt*shear[2] + meff*gammat*vtr3);

  // rescale frictional displacements and forces if needed

  fs = sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
  fn = xmu * fabs(ccel*r);

  if (fs > fn) {
    if (shrmag != 0.0) {
      shear[0] = (fn/fs) * (shear[0] + meff*gammat*vtr1/kt) -
        meff*gammat*vtr1/kt;
      shear[1] = (fn/fs) * (shear[1] + meff*gammat*vtr2/kt) -
        meff*gammat*vtr2/kt;
      shear[2] = (fn/fs) * (shear[2] + meff*gammat*vtr3/kt) -
        meff*gammat*vtr3/kt;
      fs1 *= fn/fs ;
      fs2 *= fn/fs;
      fs3 *= fn/fs;
    } else fs1 = fs2 = fs3 = 0.0;
  }

  // forces & torques

  fx = dx*ccel + fs1;
  fy = dy*ccel + fs2;
  fz = dz*ccel + fs3;

  f[0] += fx;
  f[1] += fy;
  f[2] += fz;

  torque[0] -= dy*fs3 - dz*fs2;
  torque[1] -= dz*fs1 - dx*fs3;
  torque[2] -= dx*fs2 - dy*fs1;

  //~ Call function for rolling resistance model [KH - 30 October 2013]
  double effectivekt = kt*polyhertz;
  double db[3], localdM[3], globaldM[3]; //~ Pass by reference
  if (*rolling && shearupdate)
    rolling_resistance(i,sheardim,dx,dy,dz,r,radius,ccel,fn,
		       effectivekt,torque,shear,db,localdM,globaldM);

  //~ Add contributions to traced energy [KH - 20 February 2014]
  double aveshearforce, slipdisp, oldsheardisp, incrementaldisp;
  double incdissipf, nstr, sstr;
  if (pairenergy) {
    /*~ Increment the friction energy only if the slip condition
      is invoked*/
    oldsheardisp = sqrt(oldshear[0]*oldshear[0] + oldshear[1]*oldshear[1] + oldshear[2]*oldshear[2]);
    if (fs > fn && fn > 0.0) {
      //~ current shear displacement = fn/effectivekt;
      slipdisp = (fs-fn)/effectivekt;
      aveshearforce = 0.5*(fn + polyhertz*kt*oldsheardisp);

      //~ slipdisp and aveshearforce are both positive
      incdissipf = aveshearforce*slipdisp;
      dissipfriction += incdissipf;
      if (trace_energy) shear[3] += incdissipf;
    }

    /*~ Update the normal contribution to strain energy which 
      doesn't need to be calculated incrementally*/
    nstr = 0.4*kn*polyhertz*(radius-r)*(radius-r);
    normalstrain += nstr;
    if (trace_energy) shear[4] = nstr;
	    
    //~ The shear component does require incremental calculation
    if (shearupdate) {
      incrementaldisp = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]) - oldsheardisp;
      sstr = 0.5*incrementaldisp*(2.0*sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3)-effectivekt*incrementaldisp);
      shearstrain += sstr;
      if (trace_energy) shear[5] += sstr;
    }
  }

  fwall[0] += fx;
  fwall[1] += fy;
  fwall[2] += fz;
  if (evflag) ev_tally_wall(i,fx,fy,fz,dx,dy,dz,radius);
}

/* ---------------------------------------------------------------------- */

void FixWallGran::bonded_history(double rsq, double dx, double dy, double dz,
                                 double *vwall, double rwall, double *v,
                                 double *f, double *omega, double *torque,
                                 double radius, double meff, double *shear)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double wr1,wr2,wr3,damp,ccel,vtr1,vtr2,vtr3,vrel;
  double fn,fs,fs1,fs2,fs3,fx,fy,fz,tor1,tor2,tor3;
  double shrmag,rsht,polyhertz,rinv,rsqinv;

  double pois,E_eff,G_eff,rad_eff;
  double a0,Fcrit,delcrit,delcritinv;
  double overlap,olapsq,olapcubed,sqrtterm,tmp,keyterm,keyterm2,keyterm3;
  double aovera0,foverFc;
  double gammatsuji;

  double ktwist,kroll,twistcrit,rollcrit;
  double relrot1,relrot2,relrot3,vrl1,vrl2,vrl3,vrlmag,vrlmaginv;
  double magtwist,magtortwist;
  double magrollsq,magroll,magrollinv,magtorroll;

  //~ Note that this function has not been revised at all [KH - 23 May 2017]
  
  r = sqrt(rsq);
  rinv = 1.0/r;
  rsqinv = 1.0/rsq;

  // relative translational velocity

  vr1 = v[0] - vwall[0];
  vr2 = v[1] - vwall[1];
  vr3 = v[2] - vwall[2];

  // normal component

  vnnr = vr1*dx + vr2*dy + vr3*dz;
  vn1 = dx*vnnr / rsq;
  vn2 = dy*vnnr / rsq;
  vn3 = dz*vnnr / rsq;

  // tangential component

  vt1 = vr1 - vn1;
  vt2 = vr2 - vn2;
  vt3 = vr3 - vn3;

  // relative rotational velocity

  wr1 = radius*omega[0] * rinv;
  wr2 = radius*omega[1] * rinv;
  wr3 = radius*omega[2] * rinv;

  // normal forces = Hertzian contact + normal velocity damping
  // material properties: currently assumes identical materials

  pois = E/(2.0*G) - 1.0;
  E_eff=0.5*E/(1.0-pois*pois);
  G_eff=G/(4.0-2.0*pois);

  // rwall = 0 is infinite wall radius of curvature (flat wall)

  if (rwall == 0) rad_eff = radius;
  else rad_eff = radius*rwall/(radius+rwall);

  Fcrit = rad_eff * (3.0 * M_PI * SurfEnergy);
  a0=pow(9.0*M_PI*SurfEnergy*rad_eff*rad_eff/E_eff,1.0/3.0);
  delcrit = 1.0/rad_eff*(0.5 * a0*a0/pow(6.0,1.0/3.0));
  delcritinv = 1.0/delcrit;

  overlap = (radius-r) * delcritinv;
  olapsq = overlap*overlap;
  olapcubed = olapsq*overlap;
  sqrtterm = sqrt(1.0 + olapcubed);
  tmp = 2.0 + olapcubed + 2.0*sqrtterm;
  keyterm = pow(tmp,THIRD);
  keyterm2 = olapsq/keyterm;
  keyterm3 = sqrt(overlap + keyterm2 + keyterm);
  aovera0 = pow(6.0,-TWOTHIRDS) * (keyterm3 +
            sqrt(2.0*overlap - keyterm2 - keyterm + 4.0/keyterm3));
  foverFc = 4.0*((aovera0*aovera0*aovera0) - pow(aovera0,1.5));
  ccel = Fcrit*foverFc*rinv;

  // damp = meff*gamman*vnnr*rsqinv;
  // ccel = kn*(radius-r)*rinv - damp;
  // polyhertz = sqrt((radius-r)*radius);
  // ccel *= polyhertz;

  // use Tsuji et al form

  polyhertz = 1.2728- 4.2783*0.9 + 11.087*0.9*0.9 - 22.348*0.9*0.9*0.9 +
    27.467*0.9*0.9*0.9*0.9 - 18.022*0.9*0.9*0.9*0.9*0.9 +
    4.8218*0.9*0.9*0.9*0.9*0.9*0.9;

  gammatsuji = 0.2*sqrt(meff*kn);
  damp = gammatsuji*vnnr/rsq;
  ccel = ccel - polyhertz * damp;

  // relative velocities

  vtr1 = vt1 - (dz*wr2-dy*wr3);
  vtr2 = vt2 - (dx*wr3-dz*wr1);
  vtr3 = vt3 - (dy*wr1-dx*wr2);
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);

  // shear history effects

  if (shearupdate) {
    shear[0] += vtr1*dt;
    shear[1] += vtr2*dt;
    shear[2] += vtr3*dt;
  }
  shrmag = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);

  // rotate shear displacements

  rsht = shear[0]*dx + shear[1]*dy + shear[2]*dz;
  rsht = rsht*rsqinv;
  if (shearupdate) {
    shear[0] -= rsht*dx;
    shear[1] -= rsht*dy;
    shear[2] -= rsht*dz;
  }

  // tangential forces = shear + tangential velocity damping

  fs1 = -polyhertz * (kt*shear[0] + meff*gammat*vtr1);
  fs2 = -polyhertz * (kt*shear[1] + meff*gammat*vtr2);
  fs3 = -polyhertz * (kt*shear[2] + meff*gammat*vtr3);

  kt=8.0*G_eff*a0*aovera0;

  // shear damping uses Tsuji et al form also

  fs1 = -kt*shear[0] - polyhertz*gammatsuji*vtr1;
  fs2 = -kt*shear[1] - polyhertz*gammatsuji*vtr2;
  fs3 = -kt*shear[2] - polyhertz*gammatsuji*vtr3;

  // rescale frictional displacements and forces if needed

  fs = sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
  fn = xmu * fabs(ccel*r + 2.0*Fcrit);

  if (fs > fn) {
    if (shrmag != 0.0) {
      shear[0] = (fn/fs) * (shear[0] + polyhertz*gammatsuji*vtr1/kt) -
      polyhertz*gammatsuji*vtr1/kt;
      shear[1] = (fn/fs) * (shear[1] + polyhertz*gammatsuji*vtr2/kt) -
      polyhertz*gammatsuji*vtr2/kt;
      shear[2] = (fn/fs) * (shear[2] + polyhertz*gammatsuji*vtr3/kt) -
      polyhertz*gammatsuji*vtr3/kt;
      fs1 *= fn/fs ;
      fs2 *= fn/fs;
      fs3 *= fn/fs;
    } else fs1 = fs2 = fs3 = 0.0;
  }

  // calculate twisting and rolling components of torque
  // NOTE: this assumes spheres!

  relrot1 = omega[0];
  relrot2 = omega[1];
  relrot3 = omega[2];

  // rolling velocity
  // NOTE: this assumes mondisperse spheres!

  vrl1 = -rad_eff*rinv * (relrot2*dz - relrot3*dy);
  vrl2 = -rad_eff*rinv * (relrot3*dx - relrot1*dz);
  vrl3 = -rad_eff*rinv * (relrot1*dy - relrot2*dx);
  vrlmag = sqrt(vrl1*vrl1+vrl2*vrl2+vrl3*vrl3);
  if (vrlmag != 0.0) vrlmaginv = 1.0/vrlmag;
  else vrlmaginv = 0.0;

  // bond history effects

  shear[3] += vrl1*dt;
  shear[4] += vrl2*dt;
  shear[5] += vrl3*dt;

  // rotate bonded displacements correctly

  double rlt = shear[3]*dx + shear[4]*dy + shear[5]*dz;
  rlt /= rsq;
  shear[3] -= rlt*dx;
  shear[4] -= rlt*dy;
  shear[5] -= rlt*dz;

  // twisting torque

  magtwist = rinv*(relrot1*dx + relrot2*dy + relrot3*dz);
  shear[6] += magtwist*dt;

  ktwist = 0.5*kt*(a0*aovera0)*(a0*aovera0);
  magtortwist = -ktwist*shear[6] -
    0.5*polyhertz*gammatsuji*(a0*aovera0)*(a0*aovera0)*magtwist;

  twistcrit=TWOTHIRDS*a0*aovera0*Fcrit;
  if (fabs(magtortwist) > twistcrit)
    magtortwist = -twistcrit * magtwist/fabs(magtwist);

  // rolling torque

  magrollsq = shear[3]*shear[3] + shear[4]*shear[4] + shear[5]*shear[5];
  magroll = sqrt(magrollsq);
  if (magroll != 0.0) magrollinv = 1.0/magroll;
  else magrollinv = 0.0;

  kroll = 1.0*4.0*Fcrit*pow(aovera0,1.5);
  magtorroll = -kroll*magroll - 0.1*gammat*vrlmag;

  rollcrit = 0.01;
  if (magroll > rollcrit) magtorroll = -kroll*rollcrit;

  // forces & torques

  fx = dx*ccel + fs1;
  fy = dy*ccel + fs2;
  fz = dz*ccel + fs3;

  f[0] += fx;
  f[1] += fy;
  f[2] += fz;

  tor1 = rinv * (dy*fs3 - dz*fs2);
  tor2 = rinv * (dz*fs1 - dx*fs3);
  tor3 = rinv * (dx*fs2 - dy*fs1);
  torque[0] -= radius*tor1;
  torque[1] -= radius*tor2;
  torque[2] -= radius*tor3;

  torque[0] += magtortwist * dx*rinv;
  torque[1] += magtortwist * dy*rinv;
  torque[2] += magtortwist * dz*rinv;

  torque[0] += magtorroll * (shear[4]*dz - shear[5]*dy)*rinv*magrollinv;
  torque[1] += magtorroll * (shear[5]*dx - shear[3]*dz)*rinv*magrollinv;
  torque[2] += magtorroll * (shear[3]*dy - shear[4]*dx)*rinv*magrollinv;
}

/* ---------------------------------------------------------------------- */

void FixWallGran::shm_history(double rsq, double dx, double dy, double dz,
			      double *vwall, double *v,
			      double *f, double *omega, double *torque,
			      double radius, double meff, double *shear, int i)
{
  //~ Added this function for shm history [KH - 30 October 2013]
  
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double damp,ccel,vtr1,vtr2,vtr3,vrel;
  double fs,fslim,fx,fy,fz;
  double shsqmag,shsqnew,rsht,shratio,polyhertz,rinv,rsqinv;
  double wspinx,wspiny,wspinz,shint0,shint1,shint2,omdel;

  r = sqrt(rsq);
  rinv = 1.0/r;
  rsqinv = 1.0/rsq;

  // relative translational velocity

  vr1 = v[0] - vwall[0];
  vr2 = v[1] - vwall[1];
  vr3 = v[2] - vwall[2];

  // normal component

  vnnr = vr1*dx + vr2*dy + vr3*dz;
  vn1 = dx*vnnr / rsq;
  vn2 = dy*vnnr / rsq;
  vn3 = dz*vnnr / rsq;

  // tangential component

  vt1 = vr1 - vn1;
  vt2 = vr2 - vn2;
  vt3 = vr3 - vn3;

  // normal forces = Hertzian contact
  //~ Damping is not present in the shm pairstyle

  ccel = kn*(radius-r)*rinv;
  polyhertz = sqrt((radius-r)*radius);
  ccel *= polyhertz;

  // relative velocities
  vtr1 = vt1 - dz*omega[1]+dy*omega[2];
  vtr2 = vt2 - dx*omega[2]+dz*omega[0];
  vtr3 = vt3 - dy*omega[0]+dx*omega[1];
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);

  // shear history effects
  //~ Note that shear now refers to shear force, not shear displacement

  shsqmag = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];

  // rotate shear forces onto new contact plane conserving length

  rsht = shear[0]*dx + shear[1]*dy + shear[2]*dz;
  rsht = rsht*rsqinv;
  if (shearupdate) {
    shear[0] -= rsht*dx;
    shear[1] -= rsht*dy;
    shear[2] -= rsht*dz;
    shsqnew = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];
    if (shsqnew!=0.0) {
      shratio=sqrt(shsqmag/shsqnew);
      shear[0] *= shratio; // conserve shear force length
      shear[1] *= shratio;
      shear[2] *= shratio;
    }
  }
  
  // then perform rotation for rigid-body SPIN
  omdel=omega[0]*dx+omega[1]*dy+omega[2]*dz;
  wspinx=0.5*rsqinv*dx*omdel;
  wspiny=0.5*rsqinv*dy*omdel;
  wspinz=0.5*rsqinv*dz*omdel;
  shint0 = shear[0];
  shint1 = shear[1];
  shint2 = shear[2];

  if (shearupdate) {
    shear[0]=shint0+shint1*(-wspinz*dt)+shint2*wspiny*dt;
    shear[1]=shint0*wspinz*dt+shint1+shint2*(-wspinx*dt);
    shear[2]=shint0*(-wspiny*dt)+shint1*wspinx*dt+shint2;
  }

  // tangential forces done incrementally

  if (shearupdate) {
    shear[0] -= polyhertz*kt*vtr1*dt;//shear displacement =vtr*dt
    shear[1] -= polyhertz*kt*vtr2*dt;
    shear[2] -= polyhertz*kt*vtr3*dt;
  }

  // rescale frictional forces if needed
  
  fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  fslim = xmu * fabs(ccel*r);

  if (fs > fslim) {
    if (fs != 0.0) {
      if (shearupdate) {
	shear[0] *= fslim/fs;
	shear[1] *= fslim/fs;
	shear[2] *= fslim/fs;
      }
    } else shear[0] = shear[1] = shear[2] = 0.0;
  }

  // forces & torques
  
  fx = dx*ccel + shear[0];
  fy = dy*ccel + shear[1];
  fz = dz*ccel + shear[2];

  f[0] += fx;
  f[1] += fy;
  f[2] += fz;

  torque[0] -= dy*shear[2] - dz*shear[1];
  torque[1] -= dz*shear[0] - dx*shear[2];
  torque[2] -= dx*shear[1] - dy*shear[0];
    
  //~ Call function for rolling resistance model [KH - 30 October 2013]
  double effectivekt = kt*polyhertz;
  double db[3], localdM[3], globaldM[3]; //~ Pass by reference
  if (*rolling && shearupdate)
    rolling_resistance(i,sheardim,dx,dy,dz,r,radius,ccel,fslim,
		       effectivekt,torque,shear,db,localdM,globaldM);
  //~ Call function for D_spin resistance model [MO - 30 November 2013]

  double dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy,a,N;
  a = polyhertz;
  N = ccel*r;
  if (*D_spin && shearupdate) 
    Deresiewicz1954_spin(i,sheardim,dx,dy,dz,radius,r,torque,shear,dspin_i,
			 dspin_stm,spin_stm,dM_i,dM,K_spin,theta_r,
			 M_limit,Geq,Poiseq,Dspin_energy,a,N);

  //~ Add contributions to traced energy [KH - 20 February 2014]
  double aveshearforce, slipdisp, oldshearforce, newshearforce;
  double incdissipf, nstr, sstr, incrementaldisp, rkt;
  if (pairenergy) {
    //~ Ensure rkt cannot become infinite [KH - 21 October 2014]
    /*~~ A tiny effective kt should reult in a negligile energy.rkt=0 is reasonable under this circumstance to avoid an enormous energy value [MO - 22 October 2014]~~*/
    if (effectivekt > 1.0e-30) rkt = 1.0/effectivekt;
    else rkt = 0.0;
    /*~ Increment the friction energy only if the slip condition
      is invoked*/
    oldshearforce = sqrt(shsqmag);
    if (fs > fslim && fslim > 0.0) {
      //~ current shear displacement = fslim/effectivekt;
      slipdisp = rkt*(fs-fslim);
      aveshearforce = 0.5*(oldshearforce + fslim);

      //~ slipdisp and aveshearforce are both positive
      incdissipf = aveshearforce*slipdisp;
      dissipfriction += incdissipf;
      if (trace_energy) shear[4] += incdissipf;
    }

    /*~ Update the normal contribution to strain energy which 
      doesn't need to be calculated incrementally*/
    nstr = 0.4*kn*polyhertz*(radius-r)*(radius-r);
    normalstrain += nstr;
    if (trace_energy) shear[5] = nstr;

    //~ The shear component does require incremental calculation
    if (shearupdate) {
      newshearforce = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
      incrementaldisp = rkt*(newshearforce - oldshearforce);
      sstr = 0.5*incrementaldisp*(newshearforce + oldshearforce);
      shearstrain += sstr;
      if (trace_energy) shear[6] += sstr;

      //~~ Update the spin contribution [MO - 30 November 2014]
      if (*D_spin) {
	spinenergy += Dspin_energy;
	if (trace_energy) shear[7] += Dspin_energy;
      }
    }
  }

  //~ Assign current polyhertz value to shear[3] [KH - 30 October 2013]
  shear[3] = polyhertz;

  fwall[0] += fx;
  fwall[1] += fy;
  fwall[2] += fz;
  if (evflag) ev_tally_wall(i,fx,fy,fz,dx,dy,dz,radius);
}

/* ---------------------------------------------------------------------- */

void FixWallGran::CM_history(double rsq, double dx, double dy, double dz,
			     double *vwall, double *v,
			     double *f, double *omega, double *torque,
			     double radius, double meff, double *shear, int i)
{
  //~ Added this function for CM history [MO - 18 July 2014]

  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double damp,ccel,vtr1,vtr2,vtr3,vrel;
  double fs,fslim,fx,fy,fz;
  double shsqmag,shsqnew,rsht,shratio,polyhertz,rinv,rsqinv;
  double wspinx,wspiny,wspinz,shint0,shint1,shint2,omdel;

  r = sqrt(rsq);
  rinv = 1.0/r;
  rsqinv = 1.0/rsq;

  //**********************************************************

  // relative translational velocity

  vr1 = v[0] - vwall[0];
  vr2 = v[1] - vwall[1];
  vr3 = v[2] - vwall[2];

  // normal component

  vnnr = vr1*dx + vr2*dy + vr3*dz;
  vn1 = dx*vnnr / rsq;
  vn2 = dy*vnnr / rsq;
  vn3 = dz*vnnr / rsq;

  // tangential component

  vt1 = vr1 - vn1;
  vt2 = vr2 - vn2;
  vt3 = vr3 - vn3;


  // relative velocities
  
  vtr1 = vt1 - dz*omega[1]+dy*omega[2];
  vtr2 = vt2 - dx*omega[2]+dz*omega[0];
  vtr3 = vt3 - dy*omega[0]+dx*omega[1];
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);
  
  // shear history effects
  //~ Note that shear now refers to shear force, not shear displacement
  
  shsqmag = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]; //oldshearforce 
  
  // rotate shear forces onto new contact plane conserving length
  
  rsht = shear[0]*dx + shear[1]*dy + shear[2]*dz;
  rsht = rsht*rsqinv;
  if (shearupdate) {
    shear[0] -= rsht*dx;
    shear[1] -= rsht*dy;
    shear[2] -= rsht*dz;
    shsqnew = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];
    if (shsqnew!=0.0) {
      shratio=sqrt(shsqmag/shsqnew);
      shear[0] *= shratio; // conserve shear force length
      shear[1] *= shratio;
      shear[2] *= shratio;
    }
  }
  
  // then perform rotation for rigid-body SPIN
  omdel=omega[0]*dx+omega[1]*dy+omega[2]*dz;
  wspinx=0.5*rsqinv*dx*omdel;
  wspiny=0.5*rsqinv*dy*omdel;
  wspinz=0.5*rsqinv*dz*omdel;
  shint0 = shear[0];
  shint1 = shear[1];
  shint2 = shear[2];
  
  if (shearupdate) {
    shear[0]=shint0+shint1*(-wspinz*dt)+shint2*wspiny*dt;
    shear[1]=shint0*wspinz*dt+shint1+shint2*(-wspinx*dt);
    shear[2]=shint0*(-wspiny*dt)+shint1*wspinx*dt+shint2;
  }
  
  // Contact model of CM *********************************************e
  // The current version of CM model is wrriten by [MO - 12 June  2015]
  // Note that the roughnes of wall is considered as null, i.e. RMSf_wall = 0;
  double overlap = radius-r; // special case for ball to particle
  double R_star = radius;    // special case for ball to particle
  //********************************************
  double Poiseq_p = static_cast<double>(Poiseq_p);
  double Geq_p = static_cast<double>(Geq_p);
  //double G_star = 1.0/((2.0-Poiseq)/Geq+(2.0-Poiseq_p)/Geq_p);
  double G_star = 0.5*Geq/(2.0-Poiseq);
  //double E_star = 1.0/(0.5*(1.0-Poiseq)/Geq+0.5*(1.0-Poiseq_p)/Geq_p);
  double E_star = Geq/(1.0-Poiseq);
  double xmu_p = static_cast<double>(xmu_p);
  //double xmu_mean = 0.5*(xmu + xmu_p);
  double xmu_mean = xmu;
  double RMSf_p = static_cast<double>(RMSf_p);
  //double RMSf_eq = sqrt(RMSf*RMSf + RMSf_p*RMSf_p);
  double RMSf_eq = RMSf;
  double Hp_p = static_cast<double>(Hp_p);
  //double Hp_mean = 0.5*(Hp + Hp_p);
  double Hp_mean = Hp;
  //********************************************
  double pi = 4.0*atan(1.0);
  double overlap_p1 = 0.82 * RMSf_eq;     
  double overlap_p2 = 1.24 * RMSf_eq;
  double overlap_p_sum = overlap_p1 + overlap_p2;
  double N_T200 = 100.0*RMSf_eq*E_star*sqrt(2.0*R_star*RMSf_eq);
  double N_T2   = 1.0  *RMSf_eq*E_star*sqrt(2.0*R_star*RMSf_eq);
  double overlap_T200 = pow(3.0*N_T200/(4.0*sqrt(R_star)*E_star),2.0/3.0)+overlap_p_sum; 
  double b_coeff = 2.0*E_star*sqrt(R_star*(overlap_T200-overlap_p_sum))*(overlap_T200 - overlap_p1)/N_T200;
  double overlap_T2 = (overlap_T200 - overlap_p1)*pow(N_T2/N_T200,1.0/b_coeff) + overlap_p1; 
  double c_coeff = b_coeff * (N_T200/N_T2) * overlap_T2 * pow(overlap_T200-overlap_p1,-b_coeff) * pow(overlap_T2-overlap_p1,b_coeff-1);
  double N = 0.0;
  double effectivekn = 0.0;
  double effectivekt = 0.0;
  double N_max = 0.0;
  double overlap_asperity = 0.0;
  double overlap_combined = 0.0;
  double overlap_hertz = 0.0;
  double polyhertz_effective = 0.0;
  double tolerance = 1.0e-20; 
  double alpha; // non-dimentional roughness parameter (see Johnson 1985)
  int N_step;

  if (RMSf < tolerance) {
    RMSf_eq = overlap_p1 = overlap_p2 = overlap_p_sum = N_T200 = N_T2 = overlap_T200 = overlap_T2 = b_coeff = c_coeff = 0.0;
  }

  // Historical parameters for plastic version of CM model 
  //double overlap_max = fabs(shear[3]);
  //double energy_asperity_max = fabs(shear[4]);
	
  if (overlap <= overlap_T2) {
    N_step = 11;
    overlap_asperity = overlap;
    N = N_T2 * pow(overlap_asperity / overlap_T2,c_coeff);
    effectivekn = c_coeff * (N_T2/overlap_T2) * pow(overlap_asperity / overlap_T2, c_coeff-1.0);  
    ccel = N*rinv;
  }
  else if ((overlap > overlap_T2) && (overlap <= overlap_T200)) {
    N_step = 12;
    overlap_combined = overlap - overlap_p1;  
    N = N_T200 * pow(overlap_combined / (overlap_T200 - overlap_p1), b_coeff);
    effectivekn = b_coeff * N_T200/(overlap_T200 - overlap_p1) * pow(overlap_combined / (overlap_T200-overlap_p1), b_coeff-1.0); 
    ccel = N*rinv;
  }
  else {
    N_step = 13;
    overlap_hertz = overlap - overlap_p_sum;
    N = 4.0/3.0*E_star*sqrt(R_star)*pow(overlap_hertz,1.5);
    effectivekn = 2.0 * E_star * sqrt(R_star*overlap_hertz);
    ccel = N*rinv;
  }
	  
  polyhertz = sqrt(overlap*R_star); // equivalent radius of contact
  // radius of contact assuming Hertzian contact
  if (polyhertz != 0.0) alpha = R_star * RMSf_eq / (polyhertz*polyhertz);
  polyhertz_effective = pow(3.0/4.0 * N * R_star / E_star, 1.0/3.0);
  effectivekt = 2.0*(1.0-Poiseq)/(2.0-Poiseq)*effectivekn;
	
  if (shearupdate) {
    shear[0] -= effectivekt*vtr1*dt;//shear displacement =vtr*dt
    shear[1] -= effectivekt*vtr2*dt;
    shear[2] -= effectivekt*vtr3*dt;
  }

  // rescale frictional forces if needed

  fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  fslim = xmu_mean * fabs(ccel*r);

  if (fs > fslim) {
    if (fs != 0.0) {
      if (shearupdate) {
	shear[0] *= fslim/fs;
	shear[1] *= fslim/fs;
	shear[2] *= fslim/fs;
      }
    } else shear[0] = shear[1] = shear[2] = 0.0;
  }

  // forces & torques

  fx = dx*ccel + shear[0];
  fy = dy*ccel + shear[1];
  fz = dz*ccel + shear[2];

  f[0] += fx;
  f[1] += fy;
  f[2] += fz;

  torque[0] -= dy*shear[2] - dz*shear[1];
  torque[1] -= dz*shear[0] - dx*shear[2];
  torque[2] -= dx*shear[1] - dy*shear[0];

  //~ Add contributions to traced energy [KH - 20 February 2014]
  double aveshearforce, slipdisp, oldshearforce, newshearforce;
  double incdissipf, nstr, sstr, incrementaldisp, rkt;
  double b1inv = 1.0/(b_coeff+1.0);
  double c1inv = 1.0/(c_coeff+1.0);
  double nstr_asperity, nstr_combined, nstr_hertz;
	
  double dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy,a;
  a = polyhertz_effective;

  if (*D_spin && shearupdate) 
    Deresiewicz1954_spin(i,sheardim,dx,dy,dz,radius,r,torque,shear,dspin_i,
			 dspin_stm,spin_stm,dM_i,dM,K_spin,theta_r,
			 M_limit,Geq,Poiseq,Dspin_energy,a,N);

  if (pairenergy) {
    
    /*~ Increment the friction energy only if the slip condition
      is invoked*/
    oldshearforce = sqrt(shsqmag);
    if (fs > fslim && fslim > 0.0) {
      //~ current shear displacement = fslim/effectivekt;
      //~~ Added to avoid enormous energy value [MO - 22 October 2014]
      if (effectivekt > tolerance) slipdisp = (fs-fslim)/effectivekt;
      else slipdisp = 0.0;
      aveshearforce = 0.5*(oldshearforce + fslim);
      //~ slipdisp and aveshearforce are both positive
      incdissipf = aveshearforce*slipdisp;
      dissipfriction += incdissipf;
      if (trace_energy) shear[5] += incdissipf;
    }

    /* CM model includes plastic energy due to asperity crushing. Only elastic component is 
       stored in nstr, while the other is added in incrementaldisp [MO 19 January 2015]*/
    if (N_step == 11) {
      nstr_asperity = c1inv * N * overlap_asperity;
      nstr = nstr_asperity;
    }
    else if (N_step == 12) {  
      nstr_asperity = c1inv * N_T2 * overlap_T2;
      //nstr_combined = b1inv*N_T200*(pow(overlap_combined,b_coeff+1.0)*pow(overlap_T200 - overlap_p1,-b_coeff) 
      //				  - pow(overlap_T2-overlap_p1,b_coeff+1.0)*pow(overlap_T200-overlap_p1,b_coeff));
      nstr_combined = b1inv * (overlap_combined * N - (overlap_T2-overlap_p1) * N_T2);
      nstr = nstr_asperity + nstr_combined;
    }	    
    else {   
      nstr_asperity = c1inv * N_T2 * overlap_T2;
      //nstr_combined = b1inv * N_T200 * ((overlap_T200 - overlap_p1) 
      //				      - pow(overlap_T2-overlap_p1,b_coeff+1.0)*pow(overlap_T200-overlap_p1,b_coeff));
      nstr_combined = b1inv * ((overlap_T200 - overlap_p1) * N_T200 - (overlap_T2-overlap_p1) * N_T2);
      nstr_hertz = 0.4 * 4.0/3.0 * E_star * sqrt(R_star) * (pow(overlap_hertz,2.5) - pow(overlap_T200-overlap_p_sum,2.5));
      nstr = nstr_asperity + nstr_combined + nstr_hertz;
    }
	  
    normalstrain += nstr;
    if (trace_energy) shear[6] = nstr;
    
    //~~ Update the spin contribution [MO - 13 November 2014]
    if (*D_spin) {
      spinenergy += Dspin_energy;
      if (trace_energy) shear[8] += Dspin_energy;
    }

    //~ The shear component does require incremental calculation
    if (shearupdate) {
      newshearforce = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
      //~~ Added to avoid enormous energy value [MO 22 October 2014]
      if (effectivekt > 1.0e-30) incrementaldisp = (newshearforce - oldshearforce)/effectivekt;   
      else incrementaldisp = 0.0; 
      // because no incremental shear force [MO - 21 July 2014]
      sstr = 0.5*incrementaldisp*(newshearforce + oldshearforce);
      shearstrain += sstr;
      if (trace_energy) shear[7] += sstr;
    }
  }

  /*if (shearupdate && i == 7 && update->ntimestep == 1) {
    fprintf(screen,"timestep %i tag %i & %i overlap_p_sum %1.6e N_T200 %1.6e N_T2 %1.6e overlap_T200 %1.6e overlap_T2 %1.6e b_coeff %1.6e c_coeff %1.6e\n",update->ntimestep,i,i,overlap_p_sum,N_T200,N_T2,overlap_T200,overlap_T2,b_coeff,c_coeff); 

    }
    if (shearupdate && i == 7 && update->ntimestep % 1000 == 0) {
    fprintf(screen,"timestep %i tag %i & %i N_step %i overlap %1.6e N %1.6e kn %1.6e kt %1.6e poly_eff %1.6e poly %1.6e alpha %1.6e\n",update->ntimestep,i,i,N_step,overlap,N,effectivekn,effectivekt,polyhertz_effective,polyhertz,alpha);
    }*/
  
  /*************************************************************************
  if (shearupdate) {
    shear[3] = overlap_max;
    if (nstr_plast > energy_asperity_max) shear[4] = nstr_plast;
  }
  //*************************************************************************/
  
  fwall[0] += fx;
  fwall[1] += fy;
  fwall[2] += fz;

  if (evflag) ev_tally_wall(i,fx,fy,fz,dx,dy,dz,radius);
}

/* ---------------------------------------------------------------------- */

void FixWallGran::HMD_history(double rsq, double dx, double dy, double dz,
			      double *vwall, double *v,
			      double *f, double *omega, double *torque,
			      double radius, double meff, double *shear, int i)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double damp,ccel,vtr1,vtr2,vtr3,vrel;
  double fs,fslim,fx,fy,fz;
  double shsqmag,shsqnew,rsht,shratio,polyhertz,rinv,rsqinv;
  double wspinx,wspiny,wspinz,shint0,shint1,shint2,omdel;

  r = sqrt(rsq);
  rinv = 1.0/r;
  rsqinv = 1.0/rsq;

  // relative translational velocity

  vr1 = v[0] - vwall[0];
  vr2 = v[1] - vwall[1];
  vr3 = v[2] - vwall[2];

  // normal component

  vnnr = vr1*dx + vr2*dy + vr3*dz;
  vn1 = dx*vnnr / rsq;
  vn2 = dy*vnnr / rsq;
  vn3 = dz*vnnr / rsq;

  // tangential component

  vt1 = vr1 - vn1;
  vt2 = vr2 - vn2;
  vt3 = vr3 - vn3;

  // relative velocities

  vtr1 = vt1 - dz*omega[1]+dy*omega[2];
  vtr2 = vt2 - dx*omega[2]+dz*omega[0];
  vtr3 = vt3 - dy*omega[0]+dx*omega[1];
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);

  // shear history effects
  //~ Note that shear now refers to shear force, not shear displacement

  shsqmag = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];

  // rotate shear forces onto new contact plane conserving length

  rsht = shear[0]*dx + shear[1]*dy + shear[2]*dz;
  rsht = rsht*rsqinv;
  if (shearupdate) {
    shear[0] -= rsht*dx;
    shear[1] -= rsht*dy;
    shear[2] -= rsht*dz;
    shsqnew = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];
    if (shsqnew!=0.0) {
      shratio=sqrt(shsqmag/shsqnew);
      shear[0] *= shratio; // conserve shear force length
      shear[1] *= shratio;
      shear[2] *= shratio;
    }
  }

  // then perform rotation for rigid-body SPIN
  omdel=omega[0]*dx+omega[1]*dy+omega[2]*dz;
  wspinx=0.5*rsqinv*dx*omdel;
  wspiny=0.5*rsqinv*dy*omdel;
  wspinz=0.5*rsqinv*dz*omdel;
  shint0 = shear[0];
  shint1 = shear[1];
  shint2 = shear[2];

  if (shearupdate) {
    shear[0]=shint0+shint1*(-wspinz*dt)+shint2*wspiny*dt;
    shear[1]=shint0*wspinz*dt+shint1+shint2*(-wspinx*dt);
    shear[2]=shint0*(-wspiny*dt)+shint1*wspinx*dt+shint2;
  }
  
  //*************************************************************************************
  /* The main part of the HMD model is written down below as of 13th August 2014 [MO]*/ 
  //*************************************************************************************

  double tolerance = 1.0e-20;
  double xmu_p = static_cast<double>(xmu_p);
  //double xmu_mean = 0.5*(xmu + xmu_p);
  double xmu_mean = xmu;

  if (THETA1 == 0 && xmu_mean > tolerance) {
    
    // rotate shear displacement onto new contact plane conserving length
    double shdsq_mag,shdsq_new,rshd,shdratio,shint5,shint6,shint7;
    shdsq_mag = shear[5]*shear[5] + shear[6]*shear[6] + shear[7]*shear[7];
    rshd = shear[5]*dx + shear[6]*dy + shear[7]*dz;
    rshd *= rsqinv;
    if (shearupdate) {
      shear[5] -= rshd*dx;
      shear[6] -= rshd*dy;
      shear[7] -= rshd*dz;
      shdsq_new = shear[5]*shear[5] + shear[6]*shear[6] + shear[7]*shear[7];
      if (shdsq_new!=0.0) {
	shdratio=sqrt(shdsq_mag/shdsq_new);
	shear[5] *= shdratio; // conserve shear force length
	shear[6] *= shdratio;
	shear[7] *= shdratio;
      }
    }
    // then perform rotation for rigid-body SPIN
    shint5 = shear[5];
    shint6 = shear[6];
    shint7 = shear[7];
    if (shearupdate) {
      shear[5]=shint5+shint6*(-wspinz*dt)+shint7*wspiny*dt;
      shear[6]=shint5*wspinz*dt+shint6+shint7*(-wspinx*dt);
      shear[7]=shint5*(-wspiny*dt)+shint6*wspinx*dt+shint7;
    }

    ////////////////// 
    // rotate shear displacement onto new contact plane conserving length
    // Added for Tdisp_star1 [MO - 02 April 2015]
    double shdsq_mag_star1,shdsq_new_star1,rshd_star1,shdratio_star1,shint22,shint23,shint24;
    shdsq_mag_star1 = shear[22]*shear[22] + shear[23]*shear[23] + shear[24]*shear[24];
    rshd_star1 = shear[22]*dx + shear[23]*dy + shear[24]*dz;
    rshd_star1 *= rsqinv;
    if (shearupdate) {
      shear[22] -= rshd_star1*dx;
      shear[23] -= rshd_star1*dy;
      shear[24] -= rshd_star1*dz;
      shdsq_new_star1 = shear[22]*shear[22] + shear[23]*shear[23] + shear[24]*shear[24];
      if (shdsq_new_star1!=0.0) {
	shdratio_star1=sqrt(shdsq_mag_star1/shdsq_new_star1);
	shear[22] *= shdratio_star1; // conserve shear force length
	shear[23] *= shdratio_star1;
	shear[24] *= shdratio_star1;
      }
    }
    // then perform rotation for rigid-body SPIN
    shint22 = shear[22];
    shint23 = shear[23];
    shint24 = shear[24];
    if (shearupdate) {
      shear[22]=shint22+shint23*(-wspinz*dt)+shint24*wspiny*dt;
      shear[23]=shint22*wspinz*dt+shint23+shint24*(-wspinx*dt);
      shear[24]=shint22*(-wspiny*dt)+shint23*wspinx*dt+shint24;
    }
  }

  //*****************************************************************************************
  /*To avoid the reverse of sign of shearquantities, 'fabs' is used for absolute quantities.
    Attention should be paied for some flags which have 1 or -1 values. For these cases, the 
    quantities were stored as 10000 instead of -1 and called with 'fabs', and changed to -1 
    again at the next step.[MO - 15 December 2014] */
  //*****************************************************************************************
  // To avoid numerical drift, int "skip" is used in this model. 
  // skip = 0; usual case
  // skip = 1; skip the entire calculation about tangential component
  // if xmu_mean < tolerance or dTdisp < tolerance && T_step == 0, i.e. begining of simulation.
  // skip = 2; skip partially but update, for example, T* due to change of normal force
  // if dTdisp < thereshold. Please find an appropriate threshold. [MO - 20 January 2015]
  //*****************************************************************************************
  // normal force = Hertzian contact 
  double R_star = radius;
  //********************************************
  double Poiseq_p = static_cast<double>(Poiseq_p);
  double Geq_p = static_cast<double>(Geq_p);
  //double G_star = 1.0/((2.0-Poiseq)/Geq+(2.0-Poiseq_p)/Geq_p);
  double G_star = 1.0/((2.0-Poiseq)/Geq+(2.0-Poiseq)/Geq);
  //double E_star = 1.0/(0.5*(1.0-Poiseq)/Geq+0.5*(1.0-Poiseq_p)/Geq_p);
  double E_star = 1.0/(0.5*(1.0-Poiseq)/Geq+0.5*(1.0-Poiseq)/Geq);
  //********************************************
  double pi = 4.0*atan(1.0);                    // pi = 3.1415*******
  double overlap = radius-r;                  
  double overlap_old = fabs(shear[3]);                // overlap of spheres at the previous step 
  double a = sqrt(R_star*overlap);              // radius of contact circle
  double a_old = sqrt(R_star*overlap_old);      // radius of contact circle at the previous step 
  double N = kn*overlap*a;                      // normal force
  double N_old = kn*overlap_old*a_old;          // normal force at the previous step
  double dN = N - N_old;                        // change of normal force
  double K_N = 2.0*E_star*a;                    // normal contact stiffness (tangential, not secant)
  polyhertz = a;                                // re-use polyhertz to adjust the format for shm contact model
  ccel = kn*overlap*a*rinv;                     // previously, ccel = kn*overlap*polyhert*rinv;	
  //************************************************************************************************
  // The following codes introduce HMD model for tangential component.
  double T = 0.0;                               // bulk tangential force
  double T_old = shear[12];                     // bulk tangential force at the previous step (including its sign)
  double T_star1 = shear[8];                    // T*:  tangential force at the revasal step from loading to unloading 
  double T_star2 = shear[9];                    // T**: tangential force at the revasal step from unloading to re-loading 
  double Tdisp  = 0.0;                          // bulk tangential displacement	
  double Tdisp_mag = 0.0;                        // magnitude of bulk tangential displacement
  double Tdisp_old  = shear[4];                 // bulk tangential displacement at the previous step
  double dTdisp = 0.0;                          // increment of bulk displacement
  double Tdisp1 = 0.0;                          // tangential displacement of x direction
  double Tdisp2 = 0.0;                          // tangential displacement of y direction
  double Tdisp3 = 0.0;                          // tangential displacement of z direction  
  double Tdisp1_old = shear[5];                 // tangential displacement of x direction at the previous step
  double Tdisp2_old = shear[6];                 // tangential displacement of y direction at the previous step
  double Tdisp3_old = shear[7];                 // tangential displacement of z direction at the previous step     
  double inner_product = 0.0;                   // inter product between current and previous vectors of tangential displacement
  int T_step = 0;                               // the current loading phase, e.g. 11 = N increasing T increasing
  int T_step_old = static_cast<int>(fabs(shear[15])); // the loading case at the previous step
  int slip_T = 0;
  int UFL = 0;                                  // UFL is the direction of tangential load
  int CTD = static_cast<int>(fabs(shear[14]));         // CTD is the direction of tangential displacement 
  if (CTD == 10000) CTD = -1;
  int CDF = static_cast<int>(fabs(shear[11]));         // CDF is the direction of tangential load-displacement system
  if (CDF == 10000) CDF = -1;
  double theta1 = 1.0;
  double theta2 = 1.0;
  double theta3 = 1.0;
  double theta_t = 1.0;
  double Tdisp1_star1 = shear[22];     // tangential displacement of x direction at T_star1
  double Tdisp2_star1 = shear[23];     // tangential displacement of y direction at T_star1
  double Tdisp3_star1 = shear[24];     // tangential displacement of z direction at T_star1
  double Tdisp_star1_mag;
	

  fslim = xmu_mean * N;   // N = ccel*r 
  int    T_step_group = 0;
  int    T_step_group_old = 0;
  double Tdisp_DD = 0.0;
  double K_T = 0.0;       // tangential contact stiffness
  double dT = 0.0;        // increment of tangential contact force
  double fs_old = sqrt(shsqmag);
  double dTdisp_mag = sqrt(vtr1*vtr1+vtr2*vtr2+vtr3*vtr3)*dt; 
	

  if (xmu_mean < tolerance) {
    K_T = 0.0;
    T_step = 0;
    CTD = 1;
    CDF = 1;
    T_step = 0; 
    Tdisp1 = Tdisp2 = Tdisp3 = Tdisp = dTdisp = dTdisp_mag = 0.0;
    T = shear[0] = shear[1] = shear[2] = 0.0;
  } 
  else if (THETA1 == 1) {
    K_T = 8.0*G_star*a;
    T_step = 1;
    CTD = 1;
    CDF = 1;
    dTdisp = dTdisp_mag = 0.0;
    // update T & Tdisps later
  } 
  else if (dTdisp_mag < 1.0e-15) {
    // need to update historical parameters
    K_T = 0.0;
    T_step = T_step_old;
    T = T_old;
    Tdisp1 = Tdisp1_old;          
    Tdisp2 = Tdisp2_old;
    Tdisp3 = Tdisp3_old;
    Tdisp = Tdisp_old;
    dTdisp = dTdisp_mag = 0.0;
  }
  else {

    //******************************************************************
    // First, calculate the shear displacement
    // Skip the follwing calculation if xmu_mean or dTdisp is small
    //******************************************************************
                
    Tdisp1 = Tdisp1_old + vtr1*dt;          
    Tdisp2 = Tdisp2_old + vtr2*dt;
    Tdisp3 = Tdisp3_old + vtr3*dt;
    Tdisp_mag = sqrt(Tdisp1*Tdisp1 + Tdisp2*Tdisp2 + Tdisp3*Tdisp3);
    
    // inner_product becomes negative if tangential disp. becomes negative.
    inner_product = Tdisp1*Tdisp1_star1 + Tdisp2*Tdisp2_star1 + Tdisp3*Tdisp3_star1;
    Tdisp_star1_mag = sqrt(Tdisp1_star1*Tdisp1_star1 + Tdisp2_star1*Tdisp2_star1 + Tdisp3_star1*Tdisp3_star1);
    
    if (Tdisp_mag < tolerance && fabs(Tdisp_old) < tolerance && T_step_old == 0) {
      CTD = 1;                                     // when there is no shear disp at all.
      CDF = 1;
    }
    else if (Tdisp_mag > tolerance && fabs(Tdisp_old) < tolerance && T_step_old == 0) {
      // this is for the first increment of tangential displacement
      CTD = 1;       
      CDF = 1;
    }
    else if (CDF == 1 && (Tdisp_star1_mag < tolerance))  CTD = 1;
    else if (inner_product >= 0.0)                       CTD = 1;
    else                                                 CTD = -1; 
    
    Tdisp = CDF * CTD * Tdisp_mag; // CDF is added here [MO - 02 April 2015]    

    // give the sign for tangential displacement based on the value of the inner product 
    dTdisp = Tdisp - Tdisp_old;   // dTdisp includes direction of tangential displacement (not magnitude)
    

    //*******************************************************************************************
    // Update the Tdisp_DD
  
    /* Fig.7 of Mindlin & Deresiewicz (1953) explain why this special case is needed.
       Tdisp_DSC is the minimum tangential displacement to move onto a new loading curve of N+dN.
       If the increment of the tangential displacement is less than Tdisp_DSC, the resultant tangential force
       is less than theoretical value.
       The maximum tangential contact stiffness (8.0*G_star*a) is used until the current tangential force 
       catches up the the new loading curve. */
    //**********************************************************************************************
    double Tdisp_DSC = 0.0;
    if (a > tolerance) Tdisp_DSC = xmu_mean*dN/(8.0*G_star*a);  // sign of Tdisp_DSC depends upon sign of dN;

    Tdisp_DD = fabs(shear[13]) - fabs(dTdisp) + Tdisp_DSC;
    if (dN < 0) Tdisp_DD = fabs(shear[13]) + Tdisp_DSC;
    int    special_DD = 0; 
    if (Tdisp_DD < 0.0) Tdisp_DD = 0.0;       // Tdisp_DD <= 0.0 should be satisfied to move onto a new loading curve for N+dN.
    else                special_DD = 1;       // Special case for DD value is now active               
    //**********************************************************************************************
  
    // Identify the loading steps for tangential component
    // for the first step of the special case
    if (special_DD == 1){
      theta_t = 1.0;
      if (dN > 0.0 && T_step_old != 115 && T_step_old != 125 && T_step_old != 135 && T_step_old != 145){
	if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance)          T_step = 115;
	else if (CDF*dTdisp <  0.0 && fabs(T_star2) < tolerance)                                  T_step = 125; 
	else if (CDF*dTdisp >= 0.0 && fabs(T_old) <= fabs(T_star1))                                   T_step = 135;
	else if (CDF*dTdisp <  0.0 && fabs(T_old) <= fabs(T_star1) && fabs(T_star2) >= tolerance)      T_step = 145;
	else T_step = T_step_old;
	//else fprintf(screen,"timestep %i Unexpected case occurred in zone C (HMD-wall-ball). ERROR!!\n",update->ntimestep);
	// if the special case was invoked in the previous step and is still active
      }else if ((T_step_old == 115 || T_step_old == 135) && CDF*dTdisp >= 0.0){  
	// still in step_115 or 135; if moved to unloading of T, go to the usual case
	T_step = T_step_old;
      }else if ((T_step_old == 125 || T_step_old == 145) && CDF*dTdisp < 0.0){   
	  // still in step_125 or 145; if moved to re-loading of T, go to the usual case
	T_step = T_step_old;                                           
      }else if (T_step_old == 115 && CDF*dTdisp < 0.0){
	// the first step from loading of special case to unloading of special case.
	T_step = 125;
      }else if (T_step_old == 125 && CDF*dTdisp >= 0.0){
	// the first step from unloading of special case to reloading of special case.
	T_step = 135;
      }else if (T_step_old == 135 && CDF*dTdisp < 0.0){
	// the first step from reloading  of special case to re-unloading of special case.
	T_step = 145;
	}else if (T_step_old == 145 && CDF*dTdisp > 0){
	// the first step from re-unloading  of special case to re-reloading of special case.
	T_step = 135;
      }
      else T_step = T_step_old;  
      
      //*************************************************************************************
    }else {  // Usual cases 
      // T loading: T_step = *1 
      if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance) { 
	theta1 = 1.0-(CDF*T_old+xmu_mean*dN)/(xmu_mean*N);	
	if (theta1 <= 0.0) theta1 = theta_t = tolerance;
	else theta_t = pow(theta1, 1.0/3.0);
	if       (dN >  tolerance)      T_step = 11;   // N increasing
	else if  (dN < -tolerance)      T_step = 21;   // N decreasing
	else                            T_step =  1;   // N constant
      }
      // T unloading: T_step = *2 
      else if (CDF*dTdisp < 0.0 && fabs(T_star2) < tolerance) {
	  // For the first step from loading to unloading, T* is still null, which should be T* = T_old.
	if  (fabs(T_star1) > tolerance) theta2 = 1.0-(CDF*(T_star1-T_old)+2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	else                           theta2 = 1.0-(2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	if (theta2 <= 0.0) theta2 = theta_t = tolerance;	  
	else               theta_t  = pow(theta2, 1.0/3.0);
	if      (dN >  tolerance) T_step = 12; 
	else if (dN < -tolerance) T_step = 22;      
	else                      T_step =  2;
      }
      // T re-loading: T_step = *3
      else if (CDF*dTdisp >= 0.0 && fabs(T_old) <= fabs(T_star1)) { 
	// For the first step from unloading to re-loading, T** is still null, which should be T** = T_old.
	if  (fabs(T_star2) > tolerance) theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	else                           theta3 = 1.0-(2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	if (theta3 <= 0.0) theta3 = theta_t = tolerance;
	else               theta_t  = pow(theta3, 1.0/3.0);
	if      (dN >  tolerance) T_step = 13; 
	else if (dN < -tolerance) T_step = 23;     
	else                      T_step = 3;
      }
      // T re-unloading: T_step = *4
      else if (CDF*dTdisp < 0.0 && fabs(T_old) <= fabs(T_star1) && fabs(T_star2) >= tolerance) { 
	// This part is same with the above (T_step = *3) due to simplification.
	theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	if (theta3 <= 0.0) theta3 = theta_t = tolerance;
	else               theta_t  = pow(theta3, 1.0/3.0);
	if      (dN >  tolerance) T_step = 14;
	else if (dN < -tolerance) T_step = 24; 
	else                      T_step =  4;      
      }
      else {
	T_step = T_step_old;
	theta_t = 1.0;
	//else fprintf(screen,"timestep %i Unexpected case occurred in zone C (HMD-wall-ball). ERROR!!\n",update->ntimestep);
      }
    } 
    
    //********************************************************************************************
    // Classify the groups [MO - 19 March 2015]
    if      (T_step == 0)                                                  T_step_group = 0;
    else if (T_step == 1 || T_step == 11 || T_step == 21 || T_step == 115) T_step_group = 1;
    else if (T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) T_step_group = 2;
    else if (T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) T_step_group = 3;
    else if (T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) T_step_group = 4;
    else                                                                   T_step_group = 5; // error
    // Classify the previos groups
    if      (T_step_old == 0)                                                              T_step_group_old = 0;
    else if (T_step_old == 1 || T_step_old == 11 || T_step_old == 21 || T_step_old == 115) T_step_group_old = 1;
    else if (T_step_old == 2 || T_step_old == 12 || T_step_old == 22 || T_step_old == 125) T_step_group_old = 2;
    else if (T_step_old == 3 || T_step_old == 13 || T_step_old == 23 || T_step_old == 135) T_step_group_old = 3;
    else if (T_step_old == 4 || T_step_old == 14 || T_step_old == 24 || T_step_old == 145) T_step_group_old = 4;
    else                                                                                   T_step_group_old = 5; // error

    //***********************************************************************************
    //UFL is neccesary for the calculation of tangential contact stiffness
    if (T_step_group == 2 || T_step_group == 4) UFL = -1; // Added [MO - 21 Dec 2014]
    else                                        UFL =  1;
    //***********************************************************************************
    
    // Calculate the tangential contact stiffness and the resultant tangential contact force 
    if (theta_t > 1.0) theta_t = 1.0;
    if (fabs(dTdisp) > 1.0e-15) K_T = 8.0*G_star*theta_t*a + CDF*UFL*xmu_mean*(1.0-theta_t)*dN/dTdisp;
    else K_T = 0.0;
    dT = K_T * dTdisp; 
    T = dT + T_old;
    // rescale tangential force if full sliding takes place 	
    if (fabs(T) > fslim) {
      if (fabs(T) != 0.0) {
	slip_T = 1;
	T *= fslim/fabs(T);
      } 
    }
  } // End of If THETA = 0


  // update the tangential contact force by distributing dT into 3 components based on imcrese of relative rotational velosity
  if (shearupdate) {
    shear[0] -= K_T * vtr1 * dt;  // -= K_T * dTdisp1;
    shear[1] -= K_T * vtr2 * dt;  // -= K_T * dTdisp2;
    shear[2] -= K_T * vtr3 * dt;  // -= K_T * dTdisp3;
  }
   
  // fabs(T) and fs are not always same for 3D simulation [MO - 03 April 2015] 
  fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  
  if (fs > fslim) {
    if (fs != 0.0 && xmu_mean > tolerance) {
      slip_T = 1;
      if (shearupdate) {
	shear[0] *= fslim/fs; 
	shear[1] *= fslim/fs;
	shear[2] *= fslim/fs;   
      } 
    } else shear[0] = shear[1] = shear[2] = 0.0;
  } 
    
  double fs_new = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  if (fs_new > fslim) fs_new = fslim;

  double slip_degree = fs/fslim;
  if (THETA1 == 1 && xmu_mean > tolerance) {
    if (K_T > tolerance && fs_new > tolerance) {
      // consider as a virgin tangential loading
      T = fs_new;
      if (slip_degree >= 1.0)      theta_t = 0.0; //Tdisp = 1.5 * fslim / K_T; 
      else if (slip_degree < 0.0)  theta_t = 1.0; //Tdisp = 0.0; 
      else                         theta_t = pow((1.0-slip_degree),1.0/3.0); 
      Tdisp = 1.5 * fslim / K_T * (1.0 - theta_t*theta_t);
      Tdisp1 = Tdisp * shear[0]/fs_new;
      Tdisp2 = Tdisp * shear[1]/fs_new;
      Tdisp3 = Tdisp * shear[2]/fs_new;    

      /*fprintf(screen,"timestep %i tag %i Kt %1.6e T %1.6e shear[0] %1.6e shear[1] %1.6e shear[2] %1.6e Tdisp %1.6e Tdisp1 %1.6e Tdisp2 %1.6e Tdisp3 %1.6e overlap %1.6e a %1.6e N %1.6e fslim %1.6e slip_degree %1.6e theta_t %1.6e \n",update->ntimestep,i,K_T,T,shear[0],shear[1],shear[2],Tdisp,Tdisp1,Tdisp2,Tdisp3,overlap,a,N,fslim,slip_degree,theta_t);*/

    }
    else T = Tdisp = Tdisp1 = Tdisp2 = Tdisp3 = 0.0;
  }


  // forces & torques

  fx = dx*ccel + shear[0];
  fy = dy*ccel + shear[1];
  fz = dz*ccel + shear[2];

  f[0] += fx;
  f[1] += fy;
  f[2] += fz;

  torque[0] -= dy*shear[2] - dz*shear[1];
  torque[1] -= dz*shear[0] - dx*shear[2];
  torque[2] -= dx*shear[1] - dy*shear[0];
  
  double nstr,sstr,incdissipf;
  double dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy;

  //~~ Call function for twisting resistance model [MO - 04 November 2014]   
  if (*D_spin && shearupdate) 
    Deresiewicz1954_spin(i,sheardim,dx,dy,dz,radius,r,torque,shear,dspin_i,
			 dspin_stm,spin_stm,dM_i,dM,K_spin,theta_r,
			 M_limit,Geq,Poiseq,Dspin_energy,a,N);
  
  //~ Add contributions to traced energy [KH - 20 February 2014]
  if (pairenergy) {
    /*~ Update the normal contribution to strain energy which 
      doesn't need to be calculated incrementally*/
    nstr = 0.4*kn*a*overlap*overlap;                             // peviously, nstr = 0.4*kn*polyhertz*deltan*deltan;
    normalstrain += nstr;
    if (trace_energy) shear[27] = nstr;

    
    //************
    double effectivekt = 8.0*G_star*a;
    if (xmu_mean > tolerance) {
      if (effectivekt > tolerance) incdissipf = 0.5*fs_new*fs_new/effectivekt; // this is "SHEAR STRAIN" not frictional dissipation
      else incdissipf = 0.0;
      dissipfriction += incdissipf;
      if (trace_energy) shear[26] = incdissipf; // not accumulated
    }
    //************
    
    
    if (shearupdate) {
      //~~ Update the spin contribution [MO - 13 November 2014]
      if (*D_spin) {
	spinenergy += Dspin_energy;
	if (trace_energy) shear[29] += Dspin_energy;
      }
      /* Full sliding can be considered as accumulation of partial slip for HMD model.
	 Theoretically speaking, full sliding does not take place for HMD modle.
	 Thus, shear strain energy here is summation of shear strain energy + friction energy for shm model.*/  
      if (xmu > tolerance) {
	if (THETA1 == 1) {
	  if (fabs(K_T) > tolerance) sstr = 0.5*(fs_new + fs_old)*(fs - fs_old)/K_T;
	  else sstr = 0.0;
	}
	else if (THETA1 == 0) {
	  if (T < 0) fs_new *= -1.0; 
	  if (T_old < 0) fs_old *= -1.0; 
	  if (dTdisp < 0) dTdisp_mag *= -1.0; // not fabs(dTdisp)!!
	  sstr = 0.5*dTdisp_mag*(fs_new + fs_old);
	}
	else sstr = 0.0;
       	shearstrain += sstr;
	if (trace_energy) shear[28] += sstr;
      }
    }
  }
  //******************************************************************************************
  // Update the T_star1 and T_star2 considering the change of the normal contact load
  int sp_Tstar1 = 0;
  if (THETA1 == 0 && xmu_mean > tolerance) {
    if (T_step_group != 1 && fabs(T_star1) > tolerance) T_star1 += CDF*xmu_mean*dN;
    if (T_step_group == 3 || T_step_group == 4)         T_star2 -= CDF*xmu_mean*dN;
    //****************************************************************************************
    // Update T_star1 and T_star2
    if (T_step_group == 1) T_star1 = T_star2 = 0.0;
    else if (T_step_group == 2 && T_step_group_old == 1) {
      // the first step from T_step = 1: loading to unloading
      T_star1 = T;
      Tdisp_DD = 0.0;
      sp_Tstar1 = 1;
      Tdisp1_star1 = Tdisp1;     
      Tdisp2_star1 = Tdisp2;     
      Tdisp3_star1 = Tdisp3;    
      if (T*Tdisp < 0) {
	Tdisp1_star1 *= - 1; //[MO - 01 Jun 2015] 
	Tdisp2_star1 *= - 1; //[MO - 01 Jun 2015] 
	Tdisp3_star1 *= - 1; //[MO - 01 Jun 2015] 
      } 
    }
    else if (T_step_group == 3 && T_step_group_old == 2) {
      // the first step from T_step = 2: unloading to re-loading
      T_star2 = T;
      Tdisp_DD = 0.0;
      sp_Tstar1 = 1;
      if (fabs(T_star2) > fabs(T_star1)) { // avoid initial flactuations [MO - 01 Jun 2015]
	T_star1 = T_star2 = 0.0;
	Tdisp1_star1 = Tdisp2_star1 = Tdisp3_star1 = 0.0; //[MO - 01 Jun 2015] 
      }
    } 
    //****************************************************************************************
    if ((T_step_group == 2 || T_step_group == 3 || T_step_group == 4) && sp_Tstar1 == 0) {
      // Introduce the following rules to avoid numerical errors [MO - 30 March 2015]
      // [1] CDF*(T*) > CDF*T > CDF*(T**), [2] |T*| > |T**| 
      if (fabs(T) > fabs(T_star1) && (T_step_group == 2 || T_step_group == 3 || T_step_group == 4)) {
	if (UFL == 1 && T*T_star1 >= 0.0) T_star1 = T_star2 = 0.0;
	else if (UFL ==  1 && T*T_star1 <  0.0) T_star1 = - T;
	else if (UFL == -1 && T*T_star1 >= 0.0) T_star1 =   T;
	else if (UFL == -1 && T*T_star1 <  0.0) { 
	  T_star1 = T_star2 = 0.0;	      
	  CDF *= -1;
	  // added to consider the sign of Tdisp correctly [MO - 01 Jun 2015]
	  Tdisp1_star1 = Tdisp1; //[MO - 01 Jun 2015] 
	  Tdisp2_star1 = Tdisp2; //[MO - 01 Jun 2015] 
	  Tdisp3_star1 = Tdisp3; //[MO - 01 Jun 2015] 
	  if (T*Tdisp < 0) {
	    Tdisp1_star1 *= - 1; //[MO - 01 Jun 2015] 
	    Tdisp2_star1 *= - 1; //[MO - 01 Jun 2015] 
	    Tdisp3_star1 *= - 1; //[MO - 01 Jun 2015] 
	  } 
	}
      }
      // CDF*T_star2 must be smaller than CDF*T
      else if (CDF*T_star2 > CDF*T && (T_step_group == 3 || T_step_group == 4)) {
	if      (UFL ==  1) T_star2 = T;
	else if (UFL == -1) T_star2 = 0.0;
      }
      // |T_star2| must be smaller than |T_star1|
      else if (fabs(T_star2) >= fabs(T_star1) && (T_step_group == 3 || T_step_group == 4)) {
	if      (UFL ==  1 && T_star1*T_star2 >= 0.0) T_star1 = T_star2 = 0.0;
	else if (UFL ==  1 && T_star1*T_star2 <  0.0) T_star2 = - T_star1;
	else if (UFL == -1 && T_star1*T_star2 >= 0.0) T_star2 = 0.0;
	else if (UFL == -1 && T_star1*T_star2 <  0.0) { //changed temporary [MO 01June2015]
	  T_star1 = T_star2 = 0.0;
	  CDF *= -1;
	  // added to consider the sign of Tdisp correctly [MO - 01 Jun 2015]
	  Tdisp1_star1 = Tdisp1; //[MO - 01 Jun 2015] 
	  Tdisp2_star1 = Tdisp2; //[MO - 01 Jun 2015] 
	  Tdisp3_star1 = Tdisp3; //[MO - 01 Jun 2015] 
	  if (T*Tdisp < 0) {
	    Tdisp1_star1 *= - 1; //[MO - 01 Jun 2015] 
	    Tdisp2_star1 *= - 1; //[MO - 01 Jun 2015] 
	    Tdisp3_star1 *= - 1; //[MO - 01 Jun 2015] 
	  }
	} 
      }
    }
  }

  //**************************************************************************************
  //if (shearupdate && i == 7  && update->ntimestep == 1) 
  //fprintf(screen,"timestep tagi tagi T_step T_step_old skip slip_T UFL T_star1 T_star2 CDF dTdisp dN N Tdisp T_old T\n");
  /*if (shearupdate && i == 7  && update->ntimestep % 10 == 0)
    fprintf(screen,"%i %i %i %i %i %i %i %i %1.6e %1.6e %i %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e\n",update->ntimestep,i,i,T_step,T_step_old,skip,CTD,UFL,T_star1,T_star2,CDF,dTdisp,overlap,N,Tdisp,T_old,T,Tdisp1_star1,Tdisp2_star1,Tdisp3_star1,Tdisp1,Tdisp2,Tdisp3);*/
  //******************************************************

  // timestep tagi tagj 0:T1 1:T2 2:T3 3:overlap 4:Tdisp 5:Tdisp1 6:Tdisp2 7:Tdisp3 8:T_star1 9:T_star2, 10:0.0 11:CDF 12:T 13:Tdisp_DD 14:CTD 15:T_step 16:0.0 17:a 18:N 19:0.0 20:0.0 21:ccel 22:Tdisp*1 23:Tdisp*2 24:Tdisp*3 25:0.0 26:0.0 27:nstrain 28:senergy 29:spenergy K_N K_T theta UFL THETA1 nstr sstr dTdisp
  /*if (update->ntimestep % 10 == 0 && i == 734) 
    fprintf(screen,"%i %i %i %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %i %1.6e %1.6e %i %i %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %1.6e %i %i %1.6e %1.6e %1.6e\n",
	    update->ntimestep,i,i,shear[0],shear[1],shear[2],overlap,Tdisp,Tdisp1,Tdisp2,Tdisp3,T_star1,T_star2,shear[10],CDF,T,
	    Tdisp_DD,CTD,T_step,shear[16],a,N,shear[19],shear[20],ccel,Tdisp1_star1,Tdisp2_star1,Tdisp3_star1,shear[25],shear[26],shear[27],shear[28],shear[29],K_N,K_T,theta_t,UFL,THETA1,nstr,sstr,dTdisp);*/
    
  
  //**************************************************************************************	    
  // Store the CTD and CDF as absolute values for the next step to avoid the sign changing [MO - 15 December 2014] 
  if (CTD == -1) CTD = 10000;
  if (CDF == -1) CDF = 10000;
  
  if (shearupdate) {      
    // shear[0], shear[1] and shear[2] are tangential force of x, y and z directions, respectively.
    shear[3] = overlap;           // previous overlap
    shear[4] = Tdisp;             // previous tangential displacement
    shear[5] = Tdisp1;            // tangential displacement of x direction in the previous step
    shear[6] = Tdisp2;            // tangential displacement of y direction in the previous step
    shear[7] = Tdisp3;            // tangential displacement of z direction in the previous step
    shear[8] = T_star1;           // first reverse point of tangential load (T) from loading to unloading
    shear[9] = T_star2;           // second reverse point of tangential load from unloading to re-loading
    shear[10] = 0.0;           
    shear[11] = CDF;              // CDF indicates wheather system is loading or unloading (int 1 or -1)       
    shear[12] = T;                // tangential contact force 
    shear[13] = Tdisp_DD;         // Tdisp_DD <= 0.0 should be satisfied to move on a new loading curve for N+dN    	   
    shear[14] = CTD;              // +1 and -1 mean positive and negative shear disp, respectively (int 1 or -1)   
    shear[15] = T_step;           // loading step
    shear[16] = 0.0;      
    shear[17] = a;
    shear[18] = N;
    shear[19] = 0.0; 
    shear[20] = 0.0; 
    shear[21] = ccel;  
    shear[22] = Tdisp1_star1;
    shear[23] = Tdisp2_star1;
    shear[24] = Tdisp3_star1;
    // shear[25] is empty.
  }
    
  //*******************************************************************************************

  fwall[0] += fx;
  fwall[1] += fy;
  fwall[2] += fz; 
  
  if (evflag) ev_tally_wall(i,fx,fy,fz,dx,dy,dz,radius);  
}

/* ---------------------------------------------------------------------- */

void FixWallGran::CMD_history(double rsq, double dx, double dy, double dz,
			      double *vwall, double *v,
			      double *f, double *omega, double *torque,
			      double radius, double meff, double *shear, int i)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double damp,ccel,vtr1,vtr2,vtr3,vrel;
  double fs,fslim,fx,fy,fz;
  double shsqmag,shsqnew,rsht,shratio,polyhertz,rinv,rsqinv;
  double wspinx,wspiny,wspinz,shint0,shint1,shint2,omdel;

  r = sqrt(rsq);
  rinv = 1.0/r;
  rsqinv = 1.0/rsq;

  // relative translational velocity

  vr1 = v[0] - vwall[0];
  vr2 = v[1] - vwall[1];
  vr3 = v[2] - vwall[2];

  // normal component

  vnnr = vr1*dx + vr2*dy + vr3*dz;
  vn1 = dx*vnnr / rsq;
  vn2 = dy*vnnr / rsq;
  vn3 = dz*vnnr / rsq;

  // tangential component

  vt1 = vr1 - vn1;
  vt2 = vr2 - vn2;
  vt3 = vr3 - vn3;

  // relative velocities

  vtr1 = vt1 - dz*omega[1]+dy*omega[2];
  vtr2 = vt2 - dx*omega[2]+dz*omega[0];
  vtr3 = vt3 - dy*omega[0]+dx*omega[1];
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);

  // shear history effects
  //~ Note that shear now refers to shear force, not shear displacement

  shsqmag = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];

  // rotate shear forces onto new contact plane conserving length

  rsht = shear[0]*dx + shear[1]*dy + shear[2]*dz;
  rsht = rsht*rsqinv;
  if (shearupdate) {
    shear[0] -= rsht*dx;
    shear[1] -= rsht*dy;
    shear[2] -= rsht*dz;
    shsqnew = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];
    if (shsqnew!=0.0) {
      shratio=sqrt(shsqmag/shsqnew);
      shear[0] *= shratio; // conserve shear force length
      shear[1] *= shratio;
      shear[2] *= shratio;
    }
  }

  // then perform rotation for rigid-body SPIN
  omdel=omega[0]*dx+omega[1]*dy+omega[2]*dz;
  wspinx=0.5*rsqinv*dx*omdel;
  wspiny=0.5*rsqinv*dy*omdel;
  wspinz=0.5*rsqinv*dz*omdel;
  shint0 = shear[0];
  shint1 = shear[1];
  shint2 = shear[2];

  if (shearupdate) {
    shear[0]=shint0+shint1*(-wspinz*dt)+shint2*wspiny*dt;
    shear[1]=shint0*wspinz*dt+shint1+shint2*(-wspinx*dt);
    shear[2]=shint0*(-wspiny*dt)+shint1*wspinx*dt+shint2;
  }
  
  //*******************************************************************************************
  // The main part of the CMD model is wrriten down below as of 4th August 2015 by Masahide [MO]
  //*******************************************************************************************

  double tolerance = 1.0e-20;
  double xmu_p = static_cast<double>(xmu_p);
  //double xmu_mean = 0.5*(xmu + xmu_p);
  double xmu_mean = xmu;

  if (THETA1 == 0 && xmu_mean > tolerance) {

    // rotate shear displacement onto new contact plane conserving length
    double shdsq_mag,shdsq_new,rshd,shdratio,shint5,shint6,shint7;
    shdsq_mag = shear[5]*shear[5] + shear[6]*shear[6] + shear[7]*shear[7];
    rshd = shear[5]*dx + shear[6]*dy + shear[7]*dz;
    rshd *= rsqinv;
    if (shearupdate) {
      shear[5] -= rshd*dx;
      shear[6] -= rshd*dy;
      shear[7] -= rshd*dz;
      shdsq_new = shear[5]*shear[5] + shear[6]*shear[6] + shear[7]*shear[7];
      if (shdsq_new!=0.0) {
	shdratio=sqrt(shdsq_mag/shdsq_new);
	shear[5] *= shdratio; // conserve shear force length
	shear[6] *= shdratio;
	shear[7] *= shdratio;
      }
    }
    // then perform rotation for rigid-body SPIN
    shint5 = shear[5];
    shint6 = shear[6];
    shint7 = shear[7];
    if (shearupdate) {
      shear[5]=shint5+shint6*(-wspinz*dt)+shint7*wspiny*dt;
      shear[6]=shint5*wspinz*dt+shint6+shint7*(-wspinx*dt);
      shear[7]=shint5*(-wspiny*dt)+shint6*wspinx*dt+shint7;
    }

    // rotate shear displacement onto new contact plane conserving length
    // Added for Tdisp_star1 [MO - 02 April 2015]
    double shdsq_mag_star1,shdsq_new_star1,rshd_star1,shdratio_star1,shint22,shint23,shint24;
    shdsq_mag_star1 = shear[22]*shear[22] + shear[23]*shear[23] + shear[24]*shear[24];
    rshd_star1 = shear[22]*dx + shear[23]*dy + shear[24]*dz;
    rshd_star1 *= rsqinv;
    if (shearupdate) {
      shear[22] -= rshd_star1*dx;
      shear[23] -= rshd_star1*dy;
      shear[24] -= rshd_star1*dz;
      shdsq_new_star1 = shear[22]*shear[22] + shear[23]*shear[23] + shear[24]*shear[24];
      if (shdsq_new_star1!=0.0) {
	shdratio_star1=sqrt(shdsq_mag_star1/shdsq_new_star1);
	shear[22] *= shdratio_star1; // conserve shear force length
	shear[23] *= shdratio_star1;
	shear[24] *= shdratio_star1;
      }
    }
    // then perform rotation for rigid-body SPIN
    shint22 = shear[22];
    shint23 = shear[23];
    shint24 = shear[24];
    if (shearupdate) {
      shear[22]=shint22+shint23*(-wspinz*dt)+shint24*wspiny*dt;
      shear[23]=shint22*wspinz*dt+shint23+shint24*(-wspinx*dt);
      shear[24]=shint22*(-wspiny*dt)+shint23*wspinx*dt+shint24;
    }
  }
  //**************************************************************************
  // normal force = new CM model that is diffeence from the previous CM one. 
  // CMD model was modified [MO - 04 August 2015]
  //**************************************************************************
  double overlap = radius-r;
  double R_star = radius;
  //********************************************
  double Poiseq_p = static_cast<double>(Poiseq_p);
  double Geq_p = static_cast<double>(Geq_p);
  //double G_star = 1.0/((2.0-Poiseq)/Geq+(2.0-Poiseq_p)/Geq_p);
  double G_star = 0.5*Geq/(2.0-Poiseq);
  //double E_star = 1.0/(0.5*(1.0-Poiseq)/Geq+0.5*(1.0-Poiseq_p)/Geq_p);
  double E_star = Geq/(1.0-Poiseq);
  double RMSf_p = static_cast<double>(RMSf_p);
  //double RMSf_eq = sqrt(RMSf*RMSf + RMSf_p*RMSf_p);
  double RMSf_eq = RMSf;
  double Hp_p = static_cast<double>(Hp_p);
  //double Hp_mean = 0.5*(Hp + Hp_p);
  double Hp_mean = Hp;
  //********************************************
  double pi = 4.0*atan(1.0);
  double overlap_p1 = 0.82 * RMSf_eq;     
  double overlap_p2 = 1.24 * RMSf_eq;
  double overlap_p_sum = overlap_p1 + overlap_p2;
  double N_T200 = 100.0*RMSf_eq*E_star*sqrt(2.0*R_star*RMSf_eq);
  double N_T2   = 1.0  *RMSf_eq*E_star*sqrt(2.0*R_star*RMSf_eq);
  double overlap_T200 = pow(3.0*N_T200/(4.0*sqrt(R_star)*E_star),2.0/3.0)+overlap_p_sum; 
  
  double b_coeff,overlap_T2,c_coeff;
  if (RMSf > tolerance) {
    b_coeff = 2.0*E_star*sqrt(R_star*(overlap_T200-overlap_p_sum))*(overlap_T200 - overlap_p1)/N_T200;
    overlap_T2 = (overlap_T200 - overlap_p1)*pow(N_T2/N_T200,1.0/b_coeff) + overlap_p1; 
    c_coeff = b_coeff * (N_T200/N_T2) * overlap_T2 * pow(overlap_T200-overlap_p1,-b_coeff) * pow(overlap_T2-overlap_p1,b_coeff-1.0);
  }
  else b_coeff = overlap_T2 = c_coeff = 0.0;

  double N = 0.0;
  double effectivekn = 0.0;
  double effectivekt = 0.0;
  double N_max = 0.0;
  double overlap_asperity = 0.0;
  double overlap_combined = 0.0;
  double overlap_hertz = 0.0;
  double polyhertz_effective = 0.0;
  double alpha; // non-dimentional roughness parameter (see Johnson 1985)
  int    N_step;
  
  //********************************************************************
  if (overlap < overlap_T2) {
    N_step = 11;
    overlap_asperity = overlap;
    N = N_T2 * pow(overlap_asperity / overlap_T2,c_coeff);
    effectivekn = c_coeff * (N_T2/overlap_T2) * pow(overlap_asperity / overlap_T2, c_coeff-1.0);  
    ccel = N*rinv;
  }
  else if ((overlap >= overlap_T2) && (overlap < overlap_T200)) {
    N_step = 12;
    overlap_combined = overlap - overlap_p1;  
    N = N_T200 * pow(overlap_combined / (overlap_T200 - overlap_p1), b_coeff);
    effectivekn = b_coeff * N_T200/(overlap_T200 - overlap_p1) * pow(overlap_combined / (overlap_T200-overlap_p1), b_coeff-1.0); 
    ccel = N*rinv;
  }
  else { // this includes when RMSf = 0.0
    N_step = 13;
    overlap_hertz = overlap - overlap_p_sum;
    N = 4.0/3.0*E_star*sqrt(R_star)*pow(overlap_hertz,1.5);
    effectivekn = 2.0 * E_star * sqrt(R_star*overlap_hertz);
    ccel = N*rinv;
  }
	  
  polyhertz = sqrt(overlap*R_star); // equivalent radius of contact
  // radius of contact assuming Hertzian contact
  if (polyhertz != 0.0) alpha = R_star * RMSf_eq / (polyhertz*polyhertz);
  polyhertz_effective = pow(3.0/4.0 * N * R_star / E_star, 1.0/3.0);
  effectivekt = 2.0*(1.0-Poiseq)/(2.0-Poiseq)*effectivekn;

  //**************************************************************************
  /*Hertzian model defines a = polyhertz = sqrt(R_star*overlap); however, it is 
    an appearent radius of contact area for new CM odel. The code defines 
    a = polyhertz_effective, instead.*/
  double a = polyhertz_effective;
  double a_old = fabs(shear[17]);
  double N_old = fabs(shear[18]);
  double dN = N - N_old;
  double K_N = effectivekn; 

  //**************************************************************************
  // Tangential contact model based on Mindlin&Deresiewicz(1953) is from here.	
  //**************************************************************************
  double T = 0.0;                               // bulk tangential force
  double T_old = shear[12];                     // bulk tangential force at the previous step (including its sign)
  double T_star1 = shear[8];                    // T*:  tangential force at the revasal step from loading to unloading 
  double T_star2 = shear[9];                    // T**: tangential force at the revasal step from unloading to re-loading 
  double Tdisp  = 0.0;                          // bulk tangential displacement	
  double Tdisp_mag = 0.0;                        // magnitude of bulk tangential displacement
  double Tdisp_old  = shear[4];                 // bulk tangential displacement at the previous step
  double dTdisp = 0.0;                          // increment of bulk displacement
  double Tdisp1 = 0.0;                          // tangential displacement of x direction
  double Tdisp2 = 0.0;                          // tangential displacement of y direction
  double Tdisp3 = 0.0;                          // tangential displacement of z direction
  double Tdisp1_old = shear[5];                 // tangential displacement of x direction at the previous step
  double Tdisp2_old = shear[6];                 // tangential displacement of y direction at the previous step
  double Tdisp3_old = shear[7];                 // tangential displacement of z direction at the previous step     
  double inner_product = 0.0;                   // inter product between current and previous vectors of tangential displacement
  int T_step = 0;                               // the current loading phase, e.g. 11 = N increasing T increasing
  int T_step_old = static_cast<int>(fabs(shear[15])); // the loading case at the previous step
  int slip_T = 0;
  int UFL = 0;                                  // UFL is the direction of tangential load 	
  int CTD = static_cast<int>(fabs(shear[14]));         // CTD is the direction of tangential displacement 
  if (CTD == 10000) CTD = -1;
  int CDF = static_cast<int>(fabs(shear[11]));         // CDF is the direction of tangential load-displacement system
  if (CDF == 10000) CDF = -1;
  double theta1 = 1.0;
  double theta2 = 1.0;
  double theta3 = 1.0;
  double theta_t = 1.0;
  double Tdisp1_star1 = shear[22];     // tangential displacement of x direction at T_star1
  double Tdisp2_star1 = shear[23];     // tangential displacement of y direction at T_star1
  double Tdisp3_star1 = shear[24];     // tangential displacement of z direction at T_star1
  double Tdisp_star1_mag;
  
  fslim = xmu_mean * N;   // N = ccel*r  
  int    T_step_group = 0;
  int    T_step_group_old = 0;
  double Tdisp_DD = 0.0;
  double K_T = 0.0;       // tangential contact stiffness
  double dT = 0.0;        // increment of tangential contact force
  double fs_old = sqrt(shsqmag);	
  double dTdisp_mag = sqrt(vtr1*vtr1+vtr2*vtr2+vtr3*vtr3)*dt; 


  if (xmu_mean < tolerance) {
    K_T = 0.0;
    T_step = 0;
    CTD = 1;
    CDF = 1;
    T_step = 0; 
    Tdisp1 = Tdisp2 = Tdisp3 = Tdisp = dTdisp = dTdisp_mag = 0.0;
    T = shear[0] = shear[1] = shear[2] = 0.0;
  } 
  else if (THETA1 == 1) {
    K_T = effectivekt;
    T_step = 1;
    CTD = 1;
    CDF = 1;
    dTdisp = dTdisp_mag = 0.0;
    // update T & Tdisps later
  } 
  else if (dTdisp_mag < 1.0e-15) {
    // need to update historical parameters
    K_T = 0.0;
    T_step = T_step_old;
    T = T_old;
    Tdisp1 = Tdisp1_old;          
    Tdisp2 = Tdisp2_old;
    Tdisp3 = Tdisp3_old;
    Tdisp = Tdisp_old;
    dTdisp = dTdisp_mag = 0.0;
  }
  else {
  
    //******************************************************************
    // First, calculate the shear displacement
    // Skip the follwing calculation if xmu_mean or dTdisp is small
    //******************************************************************
  
    Tdisp1 = Tdisp1_old + vtr1*dt;          
    Tdisp2 = Tdisp2_old + vtr2*dt;
    Tdisp3 = Tdisp3_old + vtr3*dt;
    Tdisp_mag = sqrt(Tdisp1*Tdisp1 + Tdisp2*Tdisp2 + Tdisp3*Tdisp3);
	    
    // inner_product becomes negative if tangential disp. becomes negative.
    inner_product = Tdisp1*Tdisp1_star1 + Tdisp2*Tdisp2_star1 + Tdisp3*Tdisp3_star1;
    Tdisp_star1_mag = sqrt(Tdisp1_star1*Tdisp1_star1 + Tdisp2_star1*Tdisp2_star1 + Tdisp3_star1*Tdisp3_star1);
    
    if (Tdisp_mag < tolerance && fabs(Tdisp_old) < tolerance && T_step_old == 0) {
      CTD = 1;                                     // when there is no shear disp at all.
      CDF = 1;
    }
    else if (Tdisp_mag > tolerance && fabs(Tdisp_old) < tolerance && T_step_old == 0) {
      // this is for the first increment of tangential displacement
      CTD = 1;       
      CDF = 1;
    }
    else if (CDF == 1 && (Tdisp_star1_mag < tolerance))  CTD = 1;
    else if (inner_product >= 0.0)                       CTD = 1;
    else                                                 CTD = -1; 
    
    Tdisp = CDF * CTD * Tdisp_mag; // CDF is added here [MO - 02 April 2015]    

    // give the sign for tangential displacement based on the value of the inner product 
    dTdisp = Tdisp - Tdisp_old;   // dTdisp includes direction of tangential displacement (not magnitude)


    //***********************************************************************************
    /* Fig.7 of Mindlin & Deresiewicz (1953) explain why this special case is needed.
       Tdisp_DSC is the minimum tangential displacement to move onto a new loading curve of N+dN.
       If the increment of the tangential displacement is less than Tdisp_DSC, the resultant tangential 
       force is less than theoretical value.
       The maximum tangential contact stiffness (8.0*G_star*a) is used until the current tangential force
       catches up the the new loading curve. */
    //***********************************************************************************
    double Tdisp_DSC = 0.0;
    if (a > tolerance) Tdisp_DSC = xmu_mean*dN/effectivekt;  // sign of Tdisp_DSC depends upon sign of dN;

    double Tdisp_DD = fabs(shear[13]) - fabs(dTdisp) + Tdisp_DSC;
    int    special_DD = 0; 
    if (Tdisp_DD < 0.0) Tdisp_DD = 0.0;            // Tdisp_DD <= 0.0 should be satisfied to move onto a new loading curve for N+dN.
    else                special_DD = 1;            // Special case for DD value is now active                         
    //*************************************************************************************
  
  
    // Calculate the tangential contact stiffness and the resultant tangential contact force
    double K_T = 0.0;       // tangential contact stiffness
    double dT = 0.0;        // increment of tangential contact force
    double fs_ratio;        // ratio to re-scale the tangential force if full sliding take places
    double fs_old = sqrt(shsqmag);
    double fs_new;
    //**********************************************************************************
  
    // Identify the loading steps for tangential component
    // for the first step of the special case
    if (special_DD == 1){
      theta_t = 1.0;
      if (dN > 0.0 && T_step_old != 115 && T_step_old != 125 && T_step_old != 135 && T_step_old != 145){
	if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance)          T_step = 115;
	else if (CDF*dTdisp <  0.0 && fabs(T_star2) < tolerance)                                  T_step = 125; 
	else if (CDF*dTdisp >= 0.0 && fabs(T_old) <= fabs(T_star1))                                   T_step = 135;
	else if (CDF*dTdisp <  0.0 && fabs(T_old) <= fabs(T_star1) && fabs(T_star2) >= tolerance)      T_step = 145;
	else T_step = T_step_old;
	//else fprintf(screen,"timestep %i Unexpected case occurred in zone C (HMD-wall-ball). ERROR!!\n",update->ntimestep);
	// if the special case was invoked in the previous step and is still active
      }else if ((T_step_old == 115 || T_step_old == 135) && CDF*dTdisp >= 0.0){  
	// still in step_115 or 135; if moved to unloading of T, go to the usual case
	T_step = T_step_old;
      }else if ((T_step_old == 125 || T_step_old == 145) && CDF*dTdisp < 0.0){   
	// still in step_125 or 145; if moved to re-loading of T, go to the usual case
	T_step = T_step_old;                                           
      }else if (T_step_old == 115 && CDF*dTdisp < 0.0){
	// the first step from loading of special case to unloading of special case.
	T_step = 125;
      }else if (T_step_old == 125 && CDF*dTdisp >= 0.0){
	// the first step from unloading of special case to reloading of special case.
	T_step = 135;
      }else if (T_step_old == 135 && CDF*dTdisp < 0.0){
	// the first step from reloading  of special case to re-unloading of special case.
	T_step = 145;
      }else if (T_step_old == 145 && CDF*dTdisp > 0){
	// the first step from re-unloading  of special case to re-reloading of special case.
	T_step = 135;
      }
      else T_step = T_step_old;  
      
      //*************************************************************************************
    }else {  // Usual cases 
      // T loading: T_step = *1 
      if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance) { 
	theta1 = 1.0-(CDF*T_old+xmu_mean*dN)/(xmu_mean*N);	
	if (theta1 <= 0.0) theta1 = theta_t = tolerance;
	else theta_t = pow(theta1, 1.0/3.0);
	if       (dN >  tolerance)      T_step = 11;   // N increasing
	else if  (dN < -tolerance)      T_step = 21;   // N decreasing
	else                            T_step =  1;   // N constant
      }
      // T unloading: T_step = *2 
      else if (CDF*dTdisp < 0.0 && fabs(T_star2) < tolerance) {
	// For the first step from loading to unloading, T* is still null, which should be T* = T_old.
	if  (fabs(T_star1) > tolerance) theta2 = 1.0-(CDF*(T_star1-T_old)+2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	else                           theta2 = 1.0-(2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	if (theta2 <= 0.0) theta2 = theta_t = tolerance;	  
	else               theta_t  = pow(theta2, 1.0/3.0);
	if      (dN >  tolerance) T_step = 12; 
	else if (dN < -tolerance) T_step = 22;      
	else                      T_step =  2;
      }
      // T re-loading: T_step = *3
      else if (CDF*dTdisp >= 0.0 && fabs(T_old) <= fabs(T_star1)) { 
	// For the first step from unloading to re-loading, T** is still null, which should be T** = T_old.
	if  (fabs(T_star2) > tolerance) theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	else                           theta3 = 1.0-(2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	if (theta3 <= 0.0) theta3 = theta_t = tolerance;
	else               theta_t  = pow(theta3, 1.0/3.0);
	if      (dN >  tolerance) T_step = 13; 
	else if (dN < -tolerance) T_step = 23;     
	else                      T_step = 3;
      }
      // T re-unloading: T_step = *4
      else if (CDF*dTdisp < 0.0 && fabs(T_old) <= fabs(T_star1) && fabs(T_star2) >= tolerance) { 
	// This part is same with the above (T_step = *3) due to simplification.
	theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu_mean*dN)/(2.0*xmu_mean*N);
	if (theta3 <= 0.0) theta3 = theta_t = tolerance;
	else               theta_t  = pow(theta3, 1.0/3.0);
	if      (dN >  tolerance) T_step = 14;
	else if (dN < -tolerance) T_step = 24; 
	else                      T_step =  4;      
      }
      else {
	T_step = T_step_old;
	theta_t = 1.0;
	//else fprintf(screen,"timestep %i Unexpected case occurred in zone C (HMD-wall-ball). ERROR!!\n",update->ntimestep);
      }
    } 
    
    //********************************************************************************************
    // Classify the groups [MO - 19 March 2015]
    if      (T_step == 0)                                                  T_step_group = 0;
    else if (T_step == 1 || T_step == 11 || T_step == 21 || T_step == 115) T_step_group = 1;
    else if (T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) T_step_group = 2;
    else if (T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) T_step_group = 3;
    else if (T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) T_step_group = 4;
    else                                                                   T_step_group = 5; // error
    // Classify the previos groups
    if      (T_step_old == 0)                                                              T_step_group_old = 0;
    else if (T_step_old == 1 || T_step_old == 11 || T_step_old == 21 || T_step_old == 115) T_step_group_old = 1;
    else if (T_step_old == 2 || T_step_old == 12 || T_step_old == 22 || T_step_old == 125) T_step_group_old = 2;
    else if (T_step_old == 3 || T_step_old == 13 || T_step_old == 23 || T_step_old == 135) T_step_group_old = 3;
    else if (T_step_old == 4 || T_step_old == 14 || T_step_old == 24 || T_step_old == 145) T_step_group_old = 4;
    else                                                                                   T_step_group_old = 5; // error

    //***********************************************************************************
    //UFL is neccesary for the calculation of tangential contact stiffness
    if (T_step_group == 2 || T_step_group == 4) UFL = -1; // Added [MO - 21 Dec 2014]
    else                                        UFL =  1;
    //***********************************************************************************
  
    // Calculate the tangential contact stiffness and the resultant tangential contact force 
    if (theta_t > 1.0) theta_t = 1.0;
    if (fabs(dTdisp) > 1.0e-15) K_T = effectivekt*theta_t + CDF*UFL*xmu_mean*(1.0-theta_t)*dN/dTdisp;      
    else K_T = 0.0;
    dT = K_T * dTdisp; 
    T = dT + T_old;
    // rescale tangential force if full sliding takes place 	
    if (fabs(T) > fslim) {
      if (fabs(T) != 0.0) {
	slip_T = 1;
	T *= fslim/fabs(T);
      } 
    }
  } // End of If THETA = 0


    // update the tangential contact force by distributing dT into 3 components based on imcrese of relative rotational velosity
  if (shearupdate) {
    shear[0] -= K_T * vtr1 * dt;  // -= K_T * dTdisp1;
    shear[1] -= K_T * vtr2 * dt;  // -= K_T * dTdisp2;
    shear[2] -= K_T * vtr3 * dt;  // -= K_T * dTdisp3;
  }

  
  // fabs(T) and fs are not always same for 3D simulation [MO - 03 April 2015] 
  fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  
  if (fs > fslim) {
    if (fs != 0.0 && xmu_mean > tolerance) {
      slip_T = 1;
      if (shearupdate) {
	shear[0] *= fslim/fs; 
	shear[1] *= fslim/fs;
	shear[2] *= fslim/fs;   
      } 
    } else shear[0] = shear[1] = shear[2] = 0.0;
  } 
    
  double fs_new = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  if (fs_new > fslim) fs_new = fslim;

  double slip_degree = fs/fslim;
  if (THETA1 == 1 && xmu_mean > tolerance) {
    if (K_T > tolerance && fs_new > tolerance) {
      // consider as a virgin tangential loading
      T = fs_new;
      if (slip_degree >= 1.0)      theta_t = 0.0; //Tdisp = 1.5 * fslim / K_T; 
      else if (slip_degree < 0.0)  theta_t = 1.0; //Tdisp = 0.0; 
      else                         theta_t = pow((1.0-slip_degree),1.0/3.0); 
      Tdisp = 1.5 * fslim / K_T * (1.0 - theta_t*theta_t);
      Tdisp1 = Tdisp * shear[0]/fs_new;
      Tdisp2 = Tdisp * shear[1]/fs_new;
      Tdisp3 = Tdisp * shear[2]/fs_new;    
    }
    else T = Tdisp = Tdisp1 = Tdisp2 = Tdisp3 = 0.0;
  }
	
  // forces & torques

  fx = dx*ccel + shear[0];
  fy = dy*ccel + shear[1];
  fz = dz*ccel + shear[2];

  f[0] += fx;
  f[1] += fy;
  f[2] += fz;

  torque[0] -= dy*shear[2] - dz*shear[1];
  torque[1] -= dz*shear[0] - dx*shear[2];
  torque[2] -= dx*shear[1] - dy*shear[0];
  
  double b1inv = 1.0/(b_coeff+1.0);
  double c1inv = 1.0/(c_coeff+1.0);
  double nstr,nstr_asperity, nstr_combined, nstr_hertz;
  double sstr,incdissipf,dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy;

  //~ Add contributions to traced energy [KH - 20 February 2014]
  //~~ Call function for twisting resistance model [MO - 04 November 2014]
  if (*D_spin && shearupdate) { 
    Deresiewicz1954_spin(i,sheardim,dx,dy,dz,radius,r,torque,shear,dspin_i,
			   dspin_stm,spin_stm,dM_i,dM,K_spin,theta_r,
			   M_limit,Geq,Poiseq,Dspin_energy,a,N);
  }
  if (pairenergy) {
    //********************************************************************
    // Calculation of normal strain energy
    //********************************************************************	 
    if (N_step == 11) {
      nstr_asperity = c1inv * N * overlap_asperity;
      nstr = nstr_asperity;
    }
    else if (N_step == 12) {  
      nstr_asperity = c1inv * N_T2 * overlap_T2;
      //nstr_combined = b1inv*N_T200*(pow(overlap_combined,b_coeff+1.0)*pow(overlap_T200 - overlap_p1,-b_coeff) 
      //				  - pow(overlap_T2-overlap_p1,b_coeff+1.0)*pow(overlap_T200-overlap_p1,b_coeff));
      nstr_combined = b1inv * (overlap_combined * N - (overlap_T2-overlap_p1) * N_T2);
      nstr = nstr_asperity + nstr_combined;
    }	    
    else {   
      nstr_asperity = c1inv * N_T2 * overlap_T2;
      //nstr_combined = b1inv * N_T200 * ((overlap_T200 - overlap_p1) 
      //				      - pow(overlap_T2-overlap_p1,b_coeff+1.0)*pow(overlap_T200-overlap_p1,b_coeff));
      nstr_combined = b1inv * ((overlap_T200 - overlap_p1) * N_T200 - (overlap_T2-overlap_p1) * N_T2);
      nstr_hertz = 0.4 * 4.0/3.0 * E_star * sqrt(R_star) * (pow(overlap_hertz,2.5) - pow(overlap_T200-overlap_p_sum,2.5));
      nstr = nstr_asperity + nstr_combined + nstr_hertz;
    }    
    normalstrain += nstr;
    if (trace_energy) shear[27] = nstr;
      

    //************
    if (xmu_mean > tolerance) {
      if (effectivekt > tolerance) incdissipf = 0.5*fs_new*fs_new/effectivekt; // this is "SHEAR STRAIN" not frictional dissipation
      else incdissipf = 0.0;
      dissipfriction += incdissipf;
      if (trace_energy) shear[26] = incdissipf; // not accumulated
    }
    //************


    /* Full sliding can be considered as accumulation of partial slip for HMD model.
       Theoretically speaking, full sliding does not take place for HMD modle.
       Thus, shear strain energy here is summation of shear strain energy + friction for shm model.*/
    if (xmu > tolerance) {
      if (THETA1 == 1) {
	if (fabs(K_T) > tolerance) sstr = 0.5*(fs_new + fs_old)*(fs - fs_old)/K_T;
	else sstr = 0.0;
      }
      else if (THETA1 == 0) {
	if (T < 0) fs_new *= -1.0; 
	if (T_old < 0) fs_old *= -1.0; 
	if (dTdisp < 0) dTdisp_mag *= -1.0; // not fabs(dTdisp)!!
	sstr = 0.5*dTdisp_mag*(fs_new + fs_old);
      }
      else sstr = 0.0;
      shearstrain += sstr;
      if (trace_energy) shear[28] += sstr;
    }
    
    //~~ Update the spin contribution [MO - 13 November 2014]
    if (D_spin) {
      spinenergy += Dspin_energy;
      if (trace_energy) shear[29] += Dspin_energy;
    }	    
  }
	  
  //*************************************************************************************
  // Update the T_star1 and T_star2 considering the change of the normal contact load
  int sp_Tstar1 = 0;
  if (THETA1 == 0 && xmu_mean > tolerance) {
    if (T_step_group != 1 && fabs(T_star1) > tolerance) T_star1 += CDF*xmu_mean*dN;
    if (T_step_group == 3 || T_step_group == 4)         T_star2 -= CDF*xmu_mean*dN;  	    
    //***********************************************************************************
    // Update T_star1 and T_star2
    if (T_step_group == 1) T_star1 = T_star2 = 0.0;
    else if (T_step_group == 2 && T_step_group_old == 1) {
      // the first step from T_step = 1: loading to unloading
      T_star1 = T;
      Tdisp_DD = 0.0;
      sp_Tstar1 = 1;
      Tdisp1_star1 = Tdisp1;     
      Tdisp2_star1 = Tdisp2;     
      Tdisp3_star1 = Tdisp3;    
      if (T*Tdisp < 0) {
	Tdisp1_star1 *= - 1; //[MO - 01 Jun 2015] 
	Tdisp2_star1 *= - 1; //[MO - 01 Jun 2015] 
	Tdisp3_star1 *= - 1; //[MO - 01 Jun 2015] 
      } 
    }
    else if (T_step_group == 3 && T_step_group_old == 2) {
      // the first step from T_step = 2: unloading to re-loading
      T_star2 = T;
      Tdisp_DD = 0.0;
      sp_Tstar1 = 1;
      if (fabs(T_star2) > fabs(T_star1)) { // avoid initial flactuations [MO - 01 Jun 2015]
	T_star1 = T_star2 = 0.0;
	Tdisp1_star1 = Tdisp2_star1 = Tdisp3_star1 = 0.0; //[MO - 01 Jun 2015] 
      }
    } 
    //****************************************************************************************
    if ((T_step_group == 2 || T_step_group == 3 || T_step_group == 4) && sp_Tstar1 == 0) {
      // Introduce the following rules to avoid numerical errors [MO - 30 March 2015]
      // [1] CDF*(T*) > CDF*T > CDF*(T**), [2] |T*| > |T**| 
      if (fabs(T) > fabs(T_star1) && (T_step_group == 2 || T_step_group == 3 || T_step_group == 4)) {
	if (UFL == 1 && T*T_star1 >= 0.0) T_star1 = T_star2 = 0.0;
	else if (UFL ==  1 && T*T_star1 <  0.0) T_star1 = - T;
	else if (UFL == -1 && T*T_star1 >= 0.0) T_star1 =   T;
	else if (UFL == -1 && T*T_star1 <  0.0) { 
	  T_star1 = T_star2 = 0.0;	      
	  CDF *= -1;
	  // added to consider the sign of Tdisp correctly [MO - 01 Jun 2015]
	  Tdisp1_star1 = Tdisp1; //[MO - 01 Jun 2015] 
	  Tdisp2_star1 = Tdisp2; //[MO - 01 Jun 2015] 
	  Tdisp3_star1 = Tdisp3; //[MO - 01 Jun 2015] 
	  if (T*Tdisp < 0) {
	    Tdisp1_star1 *= - 1; //[MO - 01 Jun 2015] 
	    Tdisp2_star1 *= - 1; //[MO - 01 Jun 2015] 
	    Tdisp3_star1 *= - 1; //[MO - 01 Jun 2015] 
	  } 
	}
      }
      // CDF*T_star2 must be smaller than CDF*T
      else if (CDF*T_star2 > CDF*T && (T_step_group == 3 || T_step_group == 4)) {
	if      (UFL ==  1) T_star2 = T;
	else if (UFL == -1) T_star2 = 0.0;
      }
      // |T_star2| must be smaller than |T_star1|
      else if (fabs(T_star2) >= fabs(T_star1) && (T_step_group == 3 || T_step_group == 4)) {
	if      (UFL ==  1 && T_star1*T_star2 >= 0.0) T_star1 = T_star2 = 0.0;
	else if (UFL ==  1 && T_star1*T_star2 <  0.0) T_star2 = - T_star1;
	else if (UFL == -1 && T_star1*T_star2 >= 0.0) T_star2 = 0.0;
	else if (UFL == -1 && T_star1*T_star2 <  0.0) { //changed temporary [MO 01June2015]
	  T_star1 = T_star2 = 0.0;
	  CDF *= -1;
	  // added to consider the sign of Tdisp correctly [MO - 01 Jun 2015]
	  Tdisp1_star1 = Tdisp1; //[MO - 01 Jun 2015] 
	  Tdisp2_star1 = Tdisp2; //[MO - 01 Jun 2015] 
	  Tdisp3_star1 = Tdisp3; //[MO - 01 Jun 2015] 
	  if (T*Tdisp < 0) {
	    Tdisp1_star1 *= - 1; //[MO - 01 Jun 2015] 
	    Tdisp2_star1 *= - 1; //[MO - 01 Jun 2015] 
	    Tdisp3_star1 *= - 1; //[MO - 01 Jun 2015] 
	  }
	} 
      }
    }
  }
  
  //*********************************************************************************************************    
  // Store the CTD and CDF as absolute values for the next step to avoid the sign changing [MO - 15 December 2014] 
  if (CTD == -1) CTD = 10000;
  if (CDF == -1) CDF = 10000;
  
  if (shearupdate) {      
    // shear[0], shear[1] and shear[2] are tangential force of x, y and z directions, respectively.
    shear[3] = overlap;           // previous overlap
    shear[4] = Tdisp;             // previous tangential displacement
    shear[5] = Tdisp1;            // tangential displacement of x direction in the previous step
    shear[6] = Tdisp2;            // tangential displacement of y direction in the previous step
    shear[7] = Tdisp3;            // tangential displacement of z direction in the previous step
    shear[8] = T_star1;           // first reverse point of tangential load (T) from loading to unloading
    shear[9] = T_star2;           // second reverse point of tangential load from unloading to re-loading
    shear[10] = 0.0;           
    shear[11] = CDF;              // CDF indicates wheather system is loading or unloading (int 1 or -1)       
    shear[12] = T;                // tangential contact force 
    shear[13] = Tdisp_DD;         // Tdisp_DD <= 0.0 should be satisfied to move on a new loading curve for N+dN    	   
    shear[14] = CTD;              // +1 and -1 mean positive and negative shear disp, respectively (int 1 or -1)   
    shear[15] = T_step;           // loading step
    shear[16] = 0.0;      
    shear[17] = a;
    shear[18] = N;
    shear[19] = 0.0; 
    shear[20] = 0.0; 
    shear[21] = ccel;  
    shear[22] = Tdisp1_star1;
    shear[23] = Tdisp2_star1;
    shear[24] = Tdisp3_star1;
    // shear[25] is empty.
  }
  //*******************************************************************************************

  fwall[0] += fx;
  fwall[1] += fy;
  fwall[2] += fz;
 
  if (evflag) ev_tally_wall(i,fx,fy,fz,dx,dy,dz,radius);  
}

/* ----------------------------------------------------------------------
  memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixWallGran::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 0.0;
  if (history) bytes += nmax*sheardim * sizeof(double);   // shear history
  if (fix_rigid) bytes += nmax * sizeof(int);             // mass_rigid
  return bytes;
}

/* ----------------------------------------------------------------------
  allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixWallGran::grow_arrays(int nmax)
{
  if (history) memory->grow(shearone,nmax,sheardim,"fix_wall_gran:shearone");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixWallGran::copy_arrays(int i, int j, int delflag)
{
  if (history)
    for (int m = 0; m < sheardim; m++)
      shearone[j][m] = shearone[i][m];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixWallGran::set_arrays(int i)
{
  if (history)
    for (int m = 0; m < sheardim; m++)
      shearone[i][m] = 0;
}

/* ----------------------------------------------------------------------
  pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixWallGran::pack_exchange(int i, double *buf)
{
  if (!history) return 0;

  int n = 0;
  for (int m = 0; m < sheardim; m++)
    buf[n++] = shearone[i][m];
  return n;
}

/* ----------------------------------------------------------------------
  unpack values into local atom-based arrays after exchange
------------------------------------------------------------------------- */

int FixWallGran::unpack_exchange(int nlocal, double *buf)
{
  if (!history) return 0;

  int n = 0;
  for (int m = 0; m < sheardim; m++)
    shearone[nlocal][m] = buf[n++];
  return n;
}

/* ----------------------------------------------------------------------
  pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixWallGran::pack_restart(int i, double *buf)
{
  if (!history) return 0;

  int n = 0;
  buf[n++] = sheardim + 1;
  for (int m = 0; m < sheardim; m++)
    buf[n++] = shearone[i][m];
  return n;
}

/* ----------------------------------------------------------------------
  unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixWallGran::unpack_restart(int nlocal, int nth)
{
  if (!history) return;

  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  for (int i = 0; i < sheardim; i++)
    shearone[nlocal][i] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
  maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixWallGran::maxsize_restart()
{
  if (!history) return 0;
  return 1 + sheardim;
}

/* ----------------------------------------------------------------------
  size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixWallGran::size_restart(int nlocal)
{
  if (!history) return 0;
  return 1 + sheardim;
}

/* ---------------------------------------------------------------------- */

void FixWallGran::reset_dt()
{
  dt = update->dt;
}

/* ---------------------------------------------------------------------- 
  Allows the user to do a fix_modify at the input script and change the
  parameters of the fix. Only allows a few things to be modified.
  Returns the number of arguments read.
------------------------------------------------------------------------- */

int FixWallGran::modify_param(int narg, char **arg)
{
  if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
  int argsread=0;

  if (strcmp(arg[argsread],"gamman") == 0) {// do while loop instead?
    fprintf(screen, "changed wall gamman from %f to ",gamman);
    gamman=force->numeric(FLERR,arg[argsread+1]);
    argsread+=2;
    fprintf(screen, "%f\n",gamman);
  }
  else if (strcmp(arg[argsread],"gammat") == 0) {
    fprintf(screen, "changed wall gammat from %f to ",gammat);
    gammat=force->numeric(FLERR,arg[argsread+1]);
    argsread+=2;
    fprintf(screen, "%f\n",gammat);
  }
  else if (strcmp(arg[argsread],"mu") == 0) {
    fprintf(screen, "changed wall friction coefficient from %f to ",xmu);
    xmu=force->numeric(FLERR,arg[argsread+1]);
    argsread+=2;
    fprintf(screen, "%f\n",xmu);
  }
  else if (strcmp(arg[argsread],"dampflag") == 0) {
    dampflag=force->inumeric(FLERR,arg[argsread+1]);
    argsread+=2;
    fprintf(screen, "changed wall dampflag to %d \n",dampflag);
  }
  else if (strcmp(arg[argsread],"translate") == 0) {
    if (strcmp(arg[argsread+1],"off") == 0) {
      wtranslate=0;
      vwall[0]=vwall[1]=vwall[2]=0.0;
      argsread+=2;
      fprintf(screen, "stopped wall translation\n");
    } else {
      wtranslate = 1;
      vwall[0] = force->numeric(FLERR,arg[argsread+1]);
      vwall[1] = force->numeric(FLERR,arg[argsread+2]);
      vwall[2] = force->numeric(FLERR,arg[argsread+3]);
      argsread+= 4;
      fprintf(screen, "changed wall velocity to [ %e %e %e ]\n",vwall[0],vwall[1],vwall[2]);
    }
  }
  // Added for wave propagation simulation [MO - 10 March 2015]
  else if (strcmp(arg[argsread],"wiggle") == 0) {
    if (strcmp(arg[argsread+1],"off") == 0) {
      amplitude = 0;
      // period = 0; // not necessary here.
      wiggle = 1; // Keep this 1 for wave propagation simulations [MO - 30 Aug 2015]
      argsread+=2;
      fprintf(screen, "stopped wall wiggle\n");
    } else {
      // Increase argsread from +4 to +5  [MO - 09 May 2016]
      if (argsread+5 > narg) error->all(FLERR,"Illegal fix wall/gran command");
      if (strcmp(arg[argsread+1],"x") == 0) axis = 0;
      else if (strcmp(arg[argsread+1],"y") == 0) axis = 1;
      else if (strcmp(arg[argsread+1],"z") == 0) axis = 2;
      else error->all(FLERR,"Illegal fix wall/gran command");
      amplitude = force->numeric(FLERR,arg[argsread+2]);
      period = force->numeric(FLERR,arg[argsread+3]);
      if (strcmp(arg[argsread+4],"cos") == 0) wiggletype = 1;
      else if (strcmp(arg[argsread+4],"sin") == 0) wiggletype = 2;
      else error->all(FLERR,"Illegal fix wall/gran command");
      wiggle = 1;
      //loINI = lo; 
      //hiINI = hi; 
      argsread += 5;
      fprintf(screen, "changed wall wiggle to [ %i (0,1,2 = x,y,z) amplitude = %e m period %e s]\n",axis,amplitude,period);
    }
  }
  else if (strcmp(arg[argsread],"stresscontrol") == 0) {
    if (wscontrol == 0 && strcmp(arg[argsread+1],"off") == 0) { // removed wtranslate [MO - 28 Aug 2015]
      error->all(FLERR,"Illegal fix modify wall/gran command");
    }
    wscontrol = 1;
    // wtranslate = 1; // removed [MO - 28 Aug 2015]
    if (strcmp(arg[argsread+1],"off") == 0) {
      wscontrol = 0;
      wtranslate = 0;
      ftvarying = 0;
      if (fstr) delete [] fstr;
      gain = 0.0;
      argsread += 2;
      fprintf(screen, "stopped wall stress control\n");
    } else if (strstr(arg[argsread+1],"v_") == arg[argsread+1]) {
      ftvarying = 1;
      int nn = strlen(&arg[argsread+1][2]) + 1;
      if (fstr) delete [] fstr;// command to extend fstr?
      fstr = new char[nn];
      strcpy(fstr,&arg[argsread+1][2]);
      gain = force->numeric(FLERR,arg[argsread+2]);
      argsread += 3;
      fprintf(screen, "Set wall stress control with varying target force\n");
    } else {
      targetf = force->numeric(FLERR,arg[argsread+1]);
      ftvarying = 0;
      gain = force->numeric(FLERR,arg[argsread+2]);
      argsread += 3;
      fprintf(screen, "Set wall stress control with constant target force\n");
    }
  }
  else {
    fprintf(screen,"Argument %s not yet supported\n",arg[argsread]);
    error->all(FLERR,"Illegal fix modify wall/gran command");
  }
  if (argsread==narg) {
    if (gamman <0.0 || gammat <0.0 || xmu <0.0 ) error->all(FLERR,"Check the values for the modified granular wall parameters");
    if ((wtranslate || wscontrol) && (lo != -BIG && hi != BIG)) // added wscontrol [MO - 28 Aug 2015]
      error->all(FLERR,"Cannot specify both top and bottom walls and translate for fix wall/gran - check your fix_modify"); // this check will fail if the walls have moved..
    if ((wtranslate || wscontrol) && wallstyle == ZCYLINDER) // added wscontrol [MO - 28 Aug 2015]
      error->all(FLERR,"Cannot use translate with cylinder fix wall/gran - check your fix_modify");
    if ((wtranslate || wscontrol) && (wiggle || wshear)) // added wscontrol [MO - 28 Aug 2015]
      error->all(FLERR,"Cannot translate and wiggle or shear fix wall/gran- check your fix_modify");
  }
  return argsread;
}

/* ---------------------------------------------------------------------- 
  A function that implements wall movement
------------------------------------------------------------------------- */

void FixWallGran::move_wall() {
  //~ Only update lo or hi if not NULL [KH - 27 November 2013]
  // Upate w_boxlo & w_boxhi in Domain.cpp [MO -02 Sep 2015]
  if (lo != -BIG) { 
    lo+=vwall[wallstyle]*dt;
    domain->w_boxlo[wallstyle] = lo;
    //MPI_Bcast(&domain->w_boxlo[wallstyle],1,MPI_DOUBLE,0,world);
  }
  if (hi != BIG) {
    hi+=vwall[wallstyle]*dt; 
    domain->w_boxhi[wallstyle] = hi;
    //MPI_Bcast(&domain->w_boxhi[wallstyle],1,MPI_DOUBLE,0,world);
  }
}

/* ---------------------------------------------------------------------- 
  A function that calculates the wall velocity based on a target force and a 
  specific controller
------------------------------------------------------------------------- */

void FixWallGran::velscontrol() {

  //MPI_Allreduce(fwall,fwall_all,3,MPI_DOUBLE,MPI_SUM,world);
  if (ftvarying == 1) {
    //modify->clearstep_compute();needed???
    targetf = input->variable->compute_equal(fvar);
    //modify->addstep_compute(update->ntimestep + 1);needed???
  }
  //if (update->ntimestep != time_origin)
  //vwall[wallstyle] = gain * (targetf - fwall_all[wallstyle]);


  //*****************************
  // [MO - 13 August 2015]
  /* gain -> max. engineering strain rate
     targetf -> target mean stress
     fwall_all[wallstyle] is not used in the modified calculation*/
 
  //~ Update the atom stresses if necessary
  if (stressatom->invoked_peratom != update->ntimestep) {
    stressatom->compute_peratom();
  }
  /*~ Import a 6-element array of mean stresses from ComputeStressAtom and
    accumulate the means from all processors in tallymeans*/
  double tmeans[6];
  double *means = stressatom->array_export();
  for (int i = 0; i < 6; i++) tmeans[i] = 0.0;
  MPI_Allreduce(&means[0],&tmeans[0],6,MPI_DOUBLE,MPI_SUM,world);

  //max engineering strain rate is applied until meanstress/targetstress = 0.5 [MO - 20 Aug 2015]
  double boxlength_start = fabs(domain->w_boxhi_start[wallstyle] - domain->w_boxlo_start[wallstyle]);
  double boxlength = fabs(domain->w_boxhi[wallstyle] - domain->w_boxlo[wallstyle]);
    
  if (tmeans[wallstyle] < 0.5 * targetf) vwall[wallstyle] = 0.5 * gain * boxlength_start; // max. engineering velocity
  else vwall[wallstyle] = 2 * 0.5 * gain * boxlength_start * (1.0 - tmeans[wallstyle]/targetf);
    
  w_ierates[wallstyle] = vwall[wallstyle] / boxlength; // true strain rate
             
  //****************************
}

/* ---------------------------------------------------------------------- 
  A function that calculates the outputs of the fix
  1st output:low wall position
  2nd output:high wall position
  3rd-5th outputs:forces on atoms by wall
  6th output:coordination number on wall [MO - 12 March 2015]
------------------------------------------------------------------------- */

double FixWallGran::compute_vector(int n)
{ 
  if (n == 0) return lo;
  if (n == 1) return hi;
  MPI_Allreduce(wcoordnos,wcoordnos_all,1,MPI_DOUBLE,MPI_SUM,world);
  if (n == 5) return wcoordnos_all[0]; // added to output coordination number of wall [MO - 12 March 2015]
  // only sum across procs one time?? //if (eflag == 0) {??
  MPI_Allreduce(fwall,fwall_all,3,MPI_DOUBLE,MPI_SUM,world);
  if (update->ntimestep == time_origin) error->warning(FLERR,"Force output by fix_wall_gran not computed properly");
  if (n>5) error->all(FLERR,"Illegal fix_wall_gran output"); // previously n>4 [MO - 12 March 2015]
  return fwall_all[n-2];
}

/* ---------------------------------------------------------------------- 
  Adds the wall forces to the per-atom stress accessed through
  compute stress/atom
------------------------------------------------------------------------- */

void FixWallGran::ev_tally_wall(int i, double fx, double fy, double fz,
				double dx, double dy, double dz, double radi)
{

  double volume;

  //calculate stresses and assign them to vatom array
  //if (vflag_either) {
  if (vflag_atom) {
    //if (i < nlocal) {//i already lower than nlocal
    //volume of the particle
    if(domain->dimension == 3)
      volume = 4.0*MY_PI/3.0 * radi*radi*radi; //sphere
    else if (domain->dimension == 2)
      volume = MY_PI * radi*radi; //disk
    else
      error->all(FLERR,"Cannot read correct dimension");

    vatom[i][0] -= dx * fx / volume;
    vatom[i][1] -= dy * fy / volume;
    vatom[i][2] -= dz * fz / volume;
    vatom[i][3] -= dx * fy / volume;
    vatom[i][4] -= dx * fz / volume;
    vatom[i][5] -= dy * fz / volume;
    //}
  }
  //}
}

/* ---------------------------------------------------------------------- */

void FixWallGran::rolling_resistance(int i, int numshearq, double dx, double dy, double dz, double r, double radius, double ccel, double maxshear, double effectivekt, double *torque, double *shear, double *db, double *localdM, double *globaldM)
{
  /*~ This rolling resistance model was developed by Xin Huang during
    the summer and autumn of 2013. It is the companion function of
    PairGranHookeHistory::rolling_resistance; more information about
    the rolling resistance model can be obtained by looking at this
    code [KH - 30 October 2013]*/

  //~ If *rolling_delta == 0, exit from this function prematurely
  double tolerance = 1.0e-20;
  if (*rolling_delta < tolerance) {
    for (int q = 0; q < 15; q++) shear[numshearq-15+q] = 0.0;
    for (int q = 0; q < 3; q++) db[q] = localdM[q] = globaldM[q] = 0.0;
    return;
  }

  /*~ Since the walls are planar and axis-aligned, components of the 
    unit vector along the contact normal are found by normalising
    dx, dy and dz by particle radius*/
  double nx = dx/r;
  double ny = dy/r;
  double nz = dz/r;

  /*~ Calculate the four components of the unit quaternion, q. Compare
    with a tolerance to ensure no division by zero problems*/
  double sinthetaovertwo, magnxny;
  sinthetaovertwo = sqrt(0.5*(1.0 - nz));
  if (nx < 0.0) sinthetaovertwo *= -1.0;

  double q[4];
  q[0] = sqrt(0.5*(1.0 + nz)); //~ = cos(theta/2)

  magnxny = sqrt(nx*nx + ny*ny);
  if (magnxny < tolerance) q[1] = q[2] = 0.0;
  else {
    q[1] = -sinthetaovertwo*ny/magnxny;
    q[2] = sinthetaovertwo*nx/magnxny;
  }
  q[3] = 0.0;

  //~ Compute the rotation matrix, T, from this quaternion
  double T[3][3];
  MathExtra::quat_to_mat(q,T);

  /*~ Calculate the common radius for which several options are 
    available. The default common radius was defined by Ai et al.
    (2012)*/
  double commonradius = radius;
  if (*model_type % 5 == 0) commonradius *= 2.0; //~ Jiang et al. (2005)
  if (*model_type % 3 == 0) commonradius = BIG; //~ Iwashita and Oda (1998, 2000) - 'infinite' common radius
  
  /*~ Obtain the omega values for the previous timestep from FixOldOmega.
    Note that dalpha will be 0 for the special case of a ball contacting
    a planar wall. Also all the 'da' terms are zero as the walls have
    no angular velocity. 'dur' and 'dus' equal 'db'*/
  double **oldomegas = ((FixOldOmega *) deffix)->oldomegas;
  double globaloldomega[3], localoldomega[3];
  for (int q = 0; q < 3; q++) globaloldomega[q] = oldomegas[i][q];
  
  /*~ Transfer the old omega values to the local coordinate system
    using the rotation matrix T*/
  MathExtra::transpose_matvec(T,globaloldomega,localoldomega);

  //~ Now find relative rotations, dthetar, in three directions
  double PI = 4.0*atan(1.0);
  double dthetar[3];
  for (int q = 0; q < 3; q++) {
    db[q] = radius*localoldomega[q]*dt;
    dthetar[q] = -db[q]/commonradius;
  }

  /*~ The equivalent area normal contact stiffness is found by dividing
    the magnitude of the normal contact force by the product of the normal 
    contact overlap and contact area. If division by zero, issue a warning.
    Also calculate some necessary quantities for later use*/
  int warnfrequency = 100; //~ How often to warn about stiffness calcs
  double un, B, delbyb, recipA, knbar, ksbar;

  un = radius - r; //~ Normal contact overlap
  B = sqrt(radius*radius - r*r); //~ Radius of contact plane

  if (B > tolerance) {
    delbyb = *rolling_delta*B;
    knbar = fabs(ccel*r/(un*PI*delbyb*delbyb));
  } else {
    //~ Issue a warning and broadcast lastwarning int to all procs
    if (update->ntimestep-lastwarning[0] >= warnfrequency) {
      fprintf(screen,"Cannot estimate either contact stiffness in rolling resistance model on timestep "BIGINT_FORMAT"\n",update->ntimestep);
      lastwarning[0] = lastwarning[1] = update->ntimestep;
      MPI_Bcast(&lastwarning[0],2,MPI_INT,comm->me,world);
    }
    delbyb = tolerance; //~ Tiny value
    knbar = tolerance; //~ Assume a tiny stiffness value
  }
  recipA = 1.0/(PI*delbyb*delbyb); //~ Reciprocal of modified contact area

  /*~ The equivalent area tangential contact stiffness is equal to the
    effective tangential stiffness (kt for Hookean model; kt*polyhertz
    for Hertzian model) divided by the contact area. If kt is zero or
    if the contact area is very, very small, warn the user. Values of
    ksbar are stored in the third-last column of the shear array.*/
  if (kt < tolerance && update->beginstep == update->ntimestep-1 
      && comm->me == 0)
    error->warning(FLERR,"Using zero kt: tangential contact stiffness cannot be estimated in rolling resistance model");

  if (B > tolerance && effectivekt > tolerance) ksbar = effectivekt*recipA;
  else if (shear[numshearq-3] > tolerance) ksbar = fabs(shear[numshearq-3]);
  else {
    if (kt >= tolerance && update->ntimestep-lastwarning[1] >= warnfrequency) {
      fprintf(screen,"Cannot estimate tangential contact stiffness in rolling resistance model on timestep "BIGINT_FORMAT"\n",update->ntimestep);
      lastwarning[1] = update->ntimestep;
      MPI_Bcast(&lastwarning[1],1,MPI_INT,comm->me,world);
    }
    ksbar = tolerance; //~ Assume a tiny value
  }
  shear[numshearq-3] = ksbar; //~ Store ksbar in third-last column of shear array

  /*~ Calculate the maximum allowable accumulated global moment and
    store these in mlimit for convenience. The values of (kappa+1) 
    for the contact, stored for rolling and twisting in shear[numshearq-2]
    and shear[numshearq-1], respectively, are used in the calculation. If
    the contact is newly-formed, (kappa+1) will be 0.0 indicating that
    both stored values need to be initialised to their user-specified
    values [KH - 18 July 2014]*/
  double st[3], mlimit[3];
  if (fabs(shear[numshearq-1]) < 0.5) 
    shear[numshearq-1] = shear[numshearq-2] = *kappa + 1.0;

  mlimit[0] = mlimit[1] = (fabs(shear[numshearq-2])-1.0)*radius*ccel*r;
  mlimit[2] = (fabs(shear[numshearq-1])-1.0)*radius*maxshear;

  //~ Also create an 'st' array for convenience
  double deltaBpow = MathSpecial::powint(delbyb,4);
  st[0] = st[1] = -0.25*PI*deltaBpow*knbar;
  st[2] = -0.5*PI*deltaBpow*ksbar;

  //~ Calculate local increments of rolling resistance
  for (int q = 0; q < 3; q++) localdM[q] = st[q]*dthetar[q];
  if (*model_type % 7 == 0) localdM[0] = localdM[1] = 0.0;
  if (*model_type % 11 == 0) localdM[2] = 0.0;

  double scalefactor; //~ Denominator used for scaling resistances
  double acclocal[3], accglobal[3];
  for (int q = 0; q < 3; q++) {
    /*~ If the accumulated local resistances exceed the permissible
      limits, proportionally reduce the size of the incremental
      resistances and also scale the accumulated resistances 
      so that the accumulated local resistance will equal the 
      appropriate limit once the increment is added to it at the end
      of this function. The accumulated local rolling and twisting 
      resistances are stored in the ninth-last, eighth-last and
      seventh-last columns of the shear array*/
    scalefactor = fabs(shear[numshearq-9+q] + localdM[q]);
    if (scalefactor > mlimit[q]) {
      /*~ Updated the stored value of (kappa+1) for the 
	rolling model only (not twisting). If post_limit_index
	== 0, set the stored value to 1; otherwise don't change
	it [KH - 29 July 2014]*/
      if (q < 2) {
	shear[numshearq-2] += (*post_limit_index - 1.0)*(fabs(shear[numshearq-2]) - 1.0);
	mlimit[q] *= *post_limit_index; //~ Control post-limit strength
      }
      localdM[q] *= mlimit[q]/scalefactor;
      shear[numshearq-9+q] *= mlimit[q]/scalefactor;
    }
    acclocal[q] = shear[numshearq-9+q]; // Fetch the accumulated local in one vector
  }

  /*~ Compute the global moment increments by multiplying the 
    transpose of the rotation matrix, T, by the local resistance
    increments. Do the same for the accumulated moment.*/
  MathExtra::matvec(T,localdM,globaldM);
  MathExtra::matvec(T,acclocal,accglobal);

  /*~ Now add the local resistance increments to the eighth-last, 
    seventh-last and sixth-last columns of the shear array. The
    accumulated global moment is in the sixth-last, fifth-last 
    and fourth-last columns. The accumulated values of dus are 
    stored in the three columns immediately before, and the 
    accumulated values of dur in the three columns immediately 
    before these.*/
  for (int q = 0; q < 3; q++) {
    shear[numshearq-6+q] = accglobal[q];
    shear[numshearq-9+q] += localdM[q];
    shear[numshearq-12+q] += db[q];
    shear[numshearq-15+q] += db[q];
  }

  //~ Finally update the torque values
  for (int q = 0; q < 3; q++) torque[q] -= accglobal[q];
}

/* ---------------------------------------------------------------------- */

void FixWallGran::Deresiewicz1954_spin(int i, int numshearq, double delx, double dely, double delz, double radius, double r, double *torque, double *shear, double *dspin_i, double &dspin_stm, double &spin_stm, double *dM_i, double &dM, double &K_spin, double &theta_r, double &M_limit, double Geq, double Poiseq, double &Dspin_energy, double a, double N)
{
  if (!D_switch) { 
    dspin_i[0] = dspin_i[1] = dspin_i[2] = dspin_stm = spin_stm = dM_i[0] 
      = dM_i[1] = dM_i[2] = dM = K_spin = theta_r = M_limit = Dspin_energy = 0.0;
    return;
  }
  double tolerance = 1.0e-20;
  if (xmu < tolerance || a < tolerance || N < tolerance) {
    dspin_i[0] = dspin_i[1] = dspin_i[2] = dspin_stm = spin_stm = dM_i[0] 
      = dM_i[1] = dM_i[2] = dM = K_spin = theta_r = M_limit = Dspin_energy = 0.0;
    return; // spin angle and torque is not accumulated
  }
  // initialisation of values to be returned to main compute function
  theta_r = 1.0;
  K_spin = 0.0;                
  dM_i[0] = dM_i[1] = dM_i[2] = 0.0;
  dspin_stm = 0.0;
  spin_stm = 0.0;
  dM = 0.0;

  /*~~ This code is written based on Deresiewicz(1954) using a DEM expression
    shown by Thornton&Yin(1991). Cosidred here is twisting resistance but not
    rolling resistance. This model can be activated only when Hertz&Mindlin's
    contact model is used. [MO 29 Oct 2014] ~~*/  
  //***************************************************************************
  /*To avoid the reverse of sign of shearquantities, 'fabs' is used for absolute quantities.
    Attention should be paied for some flags which have 1 or -1 values. For these cases, the 
    quantities were stored as 10000 instead of -1 and called with 'fabs', and changed to -1 
    again at the next step.[MO - 15 December 2014] */
  //***************************************************************************
  double r_inv = 1.0/r;
  double R_star = radius;
  // Equivalent values of wall-ball
  double Poiseq_p = static_cast<double>(Poiseq_p);
  double Geq_p = static_cast<double>(Geq_p);
  //double G_star = 1.0/((2.0-Poiseq)/Geq+(2.0-Poiseq_p)/Geq_p);
  double G_star = 1.0/((2.0-Poiseq)/Geq+(2.0-Poiseq)/Geq);
  double PI = 4.0*atan(1.0);
  double a_old = fabs(shear[numshearq-8]); 
  double N_old = fabs(shear[numshearq-9]);
  double dN = N - N_old; // change of normal contact force
  double xmu_p = static_cast<double>(xmu_p);
  //double xmu_mean = 0.5*(xmu + xmu_p); // mean firction coefficient is used 
  double xmu_mean = xmu;
  double a_pow3 = MathSpecial::powint(a,3);
  double K_spin_max = 16.0/3.0*G_star*a_pow3;
  M_limit = 3.0/16.0*PI*xmu_mean*N*a; // > 0.0

  // spin angle are stored as real values, but twisting momemts are stored as system values. 
  double M_old = shear[numshearq-4]; // M_system_old
  double M_star1 = shear[numshearq-5]; // M_system_star1
  double M_star2 = shear[numshearq-6]; // M_system_star2
  double spin_old = shear[numshearq-7];// Accumulated spin at previous step
  int *tag = atom->tag; //~ Write out the atom tags 
  double rsqinv = r_inv*r_inv;

  double **omega = atom->omega;
  dt = update->dt;
  // only relative rotation contributes to the spin resistance 
  double omdel_rel = omega[i][0]*delx + omega[i][1]*dely + omega[i][2]*delz;
  dspin_i[0] = rsqinv*delx*omdel_rel*dt; 
  dspin_i[1] = rsqinv*dely*omdel_rel*dt; 
  dspin_i[2] = rsqinv*delz*omdel_rel*dt; 
  // spin angle of wall-grain is zero, so spin angle of grain = relative spin angle.

  // Calculate dspin_angle
  int sign_dspin; 
  if (omdel_rel >= 0.0 ) sign_dspin = 1;
  else sign_dspin = -1;
  double dspin_mag = sqrt(dspin_i[0]*dspin_i[0]+dspin_i[1]*dspin_i[1]+dspin_i[2]*dspin_i[2]);
  double dspin = sign_dspin*dspin_mag;
  double spin = spin_old + dspin;
  double spin_i[3],spin_i_old[3],M_i_old[3],M_i[3];
  double spin_DD = fabs(shear[numshearq-11]);
  for (int q = 0; q < 3; q++){
    spin_i[q] = shear[numshearq-3+q] + dspin_i[q];
    M_i_old[q] = shear[numshearq-16+q];
  }  
  double t, theta_r_inv, M,dspin_min,M_i_mag,M_scale,M_temp;
  int special,step,sign_stm,step_old,sp_star1,dir_moment,dir_ST,slip;
  theta_r_inv = t = 1.0;
  sp_star1 = 0;                 // sp_star1 = 1 means that reversal of sign should be avoided. 
  step_old = static_cast<int>(fabs(shear[numshearq-12])); // previous step
  slip     = static_cast<int>(fabs(shear[numshearq-17])); // if slip took place at previous step, slip = 1;
  sign_stm = static_cast<int>(fabs(shear[numshearq-10])); // 1 = loading or reloading, -1 = unloading or re-unloading
  if (sign_stm == 10000) sign_stm = -1;
  dir_ST   = static_cast<int>(fabs(shear[numshearq-13])); // direction of spin-torque relation
  if (dir_ST == 10000) dir_ST = -1;
  if (a_old <= tolerance) dir_ST = 1;  // the first step after a contact is detacted.
  if (step_old == 0){ // the first step
    if (sign_dspin == 1) sign_stm = 1;  // spin_old = 0 by default, so the first step is always zero.
    else if (sign_dspin == -1) sign_stm = -1; 
  }
  
  //********************************************************************************************************
  // Skip the folowing calculation if incremental spin is null.
  //********************************************************************************************************
  if (fabs(dspin) <= tolerance) { 
    spin_stm = sign_stm * spin_old;
    spin = spin_old;
    dspin_stm = 0.0;
    dM = 0.0;  
    M = M_old;  
    for (int q = 0; q < 3; q++) {
      dM_i[q] = 0.0;
      M_i[q] = M_i_old[q];
      dspin_i[q] = 0.0;
      spin_i[q] = spin_i_old[q];
    }
  } 
  else {
    dspin_stm = sign_stm * dspin;
    spin_stm = sign_stm * spin; 
    // check if a special case is invoked*****************************************************************************
    if (a_old < tolerance) dspin_min = 0; // to initialize spin_DD
    else dspin_min = M_limit*(dN/N)/K_spin_max; // required min. spin angle not to invoke a special case
    spin_DD = spin_DD - dspin_mag + dspin_min; // accumulated shortage of spin to return normal case
    special = 0;
    if (spin_DD < tolerance) spin_DD = 0.0; // spin_DD <= 0.0 moves onto a new normal curve
    else special = 1;  // a special case is invoked.
    // **************************************************************************************************************   
    // check whether the first step of special case is invoked *****************
    if (special == 1){ //theta_r = 1.0
      //***********
      theta_r = 1.0;
      //***********
      if (dN > tolerance && step_old != 115 && step_old != 125 && step_old != 135 && step_old != 145){
	// the first step of the special case
	if (dir_ST*dspin_stm >= 0.0 && fabs(M_star1) < tolerance && fabs(M_star2) < tolerance)         step = 115;
	else if (dir_ST*dspin_stm <  0.0 && fabs(M_star2) < tolerance)                                 step = 125; 
	else if (dir_ST*dspin_stm >= 0.0 && fabs(M_old) < fabs(M_star1))                               step = 135;
	else if (dir_ST*dspin_stm <  0.0 && fabs(M_old) <= fabs(M_star1) && fabs(M_star2)>= tolerance) step = 145;
	else {
	  //fprintf(screen,"timestep %i Unexpected case occurred in zone A (Dspin-wall-ball). ERROR!!\n",update->ntimestep);
	}
	// if the special case was invoked in the previous step and is stil l active
      }else if ((step_old == 115 || step_old == 135) && dir_ST*dspin_stm >= 0){   
	// still in step_115 or 135; if moved to unloading of M, go to the usual case
	step = step_old;     
      }else if ((step_old == 125 || step_old == 145) && dir_ST*dspin_stm < 0){   
	// still in step_125 or 145; if moved to re-loading of M, go to the usual case
	step = step_old;                                           
      }else if (step_old == 115 && dir_ST*dspin_stm < 0){
	// the first step from loading of special case to unloading of special case.
	step = 125;
      }else if (step_old == 125 && dir_ST*dspin_stm >= 0){
	// the first step from unloading of special case to reloading of special case.
	step = 135;
      }else if (step_old == 135 && dir_ST*dspin_stm < 0){
	// the first step from reloading  of special case to re-unloading of special case.
	step = 145;
      }else if (step_old == 145 && dir_ST*dspin_stm > 0){
	// the first step from re-unloading  of special case to re-reloading of special case.
	step = 135;
      }
      //*************************************************************************************
    } else {  // Usual cases  // M loading: step = *1
      if (dir_ST*dspin_stm >= 0.0 && fabs(M_star1) < tolerance && fabs(M_star2) < tolerance){
	t = 1.0-1.5*((-dir_ST*M_old+3.0/16.0*PI*xmu_mean*dN*a)/(xmu_mean*N*a));
	if       (dN >  tolerance)  step = 11;   // N increasing
	else if  (dN < -tolerance)  step = 21;   // N decreasing
	else                        step =  1;   // N constant
      }
      // T unloading: step = *2 
      else if (dir_ST*dspin_stm < 0.0 && fabs(M_star2) < tolerance){
	// For the first step from loading to unloading, M* is still null, which should be M* = M_old.
	if  (fabs(M_star1) > tolerance) t = 1.0-1.5*((M_star1-(-dir_ST)*M_old)+3.0/8.0*PI*xmu_mean*dN*a)/(2.0*xmu_mean*N*a);
	else                            t = 1.0-1.5*(3.0/8.0*PI*xmu_mean*dN*a)/(2.0*xmu_mean*N*a);
	if      (dN >  tolerance)   step = 12; 
	else if (dN < -tolerance)   step = 22;      
	else                        step =  2;
      }  
      // M re-loading: step = *3
      else if (dir_ST*dspin_stm >= 0.0 && fabs(M_old) <= fabs(M_star1)){
	// For the first step from unloading to re-loading, M** is still null, which should be M** = M_old.
	if  (fabs(M_star2) > tolerance) t = 1.0-1.5*((-dir_ST*M_old-M_star2)+3.0/8.0*PI*xmu_mean*dN*a)/(2.0*xmu_mean*N*a);
	else                            t = 1.0-1.5*(3.0/8.0*PI*xmu_mean*dN*a)/(2.0*xmu_mean*N*a);
	if      (dN >  tolerance)   step = 13; 
	else if (dN < -tolerance)   step = 23;     
	else                        step = 3;
      }
      // M re-unloading: step = *4    
      else if (dir_ST*dspin_stm < 0.0 && fabs(M_old) < fabs(M_star1) && fabs(M_star2) >= tolerance){
	// This part is same with the above (step = *3) due to simplification.
	if  (fabs(M_star2) > tolerance) t = 1.0-1.5*((-dir_ST*M_old-M_star2)+3.0/8.0*PI*xmu_mean*dN*a)/(2.0*xmu_mean*N*a);
	else                            t = 1.0-1.5*(3.0/8.0*PI*xmu_mean*dN*a)/(2.0*xmu_mean*N*a);
	if      (dN >  tolerance) step = 14;
	else if (dN < -tolerance) step = 24; 
	else                      step =  4; 
      } 
      // if slip condition is still continued, the above condition may not be applicable.
      else if (slip == 1) {
	step = step_old;
	t = tolerance;
      }
      // warning!
      /*else {
	fprintf(screen,"timestep %i Unexpected case occurred in zone B (Dspin-wall-ball). ERROR!!\n",update->ntimestep);
	}*/
      if (t < 0.0) t = tolerance; // 0 < t <= 1;
      else if (t > 1.0) t = 1.0;
      theta_r_inv = 2.0*(1.0/sqrt(t))-1.0; // theta_r_inv >= 1;
      theta_r = 1.0/theta_r_inv;
      if (theta_r > 1.0)      theta_r = 1.0;
      else if (theta_r < 0.0) theta_r = 0.0;
    }
    //**********************************************************************************************::   
    // Check the drection of load  
    if (step == 2 || step == 12 || step == 22 || step == 125) dir_moment = -1;
    else if (step == 4 || step == 14 || step == 24 || step == 145) dir_moment = -1;
    else dir_moment = 1;

    if (fabs(1.0 - theta_r) < tolerance) K_spin = K_spin_max*theta_r;
    // dspin_stm = 0.0 is not applicable here because of the previous if statement.
    else K_spin = K_spin_max*theta_r + dir_moment * 3.0/16.0*PI*xmu_mean*a*(1.0-theta_r)*dN/dspin_stm;
    
    if (K_spin > K_spin_max) K_spin = K_spin_max;
    if (K_spin < 0.0) K_spin = 0.0;

    //~ Calculate increment of and accumulated twisting resistance.
    dM = dspin_stm*K_spin;
    M = M_old + dM;
    // Check whether slip is fully movilized or not. If so, rescale the accumulated twisting resistance 
    M_temp = M;
    if (fabs(M) > M_limit) {
      M *= M_limit/fabs(M);
      slip = 1;
    } 
    else slip = 0;
  
    for (int q = 0; q < 3; q++) {
      // They are not system values but three components
      dM_i[q] = dspin_i[q]/fabs(dspin) * fabs(dM);
      M_i[q] = M_i_old[q] + dM_i[q];  
    }
    // rescale the M_i to adjust to M
    M_i_mag = sqrt(M_i[0]*M_i[0]+M_i[1]*M_i[1]+M_i[2]*M_i[2]);
    for (int q = 0; q < 3; q++) {
      if (M_i_mag > 0) {
	M_scale = fabs(M)/M_i_mag;
	M_i[q] *= M_scale;
      }else M_i[q] = 0.0; 
    }        
  }
 
  //**************************************************************************************************
  // Calculation of spin energy in a similar way of shear statin energy.
  Dspin_energy = dspin_stm*0.5*(M+M_temp);
  // this is for wall-grain 
  //**************************************************************************************************
  if (step == 0) M_star1 = M_star2 = 0.0;  // dspin is null & no hysteresis
  else {
    // Update the M_star1 and M_star2 considering the change of the normal contact load
    if (step != 1 && step != 11 && step != 21 && step != 115 && fabs(M_star1) > tolerance)   M_star1 += dir_ST*3.0/16.0*PI*xmu_mean*dN*a;
    if ((step == 3 || step == 13 || step == 23 || step == 135) && fabs(M_star2) > tolerance) M_star2 -= dir_ST*3.0/16.0*PI*xmu_mean*dN*a;
    if ((step == 4 || step == 14 || step == 24 || step == 145) && fabs(M_star2) > tolerance) M_star2 -= dir_ST*3.0/16.0*PI*xmu_mean*dN*a;
    //*********************************************************************************************************
    // Update M_star1 and M_star2
    if (step == 1 || step == 11 || step == 21 || step == 115){   // do not forget to include step == 115
      M_star1 = 0.0;
      M_star2 = 0.0;
    }
    // Update M_star1 and M_star2 for new loading curve; move from re-loading to normal loading step
    if ((step == 3 || step == 13 || step == 23 || step == 135) && fabs(M) > fabs(M_star1) && fabs(M_star1) > tolerance){
      M_star1 = 0.0;
      M_star2 = 0.0;
    }
    // Update the M_star2 for new unloading curve; move to unloading step
    if ((step == 4 || step == 14 || step == 24 || step == 145) && fabs(M) > fabs(M_star2) && fabs(M_star2) > tolerance){
      M_star2 = 0.0;
    }
    if ((step == 2 || step == 12 || step == 22 || step == 125) && fabs(M_star1) <= tolerance && fabs(M_star2) <= tolerance){
      // the first step from step = 1; loading to unloading
      M_star1 = M_old; 
      spin_DD = 0.0;
      sp_star1 = 1;  // to avoid the reverse of sign_system
    } 
    if ((step == 3 || step == 13 || step == 23 || step == 135) && fabs(M_star1) > fabs(M) && fabs(M_star2) <= tolerance){  
      // the first step from step = 2; unloading to re-loading
      M_star2 = M_old;
      spin_DD = 0.0;
    }
    if ((step == 4 || step == 14 || step == 24 || step == 145) && fabs(M_star1) > fabs(M) && fabs(M_star2) <= tolerance){  
      // the first step from step = 3; re-loading to re-unloading
      spin_DD = 0.0;
    }
    //*********************************************************************************************************
    //special case when |M| for unloading reaches -|M_star1|  // move on to a new loading curve in the opposite direction
    if ((dir_moment == -1) && fabs(M) > fabs(M_star1) && fabs(M_star1) > tolerance){
      M_star1 = 0.0;
      M_star2 = 0.0;
      if (sp_star1 == 0)  dir_ST *= -1;  // the system is moved onto virgin loading curve
    }
    //*********************************************************************************************
    // Added this to avoid numerical errors [MO - 04 January 2015]
    if (fabs(M_star1 - M_star2) < tolerance) {
      M_star1 = 0.0;
      M_star2 = 0.0;
    }
    //*********************************************************************************************
  }
  // Store the sign_stm and dir_ST as absolute values for the next step to avoid the sign changing [MO - 15 December 2014] 
  if (sign_stm == -1) sign_stm = 10000;
  if (dir_ST == -1) dir_ST = 10000;

  for (int q = 0; q < 3; q++) {
    shear[numshearq-3+q] = spin_i[q]; // spin_i is not system value
    shear[numshearq-16+q] = M_i[q]; // M_i is not system value
  }
  shear[numshearq-4] = M; // M is always system value 
  shear[numshearq-5] = M_star1; // M is always system value 
  shear[numshearq-6] = M_star2; // M is always system value
  shear[numshearq-7] = spin; // not spin_stm
  shear[numshearq-8] = a;
  shear[numshearq-9] = N;    // Normal contact force
  shear[numshearq-10] = sign_stm; // int
  shear[numshearq-11] = spin_DD; 
  shear[numshearq-12] = step;    // int
  shear[numshearq-13] = dir_ST; // spin_twist relation
  shear[numshearq-17] = slip; //int
  shear[numshearq-18] += Dspin_energy; // spin_energy
  shear[numshearq-19] = 0; // empty
  
  /*~ Finally update the torque values for both i and j (the
    latter only if local to proc). The increments differ in sign*/
  for (int q = 0; q < 3; q++) torque[q] -= M_i[q];
}

/* ---------------------------------------------------------------------- */

void *FixWallGran::extract(const char *str, int &dim)
{
  /*~ This function was added so that the accumulated energy
    terms, if computed, may be extracted by ComputeEnergyGran
    [KH - 20 February 2014]*/

  //~ Extended for boundary work in FixEnergyBoundary [KH - 12 March 2014]

  dim = 0;
  if (strcmp(str,"dissipfriction") == 0) return (void *) &dissipfriction;
  else if (strcmp(str,"normalstrain") == 0) return (void *) &normalstrain;
  else if (strcmp(str,"shearstrain") == 0) return (void *) &shearstrain;
  else if (strcmp(str,"spinenergy") == 0) return (void *) &spinenergy;
  else if (strcmp(str,"wiggle") == 0) return (void *) &wiggle;
  else if (strcmp(str,"wtranslate") == 0) return (void *) &wtranslate;
  else if (strcmp(str,"wscontrol") == 0) return (void *) &wscontrol;
  else if (strcmp(str,"wallstyle") == 0) return (void *) &wallstyle;
  // vwall[axis]/fwall_all[axis] are used for wiggle command [MO - 21 Aug 2015] 
  else if (strcmp(str,"axis") == 0) return (void *) &axis;
  else if (strcmp(str,"velwall") == 0) return (void *) &vwall[axis];
  else if (strcmp(str,"fwall_all") == 0) return (void *) &fwall_all[axis];
  // w_ierates is used for stresscontrol command [MO - 21 Aug 2015]
  else if (strcmp(str,"w_ierates") == 0) return (void *) &w_ierates[wallstyle];
  else if (strcmp(str,"lo") == 0) return (void *) &lo;
  else if (strcmp(str,"hi") == 0) return (void *) &hi;
  return NULL;
}

/* ---------------------------------------------------------------------- */

void FixWallGran::write_restart(FILE *fp)
{
  /*~ Added (along with the restart function below) to store the 
    accumulated energy terms, if computed. Note that the total 
    energy dissipated by friction and stored as shear strain, 
    from all procs, is calculated and stored on proc 0 [KH - 28
    February 2014]*/

  double gatheredf = 0.0;
  double gatheredss = 0.0;
  double gatheredse = 0.0;
  MPI_Allreduce(&dissipfriction,&gatheredf,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&shearstrain,&gatheredss,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&spinenergy,&gatheredse,1,MPI_DOUBLE,MPI_SUM,world);

  int n = 0;
  double list[5]; // increased to 5 [MO - 25 August 2015]
  list[n++] = gatheredf;
  list[n++] = gatheredss;
  list[n++] = gatheredse;
  // increased to 5 [MO - 28 August 2015]
  list[n++] = hi;
  list[n++] = lo;
  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallGran::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;
  //~ Only read these quantities to proc 0 [KH - 28 February 2014]
  if (comm->me == 0) {
    dissipfriction = static_cast<double> (list[n++]);
    shearstrain = static_cast<double> (list[n++]);
    spinenergy = static_cast<double> (list[n++]);
  }
  MPI_Bcast(&n,1,MPI_INT,0,world); //~ Ensure n has the same value on all processors
  // increased the position of wall boundaries [MO - 28 August 2015]
  hi = static_cast<double> (list[n++]);
  lo = static_cast<double> (list[n++]);
}

/* ---------------------------------------------------------------------- */

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
  Contributing authors: Leo Silbert (SNL), Gary Grest (SNL)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdlib.h"
#include "string.h"
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
#include "mpi.h"
//~ Added compute header files for energy tracing [KH - 20 February 2014]
#include "compute.h"
#include "compute_energy_gran.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{XPLANE=0,YPLANE=1,ZPLANE=2,ZCYLINDER};    // XYZ PLANE need to be 0,1,2
enum{HOOKE,HOOKE_HISTORY,HERTZ_HISTORY,SHM_HISTORY,CM_HISTORY,HMD_HISTORY,CMD_HISTORY}; //~ Added SHM_HISTORY option [KH - 30 October 2013] other three were added [MO - 30 November 2014]] 

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixWallGran::FixWallGran(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  virial_flag = 1; // this fix can contribute to compute stress/atom
  //~ Reduced from 10 for shm pairstyle [KH - 30 October 2013]
  if (narg < 7) error->all(FLERR,"Illegal fix wall/gran command");

  if (!atom->sphere_flag)
    error->all(FLERR,"Fix wall/gran requires atom style sphere");

  //~ Global information is saved to restart file [KH - 20 February 2014]
  restart_global = 1;
  restart_peratom = 1;
  create_attribute = 1;

  vector_flag = 1;
  size_vector = 5;
  global_freq = 1;

  // wall/particle coefficients

  //~ Make special allowances for shm pairstyle [KH - 30 October 2013]
  //~~ Added for CM, HMD and CMD pairstyle [MO - 05 Januaray 2015]
  int iarg = 9;
  if (force->pair_match("gran/shm/history",1)) {
    Geq = force->numeric(FLERR,arg[3]);
    Poiseq = force->numeric(FLERR,arg[4]);
    xmu = force->numeric(FLERR,arg[5]);

    if (Geq < 0.0 || Poiseq < 0.0 || Poiseq > 0.5)
      error->all(FLERR,"Illegal shm pair parameter values in fix wall gran");

    kn = 4.0*Geq / (3.0*(1.0-Poiseq));
    kt = 4.0*Geq / (2.0-Poiseq);

    //~ Set dummy values for the remaining variables [KH - 9 January 2014]
    gamman = gammat = 0.0;
    dampflag = 0;
    iarg = 6; //~ Reduce number of args for SHM pairstyle [KH - 10 June 2014]
  } else if (force->pair_match("gran/CM/history",1)) {     
    Geq = force->numeric(FLERR,arg[3]);
    Poiseq = force->numeric(FLERR,arg[4]);
    xmu = force->numeric(FLERR,arg[5]);
    RMSf = force->numeric(FLERR,arg[6]);
    Hp = force->numeric(FLERR,arg[7]);
    
    if (Geq < 0.0 || Poiseq < 0.0 || Poiseq > 0.5)
      error->all(FLERR,"Illegal CM pair parameter values in fix wall gran");
    
    kn = 4.0*Geq / (3.0*(1.0-Poiseq));
    kt = 4.0*Geq / (2.0-Poiseq);
    
    //~ Set dummy values for the remaining variables [KH - 9 January 2014]
    gamman = gammat = 0.0;
    dampflag = 0;
    iarg = 8; //~ Reduce number of args for CM pairstyle [KH - 10 June 2014]
  } else if (force->pair_match("gran/HMD/history",1)) {
    Geq = force->numeric(FLERR,arg[3]);
    Poiseq = force->numeric(FLERR,arg[4]);
    xmu = force->numeric(FLERR,arg[5]);

    if (Geq < 0.0 || Poiseq < 0.0 || Poiseq > 0.5)
      error->all(FLERR,"Illegal HMD pair parameter values in fix wall gran");

    kn = 4.0*Geq / (3.0*(1.0-Poiseq));
    kt = 4.0*Geq / (2.0-Poiseq);
    
    //~ Set dummy values for the remaining variables [KH - 9 January 2014]
    gamman = gammat = 0.0;
    dampflag = 0;
    iarg = 6; //~ Reduce number of args for HMD pairstyle [MO - 18 June 2014]
  } else if (force->pair_match("gran/CMD/history",1)) {     
    Geq = force->numeric(FLERR,arg[3]);
    Poiseq = force->numeric(FLERR,arg[4]);
    xmu = force->numeric(FLERR,arg[5]);
    RMSf = force->numeric(FLERR,arg[6]);
    Hp = force->numeric(FLERR,arg[7]);
    
    if (Geq < 0.0 || Poiseq < 0.0 || Poiseq > 0.5)
      error->all(FLERR,"Illegal CMD pair parameter values in fix wall gran");
    
    kn = 4.0*Geq / (3.0*(1.0-Poiseq));
    kt = 4.0*Geq / (2.0-Poiseq);
    
    //~ Set dummy values for the remaining variables [KH - 9 January 2014]
    gamman = gammat = 0.0;
    dampflag = 0;
    iarg = 8; //~ Reduce number of args for CMD pairstyle [MO - 30 November 2014]
  }
  else {
    kn = force->numeric(FLERR,arg[3]);
    if (strcmp(arg[4],"NULL") == 0) kt = kn * 2.0/7.0;
    else kt = force->numeric(FLERR,arg[4]);

    gamman = force->numeric(FLERR,arg[5]);
    if (strcmp(arg[6],"NULL") == 0) gammat = 0.5 * gamman;
    else gammat = force->numeric(FLERR,arg[6]);

    xmu = force->numeric(FLERR,arg[7]);
    dampflag = force->inumeric(FLERR,arg[8]);
    if (dampflag == 0) gammat = 0.0;
  }

  if (kn < 0.0 || kt < 0.0 || gamman < 0.0 || gammat < 0.0 ||
      xmu < 0.0 || xmu > 10000.0 || dampflag < 0 || dampflag > 1)
    error->all(FLERR,"Illegal fix wall/gran command");

  // convert Kn and Kt from pressure units to force/distance^2 if Hertzian

  if (force->pair_match("gran/hertz/history",1) || force->pair_match("gran/shm/history",1)) {
    kn /= force->nktv2p;
    kt /= force->nktv2p;
  }
  
  // wallstyle args
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
    cylradius = force->numeric(FLERR,arg[iarg+1]);
    iarg += 2;
  }

  // check for trailing keyword/values

  wiggle = 0;
  wshear = 0;
  wtranslate = 0;
  wscontrol = 0;
  ftvarying = 0;
  fstr = NULL;
  velwall[0] = velwall[1] = velwall[2] = 0.0;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"wiggle") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix wall/gran command");
      if (strcmp(arg[iarg+1],"x") == 0) axis = 0;
      else if (strcmp(arg[iarg+1],"y") == 0) axis = 1;
      else if (strcmp(arg[iarg+1],"z") == 0) axis = 2;
      else error->all(FLERR,"Illegal fix wall/gran command");
      amplitude = force->numeric(FLERR,arg[iarg+2]);
      period = force->numeric(FLERR,arg[iarg+3]);
      wiggle = 1;
      loINI = lo;
      hiINI = hi;
      iarg += 4;
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
      velwall[0] = force->numeric(FLERR,arg[iarg+1]);
      velwall[1] = force->numeric(FLERR,arg[iarg+2]);
      velwall[2] = force->numeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"stresscontrol") == 0) {
      wscontrol = 1;
      wtranslate = 1;
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
  if (wtranslate && (lo != -BIG && hi != BIG))
    error->all(FLERR,"Cannot specify both top and bottom walls and translate for fix wall/gran");
  if (wtranslate && wallstyle == ZCYLINDER)
    error->all(FLERR,"Cannot use translate with cylinder fix wall/gran");
  if (wtranslate && (wiggle || wshear))
    error->all(FLERR,"Cannot translate and wiggle or shear fix wall/gran");

  // setup oscillations

  if (wiggle) omega = 2.0*MY_PI / period;

  //~ Find the number of shear quantities required [KH - 30 October 2013]
  numshearquants = 3;

  //~ pair/gran/shm/history has 4 shear quantities
  if (force->pair_match("shm",0)) numshearquants++;

  //~ pair/gran/CM/history has 5 shear quantities [MO - 18 July 2014] (,1) means exact match
  if (force->pair_match("gran/CM/history",1)) numshearquants += 2;
  
  //~ pair/gran/HMD/history has 26 shear quantities [MO - 21 July 2014]
  if (force->pair_match("gran/HMD/history",1)) numshearquants += 23;

  //~ pair/gran/CMD/history has 26 shear quantities [MO - 30 November 2014]
  if (force->pair_match("gran/CMD/history",1)) numshearquants += 23;

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
    if (*rolling) numshearquants += 15;

    // Option of D_spin was added [MO - 4 December 2014]
    D_spin = (int *) pair->extract("D_spin",dim);
    if (*D_spin) numshearquants += 20;

    trace_energy = (int *) pair->extract("trace_energy",dim);
    if (*trace_energy) numshearquants += 4;
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

  // perform initial allocation of atom-based arrays
  // register with Atom class

  shear = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  atom->add_callback(1);

  /*~ Note that there is no need to set up fix_old_omega as the pairstyle
    will do this automatically*/

  /*~ At present, not allowed to have cylindrical wall with rolling
    resistance model*/
  if (*rolling && wallstyle == ZCYLINDER)
    error->all(FLERR,"Not permitted to use rolling resistance with cylindrical walls");
  // Also for D_spin model
  if (*D_spin && wallstyle == ZCYLINDER)
    error->all(FLERR,"Not permitted to use D_spin resistance with cylindrical walls");

  // initialize as if particle is not touching wall

  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++)
    for (int q = 0; q < numshearquants; q++)
      shear[i][q] = 0.0; //~ [KH - 30 October 2013]

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

  // delete locally stored arrays

  memory->destroy(shear);
  delete [] fstr;
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
  dt = update->dt;

  // check variables for Ftarget

  if (fstr) {
    fvar = input->variable->find(fstr);
    if (fvar < 0)
      error->all(FLERR,"Variable name for fix wall/gran does not exist");
    if (!input->variable->equalstyle(fvar)) error->all(FLERR,"Variable for fix wall/gran is invalid style");
  }

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // set pairstyle from granular pair style

  if (force->pair_match("gran/hooke",1))
    pairstyle = HOOKE;
  else if (force->pair_match("gran/hooke/history",1))
    pairstyle = HOOKE_HISTORY;
  else if (force->pair_match("gran/hooke/history/omp",1))
    pairstyle = HOOKE_HISTORY;
  else if (force->pair_match("gran/hertz/history",1))
    pairstyle = HERTZ_HISTORY;
  else if (force->pair_match("gran/hertz/history/omp",1))
    pairstyle = HERTZ_HISTORY;
  else if (force->pair_match("gran/shm/history",1))
    pairstyle = SHM_HISTORY; //~ Extra entry added [KH - 30 October 2013]
  else if (force->pair_match("gran/CM/history",1))
    pairstyle = CM_HISTORY; //~ Extra entry added [MO - 18 July 2014]
  else if (force->pair_match("gran/HMD/history",1))
    pairstyle = HMD_HISTORY; //~ Extra entry added [MO - 21 July 2014]
  else if (force->pair_match("gran/CMD/history",1))
    pairstyle = CMD_HISTORY; //~ Extra entry added [MO - 30 November 2014]
  else error->all(FLERR,"Fix wall/gran is incompatible with Pair style");
}

/* ---------------------------------------------------------------------- */

void FixWallGran::setup(int vflag)
{
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
  // virial setup
  if (vflag) v_setup(vflag);
  else evflag = 0;

  /*~ Must update shearupdate before using with move_wall function
    [KH - 27 November 2013]*/
  shearupdate = 1;
  if (update->setupflag) shearupdate = 0;

  double dx,dy,dz,del1,del2,delxy,delr,rsq;

  // if wiggle or shear, set wall position and velocity accordingly
  // if wtranslate lo and hi track the wall position and velwall is set in the constructor
  if (wiggle) {
    double arg = omega * (update->ntimestep - time_origin) * dt;
    if (wallstyle == axis) {
      lo = loINI + amplitude - amplitude*cos(arg);
      hi = hiINI + amplitude - amplitude*cos(arg);
    }
    velwall[axis] = amplitude*omega*sin(arg);
  } else if (wtranslate || wscontrol) {
    if (shearupdate) move_wall(); // move_wall will update hi & lo
  } else if (wshear) velwall[axis] = vshear;

  fwall[0] = fwall[1] = fwall[2] = 0.0; //per-processor force// fwall_all[0] = fwall_all[1] = fwall_all[2] = 0.0;

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
        if (delr > radius[i]) dz = cylradius;
        else {
          dx = -delr/delxy * x[i][0];
          dy = -delr/delxy * x[i][1];
          if (wshear && axis != 2) {
            velwall[0] = vshear * x[i][1]/delxy;
            velwall[1] = -vshear * x[i][0]/delxy;
            velwall[2] = 0.0;
          }
        }
      }

      rsq = dx*dx + dy*dy + dz*dz;

      if (rsq > radius[i]*radius[i]) {
        if (pairstyle != HOOKE) {
	  for (int q = 0; q < numshearquants; q++)
	    shear[i][q] = 0.0; //~ Added the 'for' loop [KH - 4 November 2013]
        }
      } else {
        if (pairstyle == HOOKE)
          hooke(rsq,dx,dy,dz,velwall,v[i],f[i],omega[i],torque[i],
                radius[i],rmass[i],i);
        else if (pairstyle == HOOKE_HISTORY)
          hooke_history(rsq,dx,dy,dz,velwall,v[i],f[i],omega[i],torque[i],
                        radius[i],rmass[i],shear[i],i);
        else if (pairstyle == HERTZ_HISTORY)
          hertz_history(rsq,dx,dy,dz,velwall,v[i],f[i],omega[i],torque[i],
                        radius[i],rmass[i],shear[i],i);
	else if (pairstyle == SHM_HISTORY) //~ [KH - 30 October 2013]
          shm_history(rsq,dx,dy,dz,velwall,v[i],f[i],omega[i],torque[i],
		      radius[i],rmass[i],shear[i],i);
	else if (pairstyle == CM_HISTORY) //~ [MO - 18 July 2014]
          CM_history(rsq,dx,dy,dz,velwall,v[i],f[i],omega[i],torque[i],
		     radius[i],rmass[i],shear[i],i);
	else if (pairstyle == HMD_HISTORY) //~ [MO - 21 July 2014]
          HMD_history(rsq,dx,dy,dz,velwall,v[i],f[i],omega[i],torque[i],
		      radius[i],rmass[i],shear[i],i);
	else if (pairstyle == CMD_HISTORY) //~ [MO - 30 N0vember 2014]
          CMD_history(rsq,dx,dy,dz,velwall,v[i],f[i],omega[i],torque[i],
		      radius[i],rmass[i],shear[i],i);
      }
    }
  }
  if (wscontrol) velscontrol();   // velocity calculation for next timestep
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
                        double radius, double mass, int i)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double meff,damp,ccel,vtr1,vtr2,vtr3,vrel;
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

  meff = mass;
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
                                double radius, double mass, double *shear, int i)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double meff,damp,ccel,vtr1,vtr2,vtr3,vrel;
  double fn,fs,fs1,fs2,fs3,fx,fy,fz;
  double shrmag,rsht,rinv,rsqinv;

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

  meff = mass;
  damp = meff*gamman*vnnr*rsqinv;
  ccel = kn*(radius-r)*rinv - damp;

  // relative velocities

  vtr1 = vt1 - dz*omega[1]+dy*omega[2];
  vtr2 = vt2 - dx*omega[2]+dz*omega[0];
  vtr3 = vt3 - dy*omega[0]+dx*omega[1];
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);

  // shear history effects

  double oldfs = kt*sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
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

  fs1 = - (kt*shear[0] + meff*gammat*vtr1);
  fs2 = - (kt*shear[1] + meff*gammat*vtr2);
  fs3 = - (kt*shear[2] + meff*gammat*vtr3);

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
  double db[3], localdM[3], globaldM[3]; //~ Pass by reference
  if (*rolling && shearupdate)
    rolling_resistance(i,numshearquants,dx,dy,dz,r,radius,ccel,fn,
		       kt,torque,shear,db,localdM,globaldM);

  //~ Add contributions to traced energy [KH - 20 February 2014]
  double aveshearforce, slipdisp, incdissipf, nstr, sstr;
  if (pairenergy) {
    /*~ Increment the friction energy only if the slip condition
      is invoked*/
    if (fs > fn && fn > 0.0) {
      slipdisp = (fs-fn)/kt;
      aveshearforce = 0.5*(fn + oldfs);

      //~ slipdisp and aveshearforce are both positive
      incdissipf = aveshearforce*slipdisp;
      dissipfriction += incdissipf;
      if (*trace_energy) shear[3] += incdissipf;
    }

    /*~ Update the strain energy terms which don't need to be 
      calculated incrementally*/
    nstr = 0.5*kn*(radius-r)*(radius-r);
    sstr = 0.5*(fs1*fs1 + fs2*fs2 + fs3*fs3)/kt;
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
                                double *vwall, double *v,
                                double *f, double *omega, double *torque,
                                double radius, double mass, double *shear, int i)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double meff,damp,ccel,vtr1,vtr2,vtr3,vrel;
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

  meff = mass;
  damp = meff*gamman*vnnr*rsqinv;
  ccel = kn*(radius-r)*rinv - damp;
  polyhertz = sqrt((radius-r)*radius);
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
    rolling_resistance(i,numshearquants,dx,dy,dz,r,radius,ccel,fn,
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

void FixWallGran::shm_history(double rsq, double dx, double dy, double dz,
			      double *vwall, double *v,
			      double *f, double *omega, double *torque,
			      double radius, double mass, double *shear, int i)
{
  //~ Added this function for shm history [KH - 30 October 2013]

  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double meff,damp,ccel,vtr1,vtr2,vtr3,vrel;
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

  int ctcorrection = 0;
  if (shearupdate) {
    /*~ Apply Colin Thornton's suggested correction (see
      Eq. 18 of 2013 P. Tech. paper) [KH - 30 October 2013]*/
    if (shear[3] > polyhertz) {
    /*~ Note that as polyhertz is >= 0, there is no need to
      check for shear[3] == 0 in the expressions below*/
      shear[0] *= polyhertz/shear[3];
      shear[1] *= polyhertz/shear[3];
      shear[2] *= polyhertz/shear[3];
      ctcorrection = 1;
    }
    
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
    rolling_resistance(i,numshearquants,dx,dy,dz,r,radius,ccel,fslim,
		       effectivekt,torque,shear,db,localdM,globaldM);
  //~ Call function for D_spin resistance model [MO - 30 November 2013]

  double dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy,a,N;
  a = polyhertz;
  N = ccel*r;
  if (*D_spin && shearupdate) 
    Deresiewicz1954_spin(i,numshearquants,dx,dy,dz,radius,r,torque,shear,dspin_i,
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

      /*~ If Colin Thornton's shear force rescaling has been used, the adjustment
	made to the energy is added to dissipfriction so that energy will still
	be balanced [KH - 12 March 2014]*/
      double fsunscaled[3], fsunscaledmag, a, b, c;
      double d = 0.0;
      if (ctcorrection) {
	b = shear[3]/polyhertz;
	fs > fslim ? a = b*fs/fslim : a = b;
	c = effectivekt*dt*(b - 1.0);
	
	fsunscaled[0] = a*shear[0] + c*vtr1;
	fsunscaled[1] = a*shear[1] + c*vtr2;
	fsunscaled[2] = a*shear[2] + c*vtr3;
	fsunscaledmag = sqrt(fsunscaled[0]*fsunscaled[0] + fsunscaled[1]*fsunscaled[1] + fsunscaled[2]*fsunscaled[2]);
	
	if (fs > fslim && fslim > 0.0) {
	  d += 0.5*rkt*(fsunscaledmag + fslim)*(fsunscaledmag - fslim) - incdissipf;
	  fsunscaledmag *= fslim/fsunscaledmag;
	}
	
	d += 0.5*rkt*(fsunscaledmag - oldshearforce)*(fsunscaledmag + oldshearforce) - sstr;
	dissipfriction += d;
	if (trace_energy) shear[4] += d;
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
			     double radius, double mass, double *shear, int i)
{
  //~ Added this function for CM history [MO - 18 July 2014]

  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double meff,damp,ccel,vtr1,vtr2,vtr3,vrel;
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
  // The current version of CM model is wrriten by [MO - 03 December 2014]
  // The roughness of the wall is considered as null (RMSf_eq is not used in this fix_wall_gran.cpp )

  double overlap = radius-r; // special case for ball to particle
  double R_star = radius;    // special case for ball to particle
  double E_star = Geq/(1.0-Poiseq);
  double pi = 4.0*atan(1.0);
  double overlap_p = R_star*pow(3.0/4.0*pi*Hp/E_star,2.0);     // R_star*(3/4*pi*Hp/E_star)^2
  double N_GT = 100.0*RMSf*E_star*sqrt(2.0*R_star*RMSf);
  double overlap_GT = pow(3.0*N_GT/(4.0*sqrt(R_star)*E_star),2.0/3.0)+overlap_p; 
  double b = 2.0*E_star*sqrt(R_star*(overlap_GT-overlap_p))*overlap_GT/N_GT; 
  double overlap_max = fabs(shear[3]);
  double overlap_old = fabs(shear[4]);
  double effectivekt;
  double N_max;
  double overlap_hertz_max;
  double overlap_offset;
  double overlap_hertz;
  double tolerance = 1.0e-20; 
  int N_step;

  /*~~Update the overlap which is the maximum in the history. [MO - 5 June 2014] ~~*/
	
  if (overlap >= overlap_max) overlap_max = overlap;
  
  /*~~ step 1*, 2* and 3* stand for loading, unloading and reloasing, respectively. 
    step *4, *5 and *6 stand for asperity contact, Hertzian contact and zero force, respectively.
    Possible conbinations are 14,15,25,26,35 and 36.  [MO - 18 Jun 2014]~~*/
  
  N_max = N_GT * pow(overlap_GT, -b) * pow(overlap_max, b);   
  overlap_hertz_max = pow(N_max/(kn*sqrt(R_star)),2.0/3.0);              
  overlap_offset = overlap_max - overlap_hertz_max;   
  
  if (overlap >= overlap_GT) {
    if (overlap >= overlap_max - tolerance) N_step = 15;
    else if (overlap < overlap_max) {
      if (overlap >= overlap_old) N_step = 35;
      else if (overlap < overlap_old) N_step = 25;
    }
    overlap_hertz = overlap - overlap_p;
    polyhertz = sqrt(overlap_hertz*R_star); //[MO - 18 July 2014] R_star is now used
    effectivekt = polyhertz*kt;
  }
  else if (overlap < overlap_GT) {
    if (overlap >= overlap_max - tolerance) { 
      N_step = 14;
      overlap_hertz = overlap - overlap_offset;
      polyhertz = sqrt(overlap_hertz*R_star);
      //ccel = N_GT * pow(overlap_GT, -b) * pow(overlap, b)*rinv;
      effectivekt = 2.0*b*(1.0-Poiseq)/(2.0-Poiseq)*ccel*r*(1.0/overlap); 
    }
    else if (overlap < overlap_max - tolerance) { 
      if (overlap < overlap_offset){
	if (overlap >= overlap_old) N_step = 36;
	else if (overlap < overlap_old) N_step = 26;
	// Particles are separated due to squahed asperities.
	overlap_hertz = tolerance;
	polyhertz = sqrt(overlap_hertz*R_star);    
	effectivekt = polyhertz*kt;
      }
      else if (overlap >= overlap_offset){
	if (overlap >= overlap_old) N_step = 35;
	else if (overlap < overlap_old) N_step = 25;
	overlap_hertz = overlap - overlap_offset;
	polyhertz = sqrt(overlap_hertz*R_star);
	effectivekt = polyhertz*kt;
      }
    }
  }
  //**************************************
  ccel = kn*overlap_hertz*polyhertz*rinv;
  //**************************************
  if (shearupdate) {
    shear[0] -= effectivekt*vtr1*dt;//shear displacement =vtr*dt
    shear[1] -= effectivekt*vtr2*dt;
    shear[2] -= effectivekt*vtr3*dt;
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

   
  //~ Add contributions to traced energy [KH - 20 February 2014]
  double aveshearforce, slipdisp, oldshearforce, newshearforce;
  double incdissipf, nstr, sstr, incrementaldisp;
  double b1inv = 1.0/(b+1.0);
  
  double dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy,a,N;
  a  = polyhertz;
  N = ccel*r;

  if (*D_spin && shearupdate) 
    Deresiewicz1954_spin(i,numshearquants,dx,dy,dz,radius,r,torque,shear,dspin_i,
			 dspin_stm,spin_stm,dM_i,dM,K_spin,theta_r,
			 M_limit,Geq,Poiseq,Dspin_energy,a,N);

  if (pairenergy) {
    
    /*~ Increment the friction energy only if the slip condition
      is invoked*/
    oldshearforce = sqrt(shsqmag);
    if (fs > fslim && fslim > 0.0) {
      //~ current shear displacement = fslim/effectivekt;
      slipdisp = (fs-fslim)/effectivekt;
      aveshearforce = 0.5*(oldshearforce + fslim);
      
      //~ slipdisp and aveshearforce are both positive
      incdissipf = aveshearforce*slipdisp;
      dissipfriction += incdissipf;
      if (trace_energy) shear[5] += incdissipf;
    }
    /*~ Update the normal contribution to strain energy which 
      doesn't need to be calculated incrementally*/
    /*~~ However, the hysterisys should be considered using CM model [MO - 18 Jun 2014] ~~*/    
    if(N_step == 14){        // 14 stands for loading on asperity contact   [MO - 18 Jun 2014]
      nstr = N_GT*b1inv*pow(overlap_GT,-b)*pow(overlap,b+1.0);
      normalstrain += nstr;
    }
    if(N_step == 15){        // *5 stands for hertzian contact   [MO - 18 Jun 2014]
      // overlap-overlap_p for Hertzian curve [MO - 18 Jun 2014]
      nstr = N_GT*overlap_GT*b1inv
	-0.4*kn*sqrt(R_star)*(pow(overlap_GT-overlap_p,2.5)-pow(overlap-overlap_p,2.5));  
    }	    
    if(N_step == 26 || N_step == 36){    // 2* and 3* stand for unloading and reloading. *6 means ccel = 0.0 which should not be used in this group [MO - 18 Jun 2014] 
      if(overlap_max < overlap_GT){
      	nstr = N_GT*b1inv*pow(overlap_GT,-b)*pow(overlap_max,b+1.0)
	  -0.4*kn*sqrt(R_star)*pow(overlap_max - overlap_offset,2.5);
      }
      if(overlap_max >= overlap_GT){
	nstr = N_GT*overlap_GT*b1inv-0.4*kn*sqrt(R_star)*pow(overlap_GT-overlap_p,2.5);
      }
    }
    if(N_step == 25 || N_step == 35){        // *5 stands for hertzian contact [MO - 18 Jun 2014]
      // overlap-overlap_effective for Hertzian contact [MO - 18 Jun 2014]
      if(overlap_max < overlap_GT){
	nstr = N_GT*b1inv*pow(overlap_GT,-b)*pow(overlap_max,b+1.0)
	  -0.4*kn*sqrt(R_star)*(pow(overlap_max-overlap_offset,2.5)-pow(overlap-overlap_offset,2.5));      
      }	    
      if(overlap_max >= overlap_GT){
	nstr = N_GT*overlap_GT*b1inv
	  -0.4*kn*sqrt(R_star)*(pow(overlap_GT-overlap_p,2.5)-pow(overlap-overlap_p,2.5));
      }
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
  
  shear[3] = overlap_max;   
  shear[4] = overlap;
  
  fwall[0] += fx;
  fwall[1] += fy;
  fwall[2] += fz;
 
  if (evflag) ev_tally_wall(i,fx,fy,fz,dx,dy,dz,radius);
}

/* ---------------------------------------------------------------------- */

void FixWallGran::HMD_history(double rsq, double dx, double dy, double dz,
			      double *vwall, double *v,
			      double *f, double *omega, double *torque,
			      double radius, double mass, double *shear, int i)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double meff,damp,ccel,vtr1,vtr2,vtr3,vrel;
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
  /* The main part of the HMD model is wrriten down below as of 13th August 2014 by Masahide [MO]*/ 
  //*******************************************************************************************

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
  //***********************************************************************************************************************************
  // normal force = Hertzian contact 
  double R_star = radius;
  double E_star = Geq/(1.0-Poiseq);             // equivalent Young's modulus at contact
  double G_star = Geq/(2.0*(2.0-Poiseq));       // equivalent shear modulus at contact
  double pi = 4.0*atan(1.0);                    // pi = 3.1415*******
  double overlap = radius-r;                  
  double overlap_old = abs(shear[3]);                // overlap of spheres at the previous step 
  double a = sqrt(R_star*overlap);              // radius of contact circle
  double a_old = sqrt(R_star*overlap_old);      // radius of contact circle at the previous step 
  double N = kn*overlap*a;                      // normal force
  double N_old = kn*overlap_old*a_old;          // normal force at the previous step
  double dN = N - N_old;                        // change of normal force
  double K_N = 2.0*E_star*a;                    // normal contact stiffness (tangential, not secant)
  polyhertz = a;                                // re-use polyhertz to adjust the format for shm contact model
  ccel = kn*overlap*a*rinv;                     // previously, ccel = kn*overlap*polyhert*rinv;	
  //**********************************************************************************************************************
  // The following codes introduce HMD model for tangential component.
  double T = 0.0;                               // bulk tangential force
  double T_old = shear[12];                     // bulk tangential force at the previous step (including its sign)
  double T_star1 = shear[8];                    // T*:  tangential force at the revasal step from loading to unloading 
  double T_star2 = shear[9];                    // T**: tangential force at the revasal step from unloading to re-loading 
  double T_star3 = shear[10];                   // 0.0 if not active 
  double T_star4 = shear[11];                   // 0.0 if not active
  double Tdisp  = 0.0;                          // bulk tangential displacement	
  double Tdisp_mag =0.0;                        // magnitude of bulk tangential displacement
  double Tdisp_old  = shear[4];                 // bulk tangential displacement at the previous step
  double dTdisp = 0.0;                          // increment of bulk displacement
  double Tdisp1 = 0.0;                          // tangential displacement of x direction
  double Tdisp2 = 0.0;                          // tangential displacement of y direction
  double Tdisp3 = 0.0;                          // tangential displacement of z direction
  double Tdisp1_old = shear[5];                 // tangential displacement of x direction at the previous step
  double Tdisp2_old = shear[6];                 // tangential displacement of y direction at the previous step
  double Tdisp3_old = shear[7];                 // tangential displacement of z direction at the previous step     
  double inter_product = 0.0;                   // inter product between current and previous vectors of tangential displacement
  double tolerance = 1.0e-20;
  int T_step = 0;                               // the current loading phase, e.g. 11 = N increasing T increasing
  int T_step_old = static_cast<int>(fabs(shear[15])); // the loading case at the previous step
  int slip_T = static_cast<int>(fabs(shear[10]));     // slip_T means fully slipped due to tangential contact force
  int UFL = 0;                                  // UFL is the direction of tangential load
  int zero = 0;                                 // zero = 1 when tangential displacement is zero to avoid a numerical error 	
  int CTD = static_cast<int>(fabs(shear[14]));         // CTD is the direction of tangential displacement 
  if (CTD == 10000) CTD = -1;
  int CDF = static_cast<int>(fabs(shear[11]));         // CDF is the direction of tangential load-displacement system
  if (CDF == 10000) CDF = -1;
	
  //***************************************
  // First, calculate the shear displacement
  if (xmu < tolerance) { // Tdisp is not accumulated if xmu = 0
    if (shearupdate){                              // Update the tangential displacement  
      Tdisp1 = Tdisp1_old + vtr1*dt;               // shear displacement =vtr*dt
      Tdisp2 = Tdisp2_old + vtr2*dt;
      Tdisp3 = Tdisp3_old + vtr3*dt;
      Tdisp_mag = sqrt(Tdisp1*Tdisp1 + Tdisp2*Tdisp2 + Tdisp3*Tdisp3);
      
      if (Tdisp_mag <= tolerance){                   // Tdisp = 0.0 cause numerial error later on, so avoid that here
	Tdisp1 = 1.0e-25;
	Tdisp2 = 0.0;
	Tdisp3 = 0.0;
	Tdisp_mag = 1.0e-25;
	zero = 1;                                  // zero = 1 is a special case 
      }
    }else{
      Tdisp1 = Tdisp1_old;
      Tdisp2 = Tdisp2_old;
      Tdisp3 = Tdisp3_old;
      Tdisp_mag = sqrt(Tdisp1*Tdisp1 + Tdisp2*Tdisp2 + Tdisp3*Tdisp3);
    }    
    // inter_product becomes negative if the direction of tangential loading changes
    inter_product = Tdisp1*Tdisp1_old + Tdisp2*Tdisp2_old + Tdisp3*Tdisp3_old;
    
    if (shearupdate){ 
      if (Tdisp_mag <= tolerance && fabs(Tdisp_old) <= tolerance){
	CTD = 1;                                     // when there is no shear disp at all.
	CDF = 1;
      }
      else if (Tdisp_mag > tolerance && fabs(Tdisp_old) <= tolerance){    
	CTD = 1;                                   // this is for the first increment of tangential displacement
	CDF = 1;
      }
      else if (inter_product >= 0.0)                   CTD *= 1;
      else if (inter_product < 0.0)                    CTD *= -1;
      else fprintf(screen,"Unexpected case occurred in zone A. ERROR!!");
      
      Tdisp =  CTD * Tdisp_mag;                      // give the sign for tangential displacement based on the value of the inter product 
      dTdisp = Tdisp - Tdisp_old;                    // dTdisp includes direction of tangential displacement (not magnitude)
    }
  }
  else {
    dTdisp = Tdisp = Tdisp1 = Tdisp2 = Tdisp3 = Tdisp_mag = 0.0; // if xmu >=  0
    zero = 1;
  }
  //*******************************************************************************************************
  // Update the dTdisp_DD
  
  /* Fig.7 of Mindlin & Deresiewicz (1953) explain why this special case is needed.
     Tdisp_DSC is the minimum tangential displacement to move onto a new loading curve of N+dN.
     If the increment of the tangential displacement is less than Tdisp_DSC, the resultant tangential force is less than theoretical value.
     The maximum tangential contact stiffness (8.0*G_star*a) is used until the current tangential force catches up the the new loading curve. */
  double Tdisp_DSC = xmu*dN/(8.0*G_star*a);      // Tdisp_DSC is either positive or negative depending upon the sign of dN
  double Tdisp_DD = fabs(shear[13]) - fabs(dTdisp) + Tdisp_DSC;
  int    special_DD = 0; 
  if (Tdisp_DD < 0.0) Tdisp_DD = 0.0;            // Tdisp_DD <= 0.0 should be satisfied to move onto a new loading curve for N+dN.
  else                special_DD = 1;            // Special case for DD value is now active               
  //*******************************************************************************************************
  double theta1 = 1.0;
  double theta2 = 1.0;
  double theta3 = 1.0;                  
  double theta_t  = 1.0;                           // reduction ratio of tangential cotnact stiffness due to tangential load 
  //*******************************************************************************************************
  if (xmu < tolerance) theta_t = 1.0;
  else {
    // Idnetify the loading steps for tangential component
    // for the first step of the special case
    
    if (special_DD == 1){
      if (dN > 0.0 && T_step_old != 115 && T_step_old != 125 && T_step_old != 135 && T_step_old != 145){
	if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance)          T_step = 115;
	else if (CDF*dTdisp <  0.0 && fabs(T_star2) < tolerance)                                  T_step = 125; 
	else if (CDF*dTdisp >= 0.0 && fabs(T_old) < fabs(T_star1))                                T_step = 135;
	else if (CDF*dTdisp <  0.0 && fabs(T_old) <= fabs(T_star1) && fabs(T_star2)>= tolerance)  T_step = 145;
	else fprintf(screen,"Unexpected case occurs in zone B. ERROR!!");
	theta_t = 1.0;
	// if the special case was invoked in the previous step and is still active
      }else if ((T_step_old == 115 || T_step_old == 135) && CDF*dTdisp >= 0){  
	// still in step_115 or 135; if moved to unloading of T, go to the usual case
	T_step = T_step_old;
	theta_t = 1.0;
      }else if ((T_step_old == 125 || T_step_old == 145) && CDF*dTdisp < 0){   
	// still in step_125 or 145; if moved to re-loading of T, go to the usual case
	T_step = T_step_old;                                           
	theta_t = 1.0;
      }else if (T_step_old == 115 && CDF*dTdisp < 0){
	// the first step from loading of special case to unloading of special case.
	T_step = 125;
	theta_t = 1.0;
      }else if (T_step_old == 125 && CDF*dTdisp >= 0){
	// the first step from unloading of special case to reloading of special case.
	T_step = 135;
	theta_t = 1.0;
      }else if (T_step_old == 135 && CDF*dTdisp < 0){
	// the first step from reloading  of special case to re-unloading of special case.
	T_step = 145;
	theta_t = 1.0;
      }else if (T_step_old == 145 && CDF*dTdisp > 0){
	// the first step from re-unloading  of special case to re-reloading of special case.
	T_step = 135;
	theta_t = 1.0;
      }
      //*************************************************************************************
    }else {  // Usual cases 
      // T loading: T_step = *1 
      if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance) { 
	theta1 = 1.0-(CDF*T_old+xmu*dN)/(xmu*N);	
	if (theta1 <= 0.0) theta1 = theta_t = tolerance;
	else theta_t = pow(theta1, 1.0/3.0);
	if       (dN >  tolerance)      T_step = 11;   // N increasing
	else if  (dN < -tolerance)      T_step = 21;   // N decreasing
	else                          T_step =  1;   // N constant
      }
      // T unloading: T_step = *2 
      else if (CDF*dTdisp < 0.0 && fabs(T_star2) < tolerance) {
	// For the first step from loading to unloading, T* is still null, which should be T* = T_old.
	if  (fabs(T_star1) > tolerance) theta2 = 1.0-(CDF*(T_star1-T_old)+2.0*xmu*dN)/(2.0*xmu*N);
	else                           theta2 = 1.0-(2.0*xmu*dN)/(2.0*xmu*N);
	if (theta2 <= 0.0) theta2 = theta_t = tolerance;	  
	else               theta_t  = pow(theta2, 1.0/3.0);
	if      (dN >  tolerance) T_step = 12; 
	else if (dN < -tolerance) T_step = 22;      
	else                    T_step =  2;
      }
      // T re-loading: T_step = *3
      else if (CDF*dTdisp >= 0.0 && fabs(T_old) < fabs(T_star1)) {
	// For the first step from unloading to re-loading, T** is still null, which should be T** = T_old.
	if  (fabs(T_star2) > tolerance) theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu*dN)/(2.0*xmu*N);
	else                           theta3 = 1.0-(2.0*xmu*dN)/(2.0*xmu*N);
	if (theta3 <= 0.0) theta3 = theta_t = tolerance;
	else               theta_t  = pow(theta3, 1.0/3.0);
	if      (dN >  tolerance) T_step = 13; 
	else if (dN < -tolerance) T_step = 23;     
	else                    T_step = 3;
      }
      // T re-unloading: T_step = *4
      else if (CDF*dTdisp < 0.0 && fabs(T_old) < fabs(T_star1) && fabs(T_star2) >=  tolerance) {
	// This part is same with the above (T_step = *3) due to simplification.
	theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu*dN)/(2.0*xmu*N);
	if (theta3 <= 0.0) theta3 = theta_t = tolerance;
	else               theta_t  = pow(theta3, 1.0/3.0);
	if      (dN >  tolerance) T_step = 14;
	else if (dN < -tolerance) T_step = 24; 
	else                    T_step =  4;      
      }
      else if (slip_T == 1) {
	T_step = T_step_old;
	theta_t = tolerance;
      }
      else fprintf(screen,"Unexpected case occurred in zone C. ERROR!!");
    }
    
    //*********************************************************************************************
    //UFL is neccesary for the calculation of tangential contact stiffness
    
    if (T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) UFL = -1;
    else if (T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) UFL = -1; // Added [MO - 04 January 2014]
    else                                                              UFL =  1;
  }       // end of if function for xmu < tolerance
  
  //************************************************************************************************************************
  // Calculate the tangential contact stiffness and the resultant tangential contact force
  
  double K_T = 0.0;                             // tangential contact stiffness
  double dT = 0.0;                              // increment of tangential contact force
  double fs_ratio1;                             // ratio to re-scale the tangential force to avoid inconsistent magnitude of tangential force
  double fs_ratio2;                             // ratio to re-scale the tangential force if full sliding take places
  double K_R;                                   // ratio of tangential contact stiffness over one for shm model	
  double T_temp;                                // tangential force which exceeded fslim
  //************************************************************************************************************************
  // update T_old considring the rotation of contact plane which was done prior to the calculation of contact force
  if (T_old >= 0) T_old =   sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  else            T_old = - sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  //************************************************************************************************************************
  
  if (theta_t > 1.0) theta_t = 1.0;
    
  if (zero == 0) K_T = 8.0*G_star*theta_t*a + CDF*UFL*xmu*(1.0-theta_t)*dN/dTdisp;
  else           K_T = 0.0; // zero == 1 means dTdisp is zero and K_T becomes inf
  
  if (K_T > 8.0*G_star*a) K_T = 8.0*G_star*a;
  if (K_T < 0.0) K_T = 0.0;
  
  K_R = K_T / (8.0*G_star*a);
  
  if (shearupdate){
    if (zero == 0){
      dT = K_T * dTdisp; 
      T = dT + T_old;
      // update the tangential contact force by distributing dT into 3 components based on imcrese of relative rotational velosity
      shear[0] -= K_T * vtr1 * dt; 
      shear[1] -= K_T * vtr2 * dt;
      shear[2] -= K_T * vtr3 * dt;
      
      //rescale the component
      fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
      fs_ratio1 = fabs(T)/fs;
      shear[0] *= fs_ratio1; 
      shear[1] *= fs_ratio1;
      shear[2] *= fs_ratio1;
    }
  } else T = T_old;
  
  //************************************************************************************************************************
  // rescale tangential force if full sliding takes place 
  
  fslim = xmu * N;                               // ccel*r = N
  fs = fabs(T);                                  // re-use fs for re-scale the tangential force         
  fs_ratio2 = fslim/fs;
  T_temp = T;
  if (fs > fslim) {
    if (fs > tolerance) {
      if (shearupdate) {
	T *= fs_ratio2;
	shear[0] *= fs_ratio2; 
	shear[1] *= fs_ratio2;
	shear[2] *= fs_ratio2;   
	slip_T = 1;
      }
    } else shear[0] = shear[1] = shear[2] = 0.0;
  } else slip_T = 0;
  
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
  
  double effectivekt = K_T;
  double nstr, sstr;
  
  //~~ Call function for twisting resistance model [MO - 04 November 2014]   
  double dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy;
  if (*D_spin && shearupdate) 
    Deresiewicz1954_spin(i,numshearquants,dx,dy,dz,radius,r,torque,shear,dspin_i,
			 dspin_stm,spin_stm,dM_i,dM,K_spin,theta_r,
			 M_limit,Geq,Poiseq,Dspin_energy,a,N);
  
  //~ Add contributions to traced energy [KH - 20 February 2014]
  if (pairenergy) {
    /*~ Update the normal contribution to strain energy which 
      doesn't need to be calculated incrementally*/
    nstr = 0.4*kn*a*overlap*overlap;                             // peviously, nstr = 0.4*kn*polyhertz*deltan*deltan;
    normalstrain += nstr;
    if (trace_energy) shear[27] = nstr;

    //~~ Update the spin contribution [MO - 13 November 2014]
    if (*D_spin) {
      spinenergy += Dspin_energy;
      if (trace_energy) shear[29] += Dspin_energy;
    }
    /* Full sliding can be considered as accumulation of partial slip for HMD model.
       Theoretically speaking, full sliding does not take place for HMD modle.
       Thus, shear strain energy here is summation of shear strain energy + friction energy for shm model.*/  
    sstr = 0.5*dTdisp*(T + T_temp);
    shearstrain += sstr;
    if (trace_energy) shear[28] += sstr;
  }
  //*********************************************************************************************************
  // Update the T_star1 and T_star2 considering the change of the normal contact load
  if (T_step != 1 && T_step != 11 && T_step != 21 && T_step != 115 && fabs(T_star1)> tolerance)    T_star1 += CDF*xmu*dN;
  if ((T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) && fabs(T_star2) > tolerance) T_star2 -= CDF*xmu*dN;
  if ((T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) && fabs(T_star2) > tolerance) T_star2 -= CDF*xmu*dN;
  //*********************************************************************************************************
  // Update T_star1 and T_star2
  int sp_Tstar1 = 0;                         // CDF *= -1 should not be active if sp_Tstar1 = 1 is set.  
  if (T_step == 1 || T_step == 11 || T_step == 21 || T_step == 115) {   // do not forget to include T_step == 115
    T_star1 = 0.0;
    T_star2 = 0.0;
  }
  // Update the T_star1 and T_star2 for new loading curve: move to new loading step from re-loading step
  if ((T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) && fabs(T) > fabs(T_star1) && fabs(T_star1) > tolerance) {
    T_star1 = 0.0;
    T_star2 = 0.0;
  }
  // Update the T_star2 for new unloading curve: move to unloading step
  if ((T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) && fabs(T) > fabs(T_star2) && fabs(T_star2) > tolerance) {
    T_star2 = 0.0;
  }
  if ((T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) && fabs(T_star1) <= tolerance && fabs(T_star2) <= tolerance) {
    // the first step from T_step = 1: loading to unloading
    T_star1 = T_old;
    Tdisp_DD = 0.0;
    sp_Tstar1 = 1;
  } 
  if ((T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) && fabs(T_star1) > fabs(T) && fabs(T_star2) <= tolerance) {  
    // the first step from T_step = 2: unloading to re-loading
    T_star2 = T_old;
    Tdisp_DD = 0.0;
  } 
  //********************************************************************************************************
  //special case when |T| for unloading reaches -|T_star1|  // move on to a new loading curve in the opposite direction
  if ((UFL == -1) && fabs(T) > fabs(T_star1) && fabs(T_star1) > tolerance){
    // avoid the case when T* > T > T*-xmu*dN 
    if (sp_Tstar1 == 0) {  
      CDF *= -1;  // the system is moved onto virgin loading curve
      T_star1 = 0.0;
      T_star2 = 0.0;
    }
  }
  //*************************************************************************
  // Added this to avoid numerical errors [MO - 04 January 2015]
  if (fabs(T_star1 - T_star2) < tolerance) {
    T_star1 = 0.0;
    T_star2 = 0.0;
  }
  //*********************************************************************************************************
  /*~~ modified for MD model [MO - 17 July 2014] ~~*/
  if (zero == 1)  Tdisp = Tdisp1 = Tdisp2 = Tdisp3 = 0.0;

  // Store the CTD and CDF as absolute values for the next step to avoid the sign changing [MO - 15 December 2014] 
  if (CTD == -1) CTD = 10000;
  if (CDF == -1) CDF = 10000;

  // shear[0], shear[1] and shear[2] are tangential force of x, y and z directions, espectively.
  shear[3] = overlap;           // previous overlap
  shear[4] = Tdisp;             // previous tangential displacement
  shear[5] = Tdisp1;            // tangential displacement of x direction in the previous step
  shear[6] = Tdisp2;            // tangential displacement of y direction in the previous step
  shear[7] = Tdisp3;            // tangential displacement of z direction in the previous step
  shear[8] = T_star1;           // first reverse point of tangential load (T) from loading to unloading
  shear[9] = T_star2;           // second reverse point of tangential load from unloading to re-loading
  shear[10] = slip_T;           // if full slip is mobilized, slip_T = 1 (int);
  shear[11] = CDF;              // CDF indicates wheather system is loading or unloading //T_star4;       
  shear[12] = T;       
  shear[13] = Tdisp_DD;         // Tdisp_DD <= 0.0 should be satisfied to move on a new loading curve for N+dN 
  shear[14] = CTD;              // +1 and -1 mean positive and negative shear disp, respectively.       
  shear[15] = T_step;
  shear[16] = 0;       
  shear[17] = a;
  shear[18] = N;
  shear[19] = 0;
  shear[20] = 0;
  shear[21] = ccel;
	      
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
			      double radius, double mass, double *shear, int i)
{
  double r,vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double meff,damp,ccel,vtr1,vtr2,vtr3,vrel;
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
  // The main part of the CMD model is wrriten down below as of 13th August 2014 by Masahide [MO]
  //*******************************************************************************************

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
  //**************************************************************************
  // normal force = Caverreta et al.(2010) Refer to O'Donovan (2013)  
  //**************************************************************************
  double overlap = radius-r;
  double R_star = radius;
  double E_star = Geq/(1.0-Poiseq);
  double G_star = Geq/(2.0*(2.0-Poiseq));       // equivalent shear modulus at contact
  double pi = 4.0*atan(1.0);
  double overlap_p = R_star*pow(3.0/4.0*pi*Hp/E_star,2.0);  // R_star*(3/4*pi*Hp/E_star)^2
  double N_GT = 100.0*RMSf*E_star*sqrt(2.0*R_star*RMSf);
  double overlap_GT = pow(3.0*N_GT/(4.0*sqrt(R_star)*E_star),2.0/3.0)+overlap_p; 
  double b = 2.0*E_star*sqrt(R_star*(overlap_GT-overlap_p))*overlap_GT/N_GT; 
  double overlap_max = fabs(shear[16]);
  double overlap_old = fabs(shear[3]);          // shear[4] in CM model
  double N_max;
  double overlap_hertz_max;
  double overlap_offset;
  double overlap_hertz;
  double effectivekt;
  double tolerance = 1.0e-20; 
  double b1inv = 1.0/(b+1.0);
  int N_step;
  //**************************************************************************	
  //Update the overlap which is the maximum in the history. 
  if (overlap >= overlap_max) overlap_max = overlap;

  /*N_step 1*, 2* and 3* stand for loading, unloading and reloasing, respectively. 
    N_step *4, *5 and *6 stand for asperity contact, Hertzian contact and zero force, 
    respectively. Possible conbinations are 14,15,25,26,35 and 36. [MO - 18 Jun 2014]*/
	
  /*overlap_hertz is an equivalent overlap which is estimated using N_max.
    overlap_hertz will be used to estimate an radius of contact circle (a) later on.*/
  N_max = N_GT * pow(overlap_GT, -b) * pow(overlap_max, b);   
  overlap_hertz_max = pow(N_max/(kn*sqrt(R_star)),2.0/3.0);              
  overlap_offset = overlap_max - overlap_hertz_max;   
  //**************************************************************************
  if (overlap >= overlap_GT) {
    if (overlap >= overlap_max - tolerance) N_step = 15;
    else if (overlap < overlap_max) {
      if (overlap >= overlap_old) N_step = 35;
      else if (overlap < overlap_old) N_step = 25;
    }
    overlap_hertz = overlap - overlap_p;
    polyhertz = sqrt(overlap_hertz*R_star); //[MO - 18 July 2014] R_star is now used
    effectivekt = polyhertz*kt;
  }
  else if (overlap < overlap_GT) {
    if (overlap >= overlap_max - tolerance) { 
      N_step = 14;
      overlap_hertz = overlap - overlap_offset;
      polyhertz = sqrt(overlap_hertz*R_star);
      //ccel = N_GT * pow(overlap_GT, -b) * pow(overlap, b)*rinv;
      effectivekt = 2.0*b*(1.0-Poiseq)/(2.0-Poiseq)*ccel*r*(1.0/overlap); 
    }
    else if (overlap < overlap_max - tolerance) { 
      if (overlap < overlap_offset){
	if (overlap >= overlap_old) N_step = 36;
	else if (overlap < overlap_old) N_step = 26;
	// Particles are separated due to squahed asperities.
	overlap_hertz = tolerance;
	polyhertz = sqrt(overlap_hertz*R_star);    
	effectivekt = polyhertz*kt;
      }
      else if (overlap >= overlap_offset){
	if (overlap >= overlap_old) N_step = 35;
	else if (overlap < overlap_old) N_step = 25;
	overlap_hertz = overlap - overlap_offset;
	polyhertz = sqrt(overlap_hertz*R_star);
	effectivekt = polyhertz*kt;
      }
    }
  }
  // Calculate ccel = N/r ******************
  ccel = kn*overlap_hertz*rinv*polyhertz;
  //****************************************
                	
  //********************************************************************
  // Calculation of normal strain energy
  //********************************************************************	
  double nstr, sstr;	  
  //~ Add contributions to traced energy [KH - 20 February 2014]
  if (pairenergy) {  	    
    /*~ Normal contribution to strain energy doesn't need to be calculated incrementally*/
    /*~~ However, the hysterisys should be considered using CM model [MO - 18 Jun 2014] ~~*/
    if(N_step == 14){        // 14 stands for loading on asperity contact   [MO - 18 Jun 2014]
      nstr = N_GT*b1inv*pow(overlap_GT,-b)*pow(overlap,b+1.0);
    }
    if(N_step == 15){       
      nstr = N_GT*overlap_GT*b1inv
	-0.4*kn*sqrt(R_star)*(pow(overlap_GT-overlap_p,2.5)-pow(overlap-overlap_p,2.5));     
    }	    
    if(N_step == 26 || N_step == 36){  
      if(overlap_max < overlap_GT){
	nstr = N_GT*b1inv*pow(overlap_GT,-b)*pow(overlap_max,b+1.0)
	  -0.4*kn*sqrt(R_star)*pow(overlap_max-overlap_offset,2.5);
      }
      if(overlap_max >= overlap_GT){
	nstr = N_GT*overlap_GT*b1inv-0.4*kn*sqrt(R_star)*pow(overlap_GT-overlap_p,2.5);
      }
    }
    if(N_step == 25 || N_step == 35){ 
      if(overlap_max < overlap_GT){
	nstr = N_GT*b1inv*pow(overlap_GT,-b)*pow(overlap_max,b+1.0)
	  -0.4*kn*sqrt(R_star)*(pow(overlap_max-overlap_offset,2.5)-pow(overlap-overlap_offset,2.5));      
      }	    
      if(overlap_max >= overlap_GT){
	nstr = N_GT*overlap_GT*b1inv
	  -0.4*kn*sqrt(R_star)*(pow(overlap_GT-overlap_p,2.5)-pow(overlap-overlap_p,2.5));
      }
    }
    normalstrain += nstr;
    if (trace_energy) shear[27] = nstr;  //26, 27, 28, 29 for energy tracing
  }
  //**************************************************************************
  /*Hertzian model defines a = sqrt(R_star*overlap), but this is not applicable to CM model.
    Instead of overlap = radsum - r, "overlap_hertz" should be used to calculate a.
    For convenience, the same name of a is used in this code. Also N = kn*overlap*a does not 
    work in this model, so both a_old and N_old should be stored.*/
  double a = sqrt(R_star*overlap_hertz); // = polyhertz
  double N = ccel*r;
  double a_old = fabs(shear[17]);
  double N_old = fabs(shear[18]);
  double N_step_old = static_cast<int>(fabs(shear[19]));
  double overlap_hertz_old = fabs(shear[20]);
  double dN = N - N_old;

  //**************************************************************************
  // Tangential contact model based on Mindlin&Deresiewicz(1953) is from here.	
  //**************************************************************************
  double T = 0.0;                               // bulk tangential force
  double T_old = shear[12];                     // bulk tangential force at the previous step (including its sign)
  double T_star1 = shear[8];                    // T*:  tangential force at the revasal step from loading to unloading 
  double T_star2 = shear[9];                    // T**: tangential force at the revasal step from unloading to re-loading 
  double T_star3 = shear[10];                   // 0.0 if not active 
  double T_star4 = shear[11];                   // 0.0 if not active
  double Tdisp  = 0.0;                          // bulk tangential displacement	
  double Tdisp_mag =0.0;                        // magnitude of bulk tangential displacement
  double Tdisp_old  = shear[4];                 // bulk tangential displacement at the previous step
  double dTdisp = 0.0;                          // increment of bulk displacement
  double Tdisp1 = 0.0;                          // tangential displacement of x direction
  double Tdisp2 = 0.0;                          // tangential displacement of y direction
  double Tdisp3 = 0.0;                          // tangential displacement of z direction
  double Tdisp1_old = shear[5];                 // tangential displacement of x direction at the previous step
  double Tdisp2_old = shear[6];                 // tangential displacement of y direction at the previous step
  double Tdisp3_old = shear[7];                 // tangential displacement of z direction at the previous step     
  double inter_product = 0.0;                   // inter product between current and previous vectors of tangential displacement
  int T_step = 0;                               // the current loading phase, e.g. 11 = N increasing T increasing
  int T_step_old = static_cast<int>(fabs(shear[15])); // the loading phase at the previous step
  int slip_T = static_cast<int>(fabs(shear[10]));     // slip_T means fully slipped due to tangential contact force
  int UFL = 0;                                  // UFL is the direction of tangential load
  int zero = 0;                                 // zero = 1 when tangential displacement is zero to avoid a numerical error 
  int CTD = static_cast<int>(fabs(shear[14]));  // CTD is the direction of tangential displacement 
  if (CTD == 10000) CTD = -1;
  int CDF = static_cast<int>(fabs(shear[11]));  // CDF is the direction of tangential load-displacement system
  if (CDF == 10000) CDF = -1;	
  //***************************************
  // First, calculate the shear displacement
  if (xmu < tolerance) { // Tdisp is not accumulated if xmu = 0
    if (shearupdate){                              // Update the tangential displacement  
      Tdisp1 = Tdisp1_old + vtr1*dt;               // shear displacement =vtr*dt
      Tdisp2 = Tdisp2_old + vtr2*dt;
      Tdisp3 = Tdisp3_old + vtr3*dt;
      Tdisp_mag = sqrt(Tdisp1*Tdisp1 + Tdisp2*Tdisp2 + Tdisp3*Tdisp3);
      
      if (Tdisp_mag <= tolerance){                   // Tdisp = 0.0 cause numerial error later on, so avoid that here
	Tdisp1 = 1.0e-25;
	Tdisp2 = 0.0;
	Tdisp3 = 0.0;
	Tdisp_mag = 1.0e-25;
	zero = 1;                                  // zero = 1 is a special case 
      }
    }else{
      Tdisp1 = Tdisp1_old;
      Tdisp2 = Tdisp2_old;
      Tdisp3 = Tdisp3_old;
      Tdisp_mag = sqrt(Tdisp1*Tdisp1 + Tdisp2*Tdisp2 + Tdisp3*Tdisp3);
    }    
    // inter_product becomes negative if the direction of tangential loading changes
    inter_product = Tdisp1*Tdisp1_old + Tdisp2*Tdisp2_old + Tdisp3*Tdisp3_old;
	
    if (shearupdate){ 
      if (Tdisp_mag <= tolerance && fabs(Tdisp_old) <= tolerance){
	CTD = 1;                                     // when there is no shear disp at all.
	CDF = 1;
      }
      else if (Tdisp_mag > tolerance && fabs(Tdisp_old) <= tolerance){    
	CTD = 1;                                   // this is for the first increment of tangential displacement
	CDF = 1;
      }
      else if (inter_product >= 0.0)                   CTD *= 1;
      else if (inter_product < 0.0)                    CTD *= -1;
      else fprintf(screen,"Unexpected case occurred in zone A. ERROR!!");
      Tdisp =  CTD * Tdisp_mag;                      // sign for tangential displacement based on the value of the inter product 
      dTdisp = Tdisp - Tdisp_old;                    // dTdisp includes direction of tangential displacement (not magnitude)
    }
  }
  else {
    dTdisp = Tdisp = Tdisp1 = Tdisp2 = Tdisp3 = Tdisp_mag = 0.0; // if xmu >=  0
    zero = 1;
  }
  //***********************************************************************************
  /* Fig.7 of Mindlin & Deresiewicz (1953) explain why this special case is needed.
     Tdisp_DSC is the minimum tangential displacement to move onto a new loading curve of N+dN.
     If the increment of the tangential displacement is less than Tdisp_DSC, the resultant tangential 
     force is less than theoretical value.
     The maximum tangential contact stiffness (8.0*G_star*a) is used until the current tangential force
     catches up the the new loading curve. */
  //***********************************************************************************
  double Tdisp_DSC = xmu*dN/(8.0*G_star*a);      // Tdisp_DSC is either positive or negative depending upon the sign of dN
  double Tdisp_DD = fabs(shear[13]) - fabs(dTdisp) + Tdisp_DSC;
  int    special_DD = 0; 
  if (Tdisp_DD < 0.0) Tdisp_DD = 0.0;            // Tdisp_DD <= 0.0 should be satisfied to move onto a new loading curve for N+dN.
  else                special_DD = 1;            // Special case for DD value is now active               
  //*************************************************************************************
  double theta1 = 1.0;
  double theta2 = 1.0;
  double theta3 = 1.0;                  
  double theta_t  = 1.0;                           // reduction ratio of tangential cotnact stiffness due to tangential load 
  
  if (xmu < tolerance) theta_t = 1.0;
  else {
    //*********************************************************************************************
    // Idnetify the loading steps for tangential component
    // for the first step of the special case	  
    if (special_DD == 1){
      if (dN > 0.0 && T_step_old != 115 && T_step_old != 125 && T_step_old != 135 && T_step_old != 145){
	if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance)          T_step = 115;
	else if (CDF*dTdisp <  0.0 && fabs(T_star2) < tolerance)                                  T_step = 125; 
	else if (CDF*dTdisp >= 0.0 && fabs(T_old) < fabs(T_star1))                                T_step = 135;
	else if (CDF*dTdisp <  0.0 && fabs(T_old) <= fabs(T_star1) && fabs(T_star2)>= tolerance)  T_step = 145;
	else fprintf(screen,"Unexpected case occurs in zone B. ERROR!!");
	theta_t = 1.0;
	// if the special case was invoked in the previous step and is still active
      }else if ((T_step_old == 115 || T_step_old == 135) && CDF*dTdisp >= 0){  
	// still in step_115 or 135; if moved to unloading of T, go to the usual case
	T_step = T_step_old;
	theta_t = 1.0;
      }else if ((T_step_old == 125 || T_step_old == 145) && CDF*dTdisp < 0){   
	// still in step_125 or 145; if moved to re-loading of T, go to the usual case
	T_step = T_step_old;                                           
	theta_t = 1.0;
      }else if (T_step_old == 115 && CDF*dTdisp < 0){
	// the first step from loading of special case to unloading of special case.
	T_step = 125;
	theta_t = 1.0;
      }else if (T_step_old == 125 && CDF*dTdisp >= 0){
	// the first step from unloading of special case to reloading of special case.
	T_step = 135;
	theta_t = 1.0;
      }else if (T_step_old == 135 && CDF*dTdisp < 0){
	// the first step from reloading  of special case to re-unloading of special case.
	T_step = 145;
	theta_t = 1.0;
      }else if (T_step_old == 145 && CDF*dTdisp > 0){
	// the first step from re-unloading  of special case to re-reloading of special case.
	T_step = 135;
	theta_t = 1.0;
      }
      //*************************************************************************************
    }else {  // Usual cases 
      // T loading: T_step = *1 
      if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance){ 
	theta1 = 1.0-(CDF*T_old+xmu*dN)/(xmu*N);	
	if (theta1 <= 0.0) theta1 = theta_t = 0.0;
	else theta_t = pow(theta1, 1.0/3.0);
	if       (dN >  tolerance)      T_step = 11;   // N increasing
	else if  (dN < -tolerance)      T_step = 21;   // N decreasing
	else                          T_step =  1;   // N constant
      }
      // T unloading: T_step = *2 
      else if (CDF*dTdisp < 0.0 && fabs(T_star2) < tolerance){
	// For the first step from loading to unloading, T* is still null, which should be T* = T_old.
	if  (fabs(T_star1) > tolerance) theta2 = 1.0-(CDF*(T_star1-T_old)+2.0*xmu*dN)/(2.0*xmu*N);
	else                           theta2 = 1.0-(2.0*xmu*dN)/(2.0*xmu*N);
	if (theta2 <= 0.0) theta2 = theta_t = 0.0;	  
	else               theta_t  = pow(theta2, 1.0/3.0);
	if      (dN >  tolerance) T_step = 12; 
	else if (dN < -tolerance) T_step = 22;      
	else                    T_step =  2;
      }
      // T re-loading: T_step = *3
      else if (CDF*dTdisp >= 0.0 && fabs(T_old) < fabs(T_star1)){
	// For the first step from unloading to re-loading, T** is still null, which should be T** = T_old.
	if  (fabs(T_star2) > tolerance) theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu*dN)/(2.0*xmu*N);
	else                           theta3 = 1.0-(2.0*xmu*dN)/(2.0*xmu*N);
	if (theta3 <= 0.0) theta3 = theta_t = 0.0;
	else               theta_t  = pow(theta3, 1.0/3.0);
	if      (dN >  tolerance) T_step = 13; 
	else if (dN < -tolerance) T_step = 23;     
	else                    T_step = 3;
      }
      // T re-unloading: T_step = *4
      else if (CDF*dTdisp < 0.0 && fabs(T_old) < fabs(T_star1) && fabs(T_star2) >=  tolerance){
	// This part is same with the above (T_step = *3) due to simplification.
	theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu*dN)/(2.0*xmu*N);
	if (theta3 <= 0.0) theta3 = theta_t = 0.0;
	else               theta_t  = pow(theta3, 1.0/3.0);
	if      (dN >  tolerance) T_step = 14;
	else if (dN < -tolerance) T_step = 24; 
	else                    T_step =  4;      
      }
      else if (slip_T == 1){
	T_step = T_step_old;
	theta_t = tolerance;
      }
      else fprintf(screen,"Unexpected case occurred in zone C. ERROR!!");
    }
    //********************************************************************************
    //UFL is neccesary for the calculation of tangential contact stiffness
    if (T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) UFL = -1;
    else if (T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) UFL = -1;
    else                                                              UFL =  1;
  }       // end of if function for xmu < tolerance
  //**********************************************************************************
  // Calculate the tangential contact stiffness and the resultant tangential contact force
  double K_T = 0.0;                             // tangential contact stiffness
  double dT = 0.0;                              // increment of tangential contact force
  double fs_ratio1;                             // ratio to re-scale the tangential force to avoid inconsistency.
  double fs_ratio2;                             // ratio to re-scale the tangential force if full sliding take places
  double K_R;                                   // ratio of tangential contact stiffness over one for shm model	
  double T_temp;                                // tangential force which exceeded fslim
  //**********************************************************************************
  // update T_old to ensure the magnitude of shear force is consistent from the previous step.
  if (T_old >= 0) T_old =   sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  else            T_old = - sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
  //**********************************************************************************

  if (theta_t > 1.0) theta_t = 1.0;
  if (zero == 0) K_T = 8.0*G_star*theta_t*a + CDF*UFL*xmu*(1.0-theta_t)*dN/dTdisp;
  else           K_T = 0.0; // zero == 1 means dTdisp is zero and K_T becomes inf
  K_R = K_T / (8.0*G_star*a);

  if (shearupdate){
    if (zero == 0){
      dT = K_T * dTdisp; 
      T = dT + T_old;
      // update the tangential contact force by distributing dT into 3 components based on imcrese of relative rotational velosity
      shear[0] -= K_T * vtr1 * dt; 
      shear[1] -= K_T * vtr2 * dt;
      shear[2] -= K_T * vtr3 * dt;
	     
      //rescale the component
      fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
      fs_ratio1 = fabs(T)/fs;
      shear[0] *= fs_ratio1; 
      shear[1] *= fs_ratio1;
      shear[2] *= fs_ratio1;
    }
  } else T = T_old;
  //****************************************************************************************
  // rescale tangential force if full sliding takes place 
  fslim = xmu * N;                               // ccel*r = N
  fs = fabs(T);                                  // re-use fs for re-scale the tangential force         
  fs_ratio2 = fslim/fs;
  T_temp = T;
  if (fs > fslim) {
    if (fs > tolerance) {
      if (shearupdate) {
	T *= fs_ratio2;
	shear[0] *= fs_ratio2; 
	shear[1] *= fs_ratio2;
	shear[2] *= fs_ratio2;   
	slip_T = 1;
      }
    } else shear[0] = shear[1] = shear[2] = 0.0;
  } else slip_T = 0;
	
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
  
  //~~ Call function for twisting resistance model [MO - 04 November 2014]   
  double dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy;
  if (*D_spin && shearupdate) 
    Deresiewicz1954_spin(i,numshearquants,dx,dy,dz,radius,r,torque,shear,dspin_i,
			 dspin_stm,spin_stm,dM_i,dM,K_spin,theta_r,
			 M_limit,Geq,Poiseq,Dspin_energy,a,N);
  
  //~ Add contributions to traced energy [KH - 20 February 2014]
  if (pairenergy) {
    /* Full sliding can be considered as accumulation of partial slip for HMD model.
       Theoretically speaking, full sliding does not take place for HMD modle.
       Thus, shear strain energy here is summation of shear strain energy + friction for shm model.*/
    sstr = 0.5*dTdisp*(T + T_temp);
    shearstrain += sstr;
    if (trace_energy) shear[28] += sstr; // 26 is empty. 

    //~~ Update the spin contribution [MO - 13 November 2014]
    if (*D_spin) {
      spinenergy += Dspin_energy;
      if (trace_energy) shear[29] += Dspin_energy;
    }	    
  }
	
  //*************************************************************************************
  // Update T_star1 and T_star2
  //*************************************************************************************
  // Update the T_star1 and T_star2 considering the change of the normal contact load
  if (T_step != 1 && T_step != 11 && T_step != 21 && T_step != 115 && fabs(T_star1)> tolerance) T_star1 += CDF*xmu*dN;
  if ((T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) && fabs(T_star2) > tolerance) T_star2 -= CDF*xmu*dN;
  if ((T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) && fabs(T_star2) > tolerance) T_star2 -= CDF*xmu*dN;
  //*************************************************************************************************
  int sp_Tstar1 = 0;                         // CDF *= -1 should not be active if sp_Tstar1 = 1 is set.  
  if (T_step == 1 || T_step == 11 || T_step == 21 || T_step == 115){   // do not forget to include T_step == 115
    T_star1 = 0.0;
    T_star2 = 0.0;
  }
  // Update the T_star1 and T_star2 for new loading curve; move to loading step
  if ((T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) && fabs(T) > fabs(T_star1) && fabs(T_star1) > tolerance){
    T_star1 = 0.0;
    T_star2 = 0.0;
  }
  // Update the T_star2 for new unloading curve; move to unloading step
  if ((T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) && fabs(T) > fabs(T_star2) && fabs(T_star2) > tolerance){
    T_star2 = 0.0;
  }
  if ((T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) && fabs(T_star1) <= tolerance && fabs(T_star2) <= tolerance){
    // the first step from T_step = 1; loading to unloading
    T_star1 = T_old;
    Tdisp_DD = 0.0;
    sp_Tstar1 = 1;
  } 
  if ((T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) && fabs(T_star1) > fabs(T) && fabs(T_star2) <= tolerance){  
    // the first step from T_step = 2; unloading to re-loading
    T_star2 = T_old;
    Tdisp_DD = 0.0;
  } 
  //*************************************************************************************************
  //special case when |T| for unloading reaches -|T_star1|  // move on to a new loading curve in the opposite direction
  if ((UFL == -1) && fabs(T) > fabs(T_star1) && fabs(T_star1) > tolerance){
    T_star1 = 0.0;
    T_star2 = 0.0;
    // avoid the case when T* > T > T*-xmu*dN 
    if (sp_Tstar1 == 0)  CDF *= -1;  // the system is moved onto virgin loading curve
  }
  //*************************************************************************
  // Added this to avoid numerical errors [MO - 04 January 2015]
  if (fabs(T_star1 - T_star2) < tolerance) {
    T_star1 = 0.0;
    T_star2 = 0.0;
  }
  //*************************************************************************	    
  if (zero == 1)  Tdisp = Tdisp1 = Tdisp2 = Tdisp3 = 0.0;
  // Store the CTD and CDF as absolute values for the next step to avoid the sign changing [MO - 15 December 2014] 
  if (CTD == -1) CTD = 10000;
  if (CDF == -1) CDF = 10000;

  // shear[0], shear[1] and shear[2] are tangential force of x, y and z directions, espectively.	
  shear[3] = overlap;               // overlap
  shear[4] = Tdisp;                 // tangential displacement
  shear[5] = Tdisp1;                // tangential displacement of x direction 
  shear[6] = Tdisp2;                // tangential displacement of y direction
  shear[7] = Tdisp3;                // tangential displacement of z direction 
  shear[8] = T_star1;               // first reverse point of tangential load (T) from loading to unloading
  shear[9] = T_star2;               // second reverse point of tangential load from unloading to re-loading
  shear[10] = slip_T;               // if full slip is mobilized, slip_T = 1 (int);
  shear[11] = CDF;                  // CDF indicates wheather system is loading or unloading //T_star4;       
  shear[12] = T;       
  shear[13] = Tdisp_DD;             // Tdisp_DD <= 0.0 should be satisfied to move on a new loading curve for N+dN 
  shear[14] = CTD;                  // +1 and -1 mean positive and negative shear disp, respectively.       
  shear[15] = T_step;
  shear[16] = overlap_max;          // shear[3] in CM model
  shear[17] = a;
  shear[18] = N;
  shear[19] = N_step;
  shear[20] = overlap_hertz;
  shear[21] = ccel;

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
  double bytes = nmax * sizeof(int);
  //~ Now have additional shear parameters [KH - 30 October 2013]
  bytes += numshearquants*nmax * sizeof(double);

  //add vatom memory!
  return bytes;
}

/* ----------------------------------------------------------------------
  allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixWallGran::grow_arrays(int nmax)
{
  memory->grow(shear,nmax,numshearquants,"fix_wall_gran:shear"); //~ [KH - 30 October 2013]
}

/* ----------------------------------------------------------------------
  copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixWallGran::copy_arrays(int i, int j, int delflag)
{
  for (int q = 0; q < numshearquants; q++)
    shear[j][q] = shear[i][q]; //~ [KH - 30 October 2013]
}

/* ----------------------------------------------------------------------
  initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixWallGran::set_arrays(int i)
{
  for (int q = 0; q < numshearquants; q++)
    shear[i][q] = 0.0; //~ [KH - 30 October 2013]
}

/* ----------------------------------------------------------------------
  pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixWallGran::pack_exchange(int i, double *buf)
{
  for (int q = 0; q < numshearquants; q++)
    buf[q] = shear[i][q]; //~ [KH - 30 October 2013]

  return numshearquants;
}

/* ----------------------------------------------------------------------
  unpack values into local atom-based arrays after exchange
------------------------------------------------------------------------- */

int FixWallGran::unpack_exchange(int nlocal, double *buf)
{
  for (int q = 0; q < numshearquants; q++)
    shear[nlocal][q] = buf[q]; //~ [KH - 30 October 2013]
  
  return numshearquants;
}

/* ----------------------------------------------------------------------
  pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixWallGran::pack_restart(int i, double *buf)
{
  int m = 0;
  buf[m++] = numshearquants + 1; //~ [KH - 30 October 2013]
  for (int q = 0; q < numshearquants; q++)
    buf[m++] = shear[i][q];

  return m;
}

/* ----------------------------------------------------------------------
  unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixWallGran::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  for (int q = 0; q < numshearquants; q++) //~ [KH - 30 October 2013]
    shear[nlocal][q] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
  maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixWallGran::maxsize_restart()
{
  int y = numshearquants + 1;
  return y;
}

/* ----------------------------------------------------------------------
  size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixWallGran::size_restart(int nlocal)
{
  int y = numshearquants + 1;
  return y;
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
      velwall[0]=velwall[1]=velwall[2]=0.0;
      argsread+=2;
      fprintf(screen, "stopped wall translation\n");
    } else {
      wtranslate = 1;
      velwall[0] = force->numeric(FLERR,arg[argsread+1]);
      velwall[1] = force->numeric(FLERR,arg[argsread+2]);
      velwall[2] = force->numeric(FLERR,arg[argsread+3]);
      argsread+= 4;
      fprintf(screen, "changed wall velocity to [ %e %e %e ]\n",velwall[0],velwall[1],velwall[2]);
    }
  }
  else if (strcmp(arg[argsread],"stresscontrol") == 0) {
    if (wtranslate == 1 && wscontrol == 0 && strcmp(arg[argsread+1],"off") == 0) {
      error->all(FLERR,"Illegal fix modify wall/gran command");
    }
    wscontrol = 1;
    wtranslate = 1;
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
    if (wtranslate && (lo != -BIG && hi != BIG))
      error->all(FLERR,"Cannot specify both top and bottom walls and translate for fix wall/gran - check your fix_modify"); // this check will fail if the walls have moved..
    if (wtranslate && wallstyle == ZCYLINDER)
      error->all(FLERR,"Cannot use translate with cylinder fix wall/gran - check your fix_modify");
    if (wtranslate && (wiggle || wshear))
      error->all(FLERR,"Cannot translate and wiggle or shear fix wall/gran- check your fix_modify");
  }
  return argsread;
}

/* ---------------------------------------------------------------------- 
  A function that implements wall movement
------------------------------------------------------------------------- */

void FixWallGran::move_wall() {
  //~ Only update lo or hi if not NULL [KH - 27 November 2013]
  if (lo != -BIG) lo+=velwall[wallstyle]*dt;
  if (hi != BIG) hi+=velwall[wallstyle]*dt;
}

/* ---------------------------------------------------------------------- 
  A function that calculates the wall velocity based on a target force and a 
  specific controller
------------------------------------------------------------------------- */

void FixWallGran::velscontrol() {

  MPI_Allreduce(fwall,fwall_all,3,MPI_DOUBLE,MPI_SUM,world);
  if (ftvarying == 1) {
    //modify->clearstep_compute();needed???
    targetf = input->variable->compute_equal(fvar);
    //modify->addstep_compute(update->ntimestep + 1);needed???
  }
  //if (update->ntimestep != time_origin)
  velwall[wallstyle] = gain * (targetf - fwall_all[wallstyle]);

}

/* ---------------------------------------------------------------------- 
  A function that calculates the outputs of the fix
  1st output:low wall position
  2nd output:high wall position
  3rd-5th outputs:forces on atoms by wall
------------------------------------------------------------------------- */

double FixWallGran::compute_vector(int n)
{

  if (n == 0) return lo;
  if (n == 1) return hi;
  // only sum across procs one time?? //if (eflag == 0) {??
  MPI_Allreduce(fwall,fwall_all,3,MPI_DOUBLE,MPI_SUM,world);
  if (update->ntimestep == time_origin) error->warning(FLERR,"Force output by fix_wall_gran not computed properly");
  if (n>4) error->all(FLERR,"Illegal fix_wall_gran output");
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
  double G_star = 1.0/((2.0-Poiseq)/Geq+(2.0-Poiseq_p)/Geq_p);
  double PI = 4.0*atan(1.0);
  double a_old = fabs(shear[numshearq-8]); 
  double N_old = fabs(shear[numshearq-9]);
  double dN = N - N_old; // change of normal contact force
  double xmu_p = static_cast<double>(xmu_p);
  double xmu_mean = 0.5*(xmu + xmu_p); // mean firction coefficient is used 
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
	  fprintf(screen,"time %i Unexpected case occurred in zone A of spin model. ERROR!!\n",update->ntimestep);
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
      else {
	fprintf(screen,"time %i Unexpected case occurred in zone B of spin model. ERROR!!\n",update->ntimestep);
      }
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
  double list[2];
  list[n++] = gatheredf;
  list[n++] = gatheredss;
  list[n++] = gatheredse;
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
}

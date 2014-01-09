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

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{XPLANE=0,YPLANE=1,ZPLANE=2,ZCYLINDER};    // XYZ PLANE need to be 0,1,2
enum{HOOKE,HOOKE_HISTORY,HERTZ_HISTORY,SHM_HISTORY}; //~ Added SHM_HISTORY option [KH - 30 October 2013]

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

  restart_peratom = 1;
  create_attribute = 1;

  vector_flag = 1;
  size_vector = 5;
  global_freq = 1;

  // wall/particle coefficients

  //~ Make special allowances for shm pairstyle [KH - 30 October 2013]
  double Geq, Poiseq;
  if (force->pair_match("gran/shm/history",1)) {
    Geq = force->numeric(FLERR,arg[3]);
    Poiseq = force->numeric(FLERR,arg[4]);
    xmu = force->numeric(FLERR,arg[5]);

    if (Geq < 0.0 || Poiseq < 0.0 || xmu < 0.0 || Poiseq > 0.5) error->all(FLERR,"Illegal shm pair parameter values in fix wall gran");

    kn = 4.0*Geq / (3.0*(1.0-Poiseq));
    kt = 4.0*Geq / (2.0-Poiseq);
  } else {
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

  // convert Kn and Kt from pressure units to force/distance^2 if Hertzian

  if (force->pair_match("gran/hertz/history",1) || force->pair_match("gran/shm/history",1)) {
    kn /= force->nktv2p;
    kt /= force->nktv2p;
  }
  
  // wallstyle args

  int iarg = 9;
  if (force->pair_match("gran/shm/history",1)) iarg = 6; //~ [KH - 30 October 2013]

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

  /*~ Finally, adding a rolling resistance model causes the number of
    shear history quantities to be increased by 13 [KH - 30 October 2013]*/
  int dim = 1;
  Pair *pair;
  if (force->pair_match("gran/hooke/history",1)) 
    pair = force->pair_match("gran/hooke/history",1);
  else if (force->pair_match("gran/hertz/history",1))
    pair = force->pair_match("gran/hertz/history",1);
  else if (force->pair_match("gran/shm/history",1))
    pair = force->pair_match("gran/shm/history",1);
  else dim = 0; //~ Adding for other pairstyles [KH - 5 November 2013]

  if (dim) rolling = (int *) pair->extract("rolling",dim);
  if (*rolling) numshearquants += 13;

  /*~ Use same method to obtain model_type and rolling_delta from 
    pairstyles. Also initialise two integers used to limit the 
    numbers of warnings about failures to calculate contact stiffnesses
    in the rolling resistance model [KH - 5 November 2013]*/
  if (*rolling) {
     model_type = (int *) pair->extract("model_type",dim);
     rolling_delta = (double *) pair->extract("rolling_delta",dim);
     lastwarning[0] = lastwarning[1] = -1000000;
  }

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

  // initialize as if particle is not touching wall

  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++)
    for (int q = 0; q < numshearquants; q++)
      shear[i][q] = 0.0; //~ [KH - 30 October 2013]

  time_origin = update->ntimestep;
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
  if (*rolling && shearupdate)
    rolling_resistance(i,numshearquants,dx,dy,dz,r,radius,ccel,fn,
		       kt,torque,shear);

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
  double shrmag,rsht,polyhertz,rinv,rsqinv;

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
  if (*rolling && shearupdate)
    rolling_resistance(i,numshearquants,dx,dy,dz,r,radius,ccel,fn,
		       effectivekt,torque,shear);

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

  if (shearupdate) {
    /*~ Apply Colin Thornton's suggested correction (see
      Eq. 18 of 2013 P. Tech. paper) [KH - 30 October 2013]*/
    if (shear[3] > polyhertz) {
      /*~ Note that as polyhertz is >= 0, there is no need to
	check for shear[3] == 0 in the expressions below*/
      shear[0] *= polyhertz/shear[3];
      shear[1] *= polyhertz/shear[3];
      shear[2] *= polyhertz/shear[3];
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

  //~ Assign current polyhertz value to shear[3] [KH - 30 October 2013]
  shear[3] = polyhertz;

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
  if (*rolling && shearupdate)
    rolling_resistance(i,numshearquants,dx,dy,dz,r,radius,ccel,fslim,
		       effectivekt,torque,shear);

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

            vatom[i][0] += dx * fx / volume;
            vatom[i][1] += dy * fy / volume;
            vatom[i][2] += dz * fz / volume;
            vatom[i][3] += dx * fy / volume;
            vatom[i][4] += dx * fz / volume;
            vatom[i][5] += dy * fz / volume;
          //}
        }
    //}

}

/* ---------------------------------------------------------------------- */

void FixWallGran::rolling_resistance(int i, int numshearq, double dx, double dy, double dz, double r, double radius, double ccel, double maxshear, double effectivekt, double *torque, double *shear)
{
  /*~ This rolling resistance model was developed by Xin Huang during
    the summer and autumn of 2013. It is the companion function of
    PairGranHookeHistory::rolling_resistance; more information about
    the rolling resistance model can be obtained by looking at this
    code [KH - 30 October 2013]*/

  //~ If *rolling_delta == 0, exit from this function prematurely
  double tolerance = 1.0e-20;
  if (*rolling_delta < tolerance) {
    for (int q = 0; q < 13; q++) shear[numshearq-13+q] = 0.0;
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
  double sinthetaovertwo, recipmagnxny;
  sinthetaovertwo = sqrt(0.5*(1.0 - nz));
  if (nx < 0.0) sinthetaovertwo *= -1.0;

  recipmagnxny = 1.0/sqrt(nx*nx + ny*ny); //~ Potential division by zero
  if (nx*nx < tolerance && ny*ny < tolerance) recipmagnxny = tolerance;

  double q[4];
  q[0] = sqrt(0.5*(1.0 + nz)); //~ = cos(theta)/2
  q[1] = -sinthetaovertwo*ny*recipmagnxny;
  q[2] = sinthetaovertwo*nx*recipmagnxny;
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
  double globaloldomegai[3], localoldomegai[3];
  for (int q = 0; q < 3; q++) globaloldomegai[q] = oldomegas[i][q];
  
  /*~ Transfer the old omega values to the local coordinate system
    using the rotation matrix T*/
  MathExtra::matvec(T,globaloldomegai,localoldomegai);

  //~ Now find relative rotations, dthetar, in three directions
  double db[3], dthetar[3];

  for (int q = 0; q < 3; q++) {
    db[q] = radius*localoldomegai[q]*dt;
    dthetar[q] = -db[q]/commonradius;
  }

  /*~ The equivalent area normal contact stiffness is found by dividing
    the magnitude of the normal contact force by the product of the normal 
    contact overlap and contact area. If division by zero, issue a warning.
    Also calculate some necessary quantities for later use*/
  int warnfrequency = 100; //~ How often to warn about stiffness calcs
  double PI = 4.0*atan(1.0);
  double un, B, delbyb, recipcarea, recipA, knbar, ksbar;

  un = radius - r; //~ Normal contact overlap
  B = sqrt(radius*radius - r*r); //~ Radius of contact plane

  if (B > tolerance) {
    recipcarea = 1.0/(PI*B*B); //~ Reciprocal of contact area
    delbyb = *rolling_delta*B;
    knbar = fabs(ccel*r*recipcarea/un);
  } else {
    //~ Issue a warning and broadcast lastwarning int to all procs
    if (update->ntimestep-lastwarning[0] >= warnfrequency) {
      fprintf(screen,"Cannot estimate either contact stiffness in rolling resistance model on timestep "BIGINT_FORMAT"\n",update->ntimestep);
      lastwarning[0] = lastwarning[1] = update->ntimestep;
      MPI_Bcast(&lastwarning[0],2,MPI_INT,comm->me,world);
    }
    delbyb = tolerance; //~ Tiny value
    knbar = BIG; //~ Huge stiffness value
  }
  recipA = 1.0/(PI*delbyb*delbyb); //~ Reciprocal of modified contact area

  /*~ The equivalent area tangential contact stiffness is equal to the
    effective tangential stiffness (kt for Hookean model; kt*polyhertz
    for Hertzian model) divided by the contact area. If kt is zero or
    if the contact area is very, very small, warn the user. Values of
    ksbar are stored in the last column of the shear array.*/
  if (kt < tolerance && update->beginstep == update->ntimestep-1 
      && comm->me == 0)
    error->warning(FLERR,"Using zero kt: tangential contact stiffness cannot be estimated in rolling resistance model");

  if (B > tolerance && effectivekt > tolerance) ksbar = effectivekt*recipcarea;
  else if (shear[numshearq-1] > tolerance) ksbar = fabs(shear[numshearq-1]);
  else {
    if (kt >= tolerance && update->ntimestep-lastwarning[1] >= warnfrequency) {
      fprintf(screen,"Cannot estimate tangential contact stiffness in rolling resistance model on timestep "BIGINT_FORMAT"\n",update->ntimestep);
      lastwarning[1] = update->ntimestep;
      MPI_Bcast(&lastwarning[1],1,MPI_INT,comm->me,world);
    }
    ksbar = tolerance; //~ Assume a tiny value
  }
  shear[numshearq-1] = ksbar; //~ Store ksbar in last column of shear array

  /*~ Calculate the maximum allowable rolling and twisting resistances
    and store these in thetalimit for convenience*/
  double thetalimit[3], st[3], localdM[3];
  thetalimit[0] = thetalimit[1] = atan(un/delbyb);

  /*~ Option B: The maximum shear stress (periphery) induced by 
    twisting torque equals to the shear stress limit (default)*/
  thetalimit[2] = maxshear*recipA/(ksbar*delbyb);

  /*~ Option C: All the shear stresses induced by twisting torque
    equal to the shear stress limit */
  if (*model_type % 2 == 0) thetalimit[2] = 2.0*maxshear*delbyb/3.0;
  
  //~ Also create an 'st' array for convenience
  double deltaBpow = MathSpecial::powint(delbyb,4);
  st[0] = st[1] = -0.25*PI*deltaBpow*knbar;
  st[2] = -0.5*PI*deltaBpow*ksbar;

  /*~ Calculate local increments of rolling resistance and ensure
    that the increments don't exceed the limits*/
  for (int q = 0; q < 2; q++) {
    if (*model_type % 7 > 0) {//~ Rolling resistance enabled
      if (dthetar[q] < thetalimit[q]) localdM[q] = st[q]*dthetar[q];
      else localdM[q] = st[q]*thetalimit[q];
    } else localdM[q] = 0.0; //~ Rolling resistance disabled
  }

  /*~ Now find local increments of twisting resistance for which
    there are two cases*/
  if (*model_type % 11 > 0) {//~ Twisting resistance enabled
    if (dthetar[2] < thetalimit[2]) localdM[2] = st[2]*dthetar[2];
    else localdM[2] = st[2]*thetalimit[2];

    if (*model_type % 2 == 0) {//~ Option C
      localdM[2] = st[2]*dthetar[2];
      if (localdM[2] > thetalimit[2]) localdM[2] = thetalimit[2];
    }
  } else localdM[2] = 0.0; //~ Twisting resistance disabled

  for (int q = 0; q < 3; q++) {
    /*~ If the accumulated local resistances exceed the permissible
      limits, set the increments to zero and scale the accumulated
      resistances to equal the appropriate limit. The accumulated 
      local rolling and twisting resistances are stored in the
      seventh-last, sixth-last and fifth-last columns of the shear 
      array*/
    if (shear[numshearq-7+q]+localdM[q] > thetalimit[q]) {
      localdM[q] = 0.0;
      //~ OK to update regardless of issingle setting
      shear[numshearq-7+q] = thetalimit[q];
    }
  }

  /*~ Compute the global moment increments by multiplying the 
    transpose of the rotation matrix, T, by the local resistance
    increments*/
  double globaldM[3];
  MathExtra::transpose_matvec(T,localdM,globaldM);

  /*~ Now add the global resistance increments to the fourth-last, 
    third-last and second-last columns of the shear array and the
    local increments to the seventh-last, sixth-last and fifth-last. 
    The accumulated values of dus are stored in the three columns 
    immediately before, and the accumulated values of dur in the
    three columns immediately before these. For consistency with
    the ball-ball contacts, duplicate db as both dur and dus.*/
  for (int q = 0; q < 3; q++) {
    shear[numshearq-4+q] += globaldM[q];
    shear[numshearq-7+q] += localdM[q];
    shear[numshearq-10+q] += db[q];
    shear[numshearq-13+q] += db[q];

    //~ Finally update the torque values
    torque[q] -= globaldM[q];
  }
}

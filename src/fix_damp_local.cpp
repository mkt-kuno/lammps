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

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix_damp_local.h"
#include "atom.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "modify.h" //~ Added three files for energy tracing [KH - 9 April 2014]
#include "compute.h"
#include "comm.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixDampLocal::FixDampLocal(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal fix damp local command");

  restart_global = 1; //~ Global information is saved to the restart file

  alpha = atof(arg[3]);
  if (alpha>1.0 || alpha<0.0) error->all(FLERR,"Please check local damping coefficient - should be between 0 and 1");

  //~ Initialise the energy dissipated by damping [KH - 9 April 2014]
  energy_dissip = 0.0;

  respa_level_support = 1;
  ilevel_respa = 0;
}

/* ---------------------------------------------------------------------- */

FixDampLocal::~FixDampLocal()
{
  //delete alpha;
}

/* ---------------------------------------------------------------------- */

int FixDampLocal::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDampLocal::init()
{
  int max_respa = 0;

  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = max_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,max_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixDampLocal::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    ((Respa *) update->integrate)->copy_f_flevel(ilevel_respa);
  }

  /*~ Ascertain whether compute energy/gran is active or not. If
    it is, the energy dissipated by damping will be calculated.
    [KH - 9 April 2014]*/
  energy_calc = 0;
  for (int q = 0; q < modify->ncompute; q++)
    if (strcmp(modify->compute[q]->style,"energy/gran") == 0) {
      energy_calc = 1;
      break;
    }
}

/* ---------------------------------------------------------------------- */

void FixDampLocal::post_force(int vflag)
{
  // apply damping force to atoms in group
  // direction is opposed to velocity vector
  // magnitude is alpha times original force

  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double oldforce[3], oldtorque[3];

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      oldforce[0] = f[i][0];
      oldforce[1] = f[i][1];
      oldforce[2] = f[i][2];

      oldtorque[0] = torque[i][0];
      oldtorque[1] = torque[i][1];
      oldtorque[2] = torque[i][2];

      f[i][0] -= alpha*fabs(f[i][0])*signofnum(v[i][0]);
      f[i][1] -= alpha*fabs(f[i][1])*signofnum(v[i][1]);
      f[i][2] -= alpha*fabs(f[i][2])*signofnum(v[i][2]);

      torque[i][0] -= alpha*fabs(torque[i][0])*signofnum(omega[i][0]);
      torque[i][1] -= alpha*fabs(torque[i][1])*signofnum(omega[i][1]);
      torque[i][2] -= alpha*fabs(torque[i][2])*signofnum(omega[i][2]);

      //~ Calculate the energy dissipated by damping
      if (energy_calc) energy_dissip += dissipated_energy(alpha,i,oldforce,oldtorque);
    }
}

/* ---------------------------------------------------------------------- */

void FixDampLocal::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixDampLocal::signofnum(double numb)
{
  if (numb > 0.0) return 1.0;
  else if  (numb < 0.0) return -1.0;
  else return 0.0;
}

/* ---------------------------------------------------------------------- */

int FixDampLocal::modify_param(int narg, char **arg)
{
   fprintf(screen, "Local damping modified to %s \n",arg[0]);
   alpha=atof(arg[0]);
   int argsread=1;
   return argsread;

}

/* ---------------------------------------------------------------------- */

void *FixDampLocal::extract(const char *str, int &dim)
{
  /*~ This function was added so that the dissipated energy
    may be extracted by ComputeEnergyGran [KH - 9 April 2014]*/
  dim = 0;
  if (strcmp(str,"energy_dissip") == 0) return (void *) &energy_dissip;
  return NULL;
}

/* ---------------------------------------------------------------------- */

double FixDampLocal::dissipated_energy(double alpha, int i, double *of, double *ot)
{
  /*~ Dissipated energy has two components. The first is due to changing 
    the force components and is equal to the damping force (the magnitude
    of which is alpha * oldforce) multiplied by the incremental 
    translational particle displacement, which is the velocity multiplied
    by timestep.

    The second component of dissipated energy is due to the change of
    torque. The incremental work done is change in torque multiplied by 
    the incremental angular particle displacement: omega multiplied by 
    timestep.

    Note that if viscous damping is present alongside local damping,
    both affect the force so of and ot must be read in to this
    function [KH - 9 April 2014]*/

  double incenergy = 0.0;
  double alphadt = alpha*update->dt;
  double **v = atom->v;
  double **omega = atom->omega;

  for (int j = 0; j < 3; j++) {
    incenergy += fabs(of[j]*v[i][j]*signofnum(v[i][j]))*alphadt;
    incenergy += fabs(ot[j]*omega[i][j]*signofnum(omega[i][j]))*alphadt;
  }

  return incenergy;
}

/* ---------------------------------------------------------------------- */

void FixDampLocal::write_restart(FILE *fp)
{
  /*~ Note that the total energy dissipated by damping from all procs is 
    calculated and stored on proc 0 [KH - 9 April 2014]*/
  double gatherede = 0.0;
  MPI_Allreduce(&energy_dissip,&gatherede,1,MPI_DOUBLE,MPI_SUM,world);

  int n = 0;
  double list[1];
  list[n++] = gatherede;
  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ---------------------------------------------------------------------- */

void FixDampLocal::restart(char *buf)
{
  /*~ The total energy dissipated by damping is read to the root
    proc ONLY [KH - 9 April 2014]*/
  if (comm->me == 0) {
    int n = 0;
    double *list = (double *) buf;
    energy_dissip = static_cast<double> (list[n++]);
  }
}

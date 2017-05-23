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
#include "fix_viscous.h"
#include "atom.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "force.h"
#include "modify.h" //~ Added three files for energy tracing [KH - 9 April 2014]
#include "compute.h"
#include "comm.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixViscous::FixViscous(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  gamma(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal fix viscous command");

  restart_global = 1; //~ Global information is saved to the restart file

  double gamma_one = force->numeric(FLERR,arg[3]);
  gamma = new double[atom->ntypes+1];
  for (int i = 1; i <= atom->ntypes; i++) gamma[i] = gamma_one;

  // optional args

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"scale") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix viscous command");
      int itype = force->inumeric(FLERR,arg[iarg+1]);
      double scale = force->numeric(FLERR,arg[iarg+2]);
      if (itype <= 0 || itype > atom->ntypes)
        error->all(FLERR,"Illegal fix viscous command");
      gamma[itype] = gamma_one * scale;
      iarg += 3;
    } else error->all(FLERR,"Illegal fix viscous command");
  }

  //~ Initialise the energy dissipated by damping [KH - 9 April 2014]
  energy_dissip = 0.0;
  
  respa_level_support = 1;
  ilevel_respa = 0;
}

/* ---------------------------------------------------------------------- */

FixViscous::~FixViscous()
{
  delete [] gamma;
}

/* ---------------------------------------------------------------------- */

int FixViscous::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixViscous::init()
{
  int max_respa = 0;

  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = max_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,max_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixViscous::setup(int vflag)
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

void FixViscous::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixViscous::post_force(int vflag)
{
  // apply drag force to atoms in group
  // direction is opposed to velocity vector
  // magnitude depends on atom type

  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double drag;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      drag = gamma[type[i]];
      f[i][0] -= drag*v[i][0];
      f[i][1] -= drag*v[i][1];
      f[i][2] -= drag*v[i][2];
    }

  //~ Calculate the energy dissipated by damping
  if (energy_calc) energy_dissip += dissipated_energy(drag);
}

/* ---------------------------------------------------------------------- */

void FixViscous::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixViscous::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void *FixViscous::extract(const char *str, int &dim)
{
  /*~ This function was added so that the dissipated energy
    may be extracted by ComputeEnergyGran [KH - 9 April 2014]*/
  dim = 0;
  if (strcmp(str,"energy_dissip") == 0) return (void *) &energy_dissip;
  return NULL;
}

/* ---------------------------------------------------------------------- */

double FixViscous::dissipated_energy(double drag)
{
  /*~ Dissipated energy is equal to the damping force (given by drag *
    velocity) multiplied by the incremental translational particle
    displacement, which is the velocity multiplied by timestep
    [KH - 9 April 2014]*/

  double incenergy = 0.0;
  double dragdt = drag*update->dt;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      incenergy += (v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2])*dragdt;

  return incenergy;
}

/* ---------------------------------------------------------------------- */

void FixViscous::write_restart(FILE *fp)
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

void FixViscous::restart(char *buf)
{
  /*~ The total energy dissipated by damping is read to the root
    proc ONLY [KH - 9 April 2014]*/
  if (comm->me == 0) {
    int n = 0;
    double *list = (double *) buf;
    energy_dissip = static_cast<double> (list[n++]);
  }
}

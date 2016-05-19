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
   Contributing author: Kevin Hanley (Imperial)
------------------------------------------------------------------------- */

#include <stdlib.h>
#include <string.h>
#include "fix_old_omega.h"
#include "atom.h"
#include "update.h"
#include "memory.h"
#include "force.h"
#include "integrate.h"
#include "fix.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
This is intended for use only with the granular rolling resistance model.
The purpose of this code is to store the angular velocities on the previous
timestep in a per-atom array so that they may be compared with the
current omega values in the rolling resistance model.

Values are stored using a post_force function; the omega values in the
post_force function are unchanged from the force computation as even if
damping is present, this affects only the torque, not omega (until the
final_integrate function is called) [KH - 24 October 2013]
 ---------------------------------------------------------------------- */

FixOldOmega::FixOldOmega(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  restart_peratom = 1; //~ Per-atom information is saved to the restart file
  peratom_flag = 1;
  size_peratom_cols = 6; //~ omega x/y/z and positions of centroids
  peratom_freq = 1;
  create_attribute = 1;

  // perform initial allocation of atom-based array
  // register with Atom class
  oldomegas = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  atom->add_callback(1);

  /*~ Initialise the values stored in oldomegas[*][*] to ridiculous
    values. The reason is that these crazy values can be easily 
    distinguished from physically reasonable values. In the setup
    function below, these crazy values are sought. If absent (due to
    the importation of data from a restart file), no need to 
    initialise. If present, then initialise at physically-reasonable
    values (using the initialised omega and x values which are now
    available)*/

  for (int i = 0; i < atom->nlocal; i++) {
    oldomegas[i][0] = oldomegas[i][1] = oldomegas[i][2] = 1.0e20;
    oldomegas[i][3] = oldomegas[i][4] = oldomegas[i][5] = 1.0e20;
  }
}

/* ---------------------------------------------------------------------- */

FixOldOmega::~FixOldOmega()
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id,0);
  atom->delete_callback(id,1);

  // delete locally stored array
  memory->destroy(oldomegas);
}

/* ---------------------------------------------------------------------- */

int FixOldOmega::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixOldOmega::setup(int vflag)
{
  /*~ Now initialise everything for the first timestep if the data are
    not read in from a restart file. Initialise the oldomega values at
    the current omega values, and ditto for the x positions. This needs
    to be done in setup rather than init so that the ghosts are correct
    for the first timestep.*/
  int nall = atom->nlocal + atom->nghost;
  double **x = atom->x;
  double **omega = atom->omega;

  if (oldomegas[0][0] > 9.9e19 && oldomegas[0][1] > 9.9e19 && 
      oldomegas[0][2] > 9.9e19 && oldomegas[0][3] > 9.9e19 &&
      oldomegas[0][4] > 9.9e19 && oldomegas[0][5] > 9.9e19) {
    /*~ Since local atoms have smaller IDs than ghosts, checking only
      the first point is sufficient*/

    for (int i = 0; i < nall; i++) {
      oldomegas[i][0] = omega[i][0];
      oldomegas[i][1] = omega[i][1];
      oldomegas[i][2] = omega[i][2];
      oldomegas[i][3] = x[i][0]; //~ x position of centroid
      oldomegas[i][4] = x[i][1]; //~ y position
      oldomegas[i][5] = x[i][2]; //~ z position
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixOldOmega::pre_force(int vflag)
{
  if (update->integrate->vflag <= 2) update->integrate->vflag += 4;
}

/* ---------------------------------------------------------------------- */

void FixOldOmega::post_force(int vflag)
{
  int nall = atom->nlocal + atom->nghost; //~ Include ghosts
  double **x = atom->x;
  double **omega = atom->omega;

  for (int i = 0; i < nall; i++) {
    oldomegas[i][0] = omega[i][0];
    oldomegas[i][1] = omega[i][1];
    oldomegas[i][2] = omega[i][2];
    oldomegas[i][3] = x[i][0]; //~ x position of centroid
    oldomegas[i][4] = x[i][1];
    oldomegas[i][5] = x[i][2];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixOldOmega::memory_usage()
{
  double bytes = atom->nmax*6 * sizeof(double); //~ For oldomegas array
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixOldOmega::grow_arrays(int nmax)
{
  memory->grow(oldomegas,nmax,6,"old_omega:oldomegas");
  array_atom = oldomegas;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixOldOmega::copy_arrays(int i, int j, int delflag)
{
  for (int q = 0; q < 6; q++)
    oldomegas[j][q] = oldomegas[i][q];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixOldOmega::set_arrays(int i)
{
  double **x = atom->x;
  double **omega = atom->omega;
  oldomegas[i][0] = omega[i][0];
  oldomegas[i][1] = omega[i][1];
  oldomegas[i][2] = omega[i][2];
  oldomegas[i][3] = x[i][0];
  oldomegas[i][4] = x[i][1];
  oldomegas[i][5] = x[i][2];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixOldOmega::pack_exchange(int i, double *buf)
{
  for (int q = 0; q < 6; q++)
    buf[q] = oldomegas[i][q];
 
  return 6;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixOldOmega::unpack_exchange(int nlocal, double *buf)
{
  for (int q = 0; q < 6; q++)
    oldomegas[nlocal][q] = buf[q];
  
  return 6;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixOldOmega::pack_restart(int i, double *buf)
{
  buf[0] = 7;
  for (int q = 0; q < 6; q++)
    buf[q+1] = oldomegas[i][q];

  return 7;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixOldOmega::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  for (int q = 0; q < 6; q++)
    oldomegas[nlocal][q] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixOldOmega::maxsize_restart()
{
  return 7;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixOldOmega::size_restart(int nlocal)
{
  return 7;
}

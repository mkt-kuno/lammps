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

#include "stdlib.h"
#include "string.h"
#include "fix_energy_boundary.h"
#include "update.h"
#include "memory.h"
#include "force.h"
#include "fix.h"
#include "domain.h"
#include "modify.h"
#include "compute.h"
#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
This is intended for use only with the energy/gran compute. The purpose
of this fix is to update the boundary work on every timestep and
store the boundary work in a restart file, if required [KH - 11 March 
2014]
 ---------------------------------------------------------------------- */

FixEnergyBoundary::FixEnergyBoundary(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  restart_global = 1; //~ Global information is saved to the restart file
  nevery = 1; //~ Set how often to run the end_of_step function

  /*~ Initialise the boundary work, which is calculated cumulatively,
    at zero*/
  boundary_work = 0.0;
  sfound = 0; //~ 0 indicates a compute stress/atom hasn't been found
  pb = 0; //~ Whether or not fix deform/multistress is active
  wallactive = -1; //~ Increased later if walls are present
}

/* ---------------------------------------------------------------------- */

FixEnergyBoundary::~FixEnergyBoundary()
{
  /*~ Delete the stress/atom compute if it was necessary to set
    one up within this fix*/
  if (!sfound) modify->delete_compute("e_stress_comp");
}

/* ---------------------------------------------------------------------- */

int FixEnergyBoundary::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEnergyBoundary::setup(int vflag)
{
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

  //~ Also find fix multistress/deform if either exists
  if (domain->xperiodic || domain->yperiodic || domain->zperiodic) {
    for (int q = 0; q < modify->nfix; q++)
      if (strcmp(modify->fix[q]->style,"multistress") == 0) {
	deffix = modify->fix[q];
	pb = 1;
	break;
      }
  
    if (pb == 0)
      for (int q = 0; q < modify->nfix; q++) {
	if (strcmp(modify->fix[q]->style,"deform") == 0) {
	  deffix = modify->fix[q];
	  pb = 1;
	  break;
	}
      }
  }

  /*~ Check for the presence of walls in the simulation. Issue an
    error if walls are being moved as the work input due to moving
    walls is not being calculated at present*/
  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"wall/gran") == 0) wallactive = i;

  int dim = 0;
  int wallmove = 0;
  if (wallactive >= 0) {
    wallmove += *((int *) modify->fix[wallactive]->extract("wiggle",dim));
    wallmove += *((int *) modify->fix[wallactive]->extract("wtranslate",dim));
    wallmove += *((int *) modify->fix[wallactive]->extract("wscontrol",dim));

    if (wallmove > 0)
      error->all(FLERR,"Boundary work calculation not implemented for moving walls");
  }
}

/* ---------------------------------------------------------------------- */

void FixEnergyBoundary::end_of_step()
{
  //~ Exit prematurely if the boundaries are not moving
  if (pb == 0) return;

  //~ Update the atom stresses if necessary
  if (stressatom->invoked_peratom != update->ntimestep)
    stressatom->compute_peratom();

  /*~ Import a 6-element array of mean stresses from ComputeStressAtom and
    accumulate the means from all processors in tallymeans*/
  double tmeans[6];
  double *means = stressatom->array_export();
  for (int i = 0; i < 6; i++) tmeans[i] = 0.0;
  MPI_Allreduce(&means[0],&tmeans[0],6,MPI_DOUBLE,MPI_SUM,world);
  
  //~ For now, the boundary work is calculated only for a periodic cell
  double deltaW = 0.0; //~ The work input per unit volume (later x V)
  double corre[6]; //~ True strain rates (found from engineering rates)

  if (pb == 1) {
    /*~ Either fix multistress/deform is active; fetch both the strain 
      rates and the initial positions of the periodic boundaries. The
      latter are used to convert engineering strain rates to true
      strain rates.*/
    double *ierates = deffix->param_export();
    double *pstart = deffix->extract_pboundstart();

    for (int i = 0; i < 6; i++) corre[i] = ierates[i];
    for (int i = 0; i < 3; i++) corre[i] *= (pstart[2*i+1]-pstart[2*i])/(domain->boxhi[i] - domain->boxlo[i]);

    //~ Eq. 1.24, p. 20, "Soil Behavior and Critical State Soil Mechanics"
    for (int i = 0; i < 6; i++) deltaW -= tmeans[i]*corre[i];

    deltaW *= update->dt; //~ strain rate * timestep = strain increment

    //~ Multiply by current cell volume to find increment of work input
    for (int i = 0; i < domain->dimension; i++)
      deltaW *= (domain->boxhi[i] - domain->boxlo[i]);
  }

  boundary_work += deltaW; //~ Update the boundary work
}

/* ---------------------------------------------------------------------- */

void *FixEnergyBoundary::extract(const char *str, int &dim)
{
  /*~ This function was added so that the accumulated boundary work
    may be extracted by ComputeEnergyGran [KH - 11 March 2014]*/
  dim = 0;
  if (strcmp(str,"boundary_work") == 0) return (void *) &boundary_work;
  return NULL;
}

/* ---------------------------------------------------------------------- */

void FixEnergyBoundary::write_restart(FILE *fp)
{
  int n = 0;
  double list[1];
  list[n++] = boundary_work;
  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ---------------------------------------------------------------------- */

void FixEnergyBoundary::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;
  boundary_work = static_cast<double> (list[n++]);
}

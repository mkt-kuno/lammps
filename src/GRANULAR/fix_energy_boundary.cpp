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
#include "fix_wall_gran.h"  // added to consider wall positions [MO - 20 Aug 2015]

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
    at zero. Also initialise the two divisions of boundary work,
    volumetric work and distortional work, at 0. [KH - 1 April 2015]*/
  boundary_work = deltawv = deltawd = 0.0;
  sfound = 0; //~ 0 indicates a compute stress/atom hasn't been found
  pb = 0; //~ Whether or not fix deform/multistress is active
  wallactive = -1; //~ Increased later if walls are present

  /*~ Initialise the strain rates for the preceding time-step at
    enormous numbers [KH - 18 August 2015]*/
  for (int i = 0; i < 6; i++) oldierates[i] = 100000.0;
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

  //*************************************************************************
  // wiggle and wsconstrol can be used for energy/boundary [MO - 22 Aug 2015]
  int dim = 0;
  wiggle = 0;
  wtranslate = 0;
  wscontrol = 0;
  for (int i = 0; i < modify->nfix; i++) {
    if (strcmp(modify->fix[i]->style,"wall/gran") == 0) {
      wallactive = i;  
      wiggle     += *((int *) modify->fix[wallactive]->extract("wiggle",dim));
      wtranslate += *((int *) modify->fix[wallactive]->extract("wtranslate",dim)); 	          
      wscontrol  += *((int *) modify->fix[wallactive]->extract("wscontrol",dim));
    }
  }
  if (wtranslate > 0)
    error->all(FLERR,"Boundary work calculation not implemented for translate walls");
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
  
  //~ For now, the work terms are calculated only for a periodic cell
  double onethird = 1.0/3.0;
  double pdash, q, dep, deq;
  double deltaW, ideltawv, ideltawd;
  deltaW = ideltawv = ideltawd = 0.0;
  
  int dim = 0;
  if (pb == 1 && wiggle == 0) {
    /*~ Either fix multistress/deform is active; fetch the true strain 
      rates using the param_export function.*/
    double *ierates = deffix->param_export();

    //~ Use the best available strain rates for last step [KH - 18 August 2015]
    if (oldierates[2] > 99999.0)
      for (int i = 0; i < 6; i++) oldierates[i] = ierates[i];

    double w_ierates[3];
    w_ierates[0] = w_ierates[1] = w_ierates[2] = 0.0;
    double big = 1.0e19;      
    // fetch the updated true strain rate from fix/wall/gran [MO 20 Aug 2015]
    double w_ierates_sum = 0.0;        
    for (int i = 0; i < modify->nfix; i++) {
      if (strcmp(modify->fix[i]->style,"wall/gran") == 0 &&
	  (*((int *) modify->fix[i]->extract("wscontrol",dim)))) {
	double wall_hi_i = *((double *) modify->fix[i]->extract("hi",dim));
	double wall_lo_i = *((double *) modify->fix[i]->extract("lo",dim));
	int wallstyle_i = *((int *) modify->fix[i]->extract("wallstyle",dim)); 
	double w_ierates_i = *((double *) modify->fix[i]->extract("w_ierates",dim));
	
	if (wall_hi_i < big)   w_ierates_sum += w_ierates_i;
	if (wall_lo_i > - big) w_ierates_sum -= w_ierates_i;
	w_ierates[wallstyle_i] = w_ierates_sum;
      }
    }

    //~ Calculate the volume of the periodic cell on previous time-step
    double cellvolume = 1.0;
    for (int i = 0; i < domain->dimension; i++) {
      if (domain->periodicity[i] == 0) {
	cellvolume *= (domain->w_boxhi[i] - domain->w_boxlo[i])*(1.0 - w_ierates[i]*update->dt);
      }
      else cellvolume *= (domain->boxhi[i] - domain->boxlo[i])*(1.0 - ierates[i]*update->dt);
    }

    //~ Eq. 1.24, p. 20, "Soil Behavior and Critical State Soil Mechanics"
    // for (int i = 0; i < 6; i++) deltaW -= tmeans[i]*oldierates[i];  
    // modified to consider stresscontrol [MO - 21 Aug 2015]
    for (int i = 0; i < 3; i++) {
      if (domain->periodicity[i] == 0) {
	if (wscontrol > 0) deltaW -= tmeans[i]*w_ierates[i];
	else           deltaW -= tmeans[i]* 0.0;
      }
      else             deltaW -= tmeans[i]*oldierates[i]; 
    }
    for (int i = 3; i < 6; i++) deltaW -= tmeans[i]*oldierates[i];   
    
    // deltaW *= update->dt; //~ strain rate * timestep = strain increment
    /*~ Find the increment of boundary work input by multiplying by 
      the current cell volume and by time step (strain rate * time step
      = strain increment) [KH - 1 April 2015]*/
    deltaW *= (update->dt* cellvolume);

    //~ Now find the subdivided increments of boundary work
    pdash = (tmeans[0] + tmeans[1] + tmeans[2])*onethird;
    q = sqrt(0.5*((tmeans[0]-tmeans[1])*(tmeans[0]-tmeans[1])+(tmeans[0]-tmeans[2])*(tmeans[0]-tmeans[2])+(tmeans[2]-tmeans[1])*(tmeans[2]-tmeans[1])) + 3.0*(tmeans[3]*tmeans[3] + tmeans[4]*tmeans[4] + tmeans[5]*tmeans[5]));

    dep = (oldierates[0] + oldierates[1] + oldierates[2])*update->dt;
    deq = 2.0*(oldierates[2] - 0.5*(oldierates[0]+oldierates[1]))*update->dt*onethird;

    ideltawv = -pdash*dep*cellvolume;
    ideltawd = -q*deq*cellvolume;

    //~ Store strain rates for the next time-step [KH - 18 August 2015]
    for (int i = 0; i < 6; i++) oldierates[i] = ierates[i];
  }
       
  //***************************************************************
  // Include the work done by wiggle commnad [MO - 21 Aug 2015]
  if (wiggle > 0) { 
    for (int i = 0; i < modify->nfix; i++) {
      if (strcmp(modify->fix[i]->style,"wall/gran") == 0 &&
	  (*((int *) modify->fix[i]->extract("wiggle",dim)))) {
	// this should not be active for stresscontrol command
	double velwall = *((double *) modify->fix[i]->extract("velwall",dim));
	double fwall_all = *((double *) modify->fix[i]->extract("fwall_all",dim));      
	deltaW += velwall * fwall_all * update->dt;
      }
    }
  }
  
  boundary_work += deltaW; //~ Update the boundary work
  deltawv += ideltawv; //~ the volumetric work...
  deltawd += ideltawd; //~ and the distortional work
}

/* ---------------------------------------------------------------------- */

void *FixEnergyBoundary::extract(const char *str, int &dim)
{
  /*~ This function was added so that the accumulated boundary work
    may be extracted by ComputeEnergyGran [KH - 11 March 2014]*/
  dim = 0;
  if (strcmp(str,"boundary_work") == 0) return (void *) &boundary_work;
  else if (strcmp(str,"deltawv") == 0) return (void *) &deltawv;
  else if (strcmp(str,"deltawd") == 0) return (void *) &deltawd;
  return NULL;
}

/* ---------------------------------------------------------------------- */

void FixEnergyBoundary::write_restart(FILE *fp)
{
  int n = 0;
  double list[9];
  list[n++] = boundary_work;
  list[n++] = deltawv;
  list[n++] = deltawd;
  list[n++] = oldierates[0];
  list[n++] = oldierates[1];
  list[n++] = oldierates[2];
  list[n++] = oldierates[3];
  list[n++] = oldierates[4];
  list[n++] = oldierates[5];
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
  deltawv = static_cast<double> (list[n++]);
  deltawd = static_cast<double> (list[n++]);
  oldierates[0] = static_cast<double> (list[n++]);
  oldierates[1] = static_cast<double> (list[n++]);
  oldierates[2] = static_cast<double> (list[n++]);
  oldierates[3] = static_cast<double> (list[n++]);
  oldierates[4] = static_cast<double> (list[n++]);
  oldierates[5] = static_cast<double> (list[n++]);
}

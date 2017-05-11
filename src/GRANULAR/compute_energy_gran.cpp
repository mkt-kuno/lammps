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

#include <string.h>
#include "compute_energy_gran.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair.h"
#include "fix.h"
#include "domain.h"

using namespace LAMMPS_NS;

enum{FRICTION,RKINETIC,TKINETIC,KINETIC,NSTRAIN,SSTRAIN,STRAIN,BOUNDARY,VOLUMETRIC,DISTORTIONAL,LDAMP,VDAMP,DAMP,SPIN};

/* ---------------------------------------------------------------------- */

ComputeEnergyGran::ComputeEnergyGran(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal compute energy/gran command");

  /*~ As the number of columns is initially unknown (and possibly zero),
    begin with peratom_flag set to 0 and increase later if necessary*/
  peratom_flag = 0;
  size_peratom_cols = 0; //~ If 1 column, increment once and so on
  pairenergy = 0; //~ Indicates whether energy needs to be tracked in pairstyle

  /*~ Read in the user-defined inputs. The order of the inputs
    corresponds exactly to the ordering in the output vector.*/
  length_enum = 14;
  inputs = new int[length_enum];
  for (int i = 0; i < length_enum; i++) inputs[i] = -1;

  for (int iarg = 3; iarg < narg; iarg++) {
    if (strcmp(arg[iarg],"friction") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = FRICTION;
      else error->all(FLERR,"Duplicated friction input to ComputeEnergyGran");
      pairenergy = 1;
    } else if (strcmp(arg[iarg],"rotational_kinetic") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = RKINETIC;
      else error->all(FLERR,"Duplicated rotational_kinetic input to ComputeEnergyGran");
      peratom_flag = 1;
      size_peratom_cols++;
    } else if (strcmp(arg[iarg],"translational_kinetic") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = TKINETIC;
      else error->all(FLERR,"Duplicated translational_kinetic input to ComputeEnergyGran");
      peratom_flag = 1;
      size_peratom_cols++;
    } else if (strcmp(arg[iarg],"kinetic") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = KINETIC;
      else error->all(FLERR,"Duplicated kinetic input to ComputeEnergyGran");
      peratom_flag = 1;
      size_peratom_cols++;
    } else if (strcmp(arg[iarg],"normal_strain") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = NSTRAIN;
      else error->all(FLERR,"Duplicated normal strain input to ComputeEnergyGran");
      pairenergy = 1;
    } else if (strcmp(arg[iarg],"shear_strain") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = SSTRAIN;
      else error->all(FLERR,"Duplicated shear strain input to ComputeEnergyGran");
      pairenergy = 1;
    } else if (strcmp(arg[iarg],"strain") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = STRAIN;
      else error->all(FLERR,"Duplicated strain input to ComputeEnergyGran");
      pairenergy = 1;
    } else if (strcmp(arg[iarg],"boundary") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = BOUNDARY;
      else error->all(FLERR,"Duplicated boundary input to ComputeEnergyGran");
    } else if (strcmp(arg[iarg],"volumetric") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = VOLUMETRIC; //~ [KH - 1 April 2015]
      else error->all(FLERR,"Duplicated volumetric input to ComputeEnergyGran");
    } else if (strcmp(arg[iarg],"distortional") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = DISTORTIONAL; //~ [KH - 1 April 2015]
      else error->all(FLERR,"Duplicated distortional input to ComputeEnergyGran");
    } else if (strcmp(arg[iarg],"local_damping") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = LDAMP;
      else error->all(FLERR,"Duplicated local damping input to ComputeEnergyGran");
    } else if (strcmp(arg[iarg],"viscous_damping") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = VDAMP;
      else error->all(FLERR,"Duplicated viscous damping input to ComputeEnergyGran");
    } else if (strcmp(arg[iarg],"damping") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = DAMP;
      else error->all(FLERR,"Duplicated damping input to ComputeEnergyGran");
      // Added for D_spin [MO - 15 November 2014]
    } else if (strcmp(arg[iarg],"spin_energy") == 0) {
      if (inputs[iarg-3] < 0) inputs[iarg-3] = SPIN;
      else error->all(FLERR,"Duplicated spin energy input to ComputeEnergyGran");
      pairenergy = 1;
    } else error->all(FLERR,"Invalid input to ComputeEnergyGran");
  }

  nmax = 0;
  evector = NULL;
  earray = NULL;
  
  vector_flag = 1;
  extvector = 1; //~ Intensive quantities are stored in the vector
  size_vector = narg-3;
  vector = new double[size_vector];

  wallactive = 0; //~ Increased later if walls are present in the simulation
  wallcheck = 0; //~ 0 indicates the check for walls must still be done

  dampactive[0] = dampactive[1] = -1; //~ Increased later if damping present
  dampcheck[0] = dampcheck[1] = 0; //~ The check for damping must still be done

  /*~ Set up a new fix if boundary and/or volumetric and/or distortional work
    need to be calculated. This is in the constructor as the setmask function
    of this fix needs to be run before modify->init (otherwise the end_of_step
    function will not be invoked)*/
  for (int i = 0; i < size_vector; i++)
    if (inputs[i] > 6 && inputs[i] < 10) {
      add_fix_energy_boundary();
      break;
    }
}

/* ---------------------------------------------------------------------- */

ComputeEnergyGran::~ComputeEnergyGran()
{
  memory->destroy(evector);
  memory->destroy(earray);
  delete [] vector;
  delete [] inputs;
}

/* ---------------------------------------------------------------------- */

void ComputeEnergyGran::init()
{
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"energy/gran") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute energy/gran");

  //~ Confirm that a granular pairstyle is used
  if (!force->pair_match("gran",0))
    error->all(FLERR,"A granular pairstyle must be used for ComputeEnergyGran");
}

/* ---------------------------------------------------------------------- */

void ComputeEnergyGran::compute_vector()
{
  invoked_vector = update->ntimestep;

  //~ Call helper functions for each energy term to populate vector
  for (int i = 0; i < size_vector; i++) {
    if (inputs[i] == 0) vector[i] = pair_extract("dissipfriction"); //~ FRICTION
    else if (inputs[i] == 1) vector[i] = kinetic_extract(0); //~ RKINETIC
    else if (inputs[i] == 2) vector[i] = kinetic_extract(1); //~ TKINETIC
    else if (inputs[i] == 3) vector[i] = kinetic_extract(2); //~ KINETIC
    else if (inputs[i] == 4) vector[i] = pair_extract("normalstrain"); //~ NSTRAIN
    else if (inputs[i] == 5) vector[i] = pair_extract("shearstrain"); //~ SSTRAIN
    else if (inputs[i] == 6) vector[i] = pair_extract("normalstrain") 
			       + pair_extract("shearstrain"); //~ STRAIN
    else if (inputs[i] == 7) //~ Fetch the boundary work from FixEnergyBoundary
      vector[i] = *((double *) deffix->extract("boundary_work",dim)); //~ BOUNDARY
    else if (inputs[i] == 8)
      vector[i] = *((double *) deffix->extract("deltawv",dim)); //~ VOLUMETRIC
    else if (inputs[i] == 9)
      vector[i] = *((double *) deffix->extract("deltawd",dim)); //~ DISTORTIONAL
    else if (inputs[i] == 10) vector[i] = damping_extract("damp/local",0); //~ LDAMP
    else if (inputs[i] == 11) vector[i] = damping_extract("viscous",1); //~ VDAMP
    else if (inputs[i] == 12) vector[i] = damping_extract("damp/local",0)
				+ damping_extract("viscous",1); //~ DAMP
    else if (inputs[i] == 13) vector[i] = pair_extract("spinenergy"); //~ SPIN
  }
}

/* ---------------------------------------------------------------------- */

double ComputeEnergyGran::pair_extract(const char *str)
{
  int wallsfound = 0;
  double singleproc;
  double gathered = 0.0;

  //~ Get the data from the pairstyle
  Pair *pair = force->pair_match("gran",0);
  singleproc = *((double *) pair->extract(str,dim));

  /*~ If walls are being used, also need energy for ball-wall contacts.
    wallactive stores number of walls present [KH - 25 July 2014]*/
  if (!wallcheck) {
    for (int i = 0; i < modify->nfix; i++)
      if (strcmp(modify->fix[i]->style,"wall/gran") == 0) wallactive++;
    wallcheck = 1;
  }

  if (wallactive > 0) //~ There is at least one wall
    for (int i = 0; i < modify->nfix; i++)
      if (strcmp(modify->fix[i]->style,"wall/gran") == 0) {
	singleproc += *((double *) modify->fix[i]->extract(str,dim));
	wallsfound++;
	if (wallactive == wallsfound) break;
      }

  //~ Accumulate the data from all processors
  MPI_Allreduce(&singleproc,&gathered,1,MPI_DOUBLE,MPI_SUM,world);

  return gathered;
}

/* ---------------------------------------------------------------------- */

double ComputeEnergyGran::kinetic_extract(int i)
{
  //~ Update the kinetic energy terms if necessary
  if (invoked_peratom != update->ntimestep) compute_peratom();

  double gathered = 0.0;
  MPI_Allreduce(&kinetic[i],&gathered,1,MPI_DOUBLE,MPI_SUM,world);

  return gathered;
}

/* ---------------------------------------------------------------------- */

double ComputeEnergyGran::damping_extract(const char *str, int p)
{
  double singleproc;
  double gathered = 0.0;

  /*~ Firstly determine whether damping is active if this check
    has not already been done*/
  if (!dampcheck[p]) {
    for (int i = 0; i < modify->nfix; i++)
      if (strcmp(modify->fix[i]->style,str) == 0) dampactive[p] = i;
    dampcheck[p] = 1;
  }

  if (dampactive[p] >= 0) {
    singleproc = *((double *) modify->fix[dampactive[p]]->extract("energy_dissip",dim));
    MPI_Allreduce(&singleproc,&gathered,1,MPI_DOUBLE,MPI_SUM,world);
  }

  return gathered; //~ Will be zero if damping is inactive
}

/* ---------------------------------------------------------------------- */

void ComputeEnergyGran::add_fix_energy_boundary()
{
  //~ Check whether this fix is already present
  int feb = -1;
  for (int q = 0; q < modify->nfix; q++)
    if (strcmp(modify->fix[q]->style,"energy/boundary") == 0) feb = q;

  if (feb < 0) { //~ Fix not presently active
    char **newarg = new char*[3];
    newarg[0] = (char *) "ceg_feb";
    newarg[1] = (char *) "all";
    newarg[2] = (char *) "energy/boundary";

    modify->add_fix(3,newarg);
    delete [] newarg;
  }
  
  //~ Set pointers for this newly-created fix
  int accfix = modify->find_fix("ceg_feb");
  if (accfix < 0) error->all(FLERR,"Fix ID for fix energy/boundary does not exist");
  deffix = modify->fix[accfix];
}

/* ---------------------------------------------------------------------- */

void ComputeEnergyGran::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow energy array if necessary
  if (atom->nmax > nmax) {
    nmax = atom->nmax;

    if (size_peratom_cols == 1) {
      memory->destroy(evector);
      memory->create(evector,nmax,"energy/gran:evector");
      vector_atom = evector;
    } else {
      memory->destroy(earray);
      memory->create(earray,nmax,size_peratom_cols,"energy/gran:earray");
      array_atom = earray;
    }
  }

  //~ Set the accumulated kinetic energies to zero
  kinetic[0] = kinetic[1] = kinetic[2] = 0.0;

  /*~ Entries 2-4 are for kinetic energy in the enumeration.
    Parse the inputs vector to identify which is/are active
    and the order. The order of entries in the atom matrix
    must correspond to that specified by the user.*/
  int activeke[3], counter = 0;
  for (int i = 0; i < 3; i++) activeke[i] = -1;

  for (int i = 0; i < length_enum; i++)
    if (inputs[i] > 0 && inputs[i] < 4) {
      activeke[inputs[i]-1] = counter;
      counter++;
    }

  double energy;
  double mvv2e = force->mvv2e;
  double **v = atom->v;
  double **omega = atom->omega;
  double *radius = atom->radius;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  //~ Rotational kinetic energy is first in the enumeration
  if (activeke[0] >= 0)
    for (int i = 0; i < nlocal; i++) {
      energy = peratom_rke(omega[i][0],omega[i][1],omega[i][2],radius[i]);
      rmass ? energy *= mvv2e*rmass[i] : energy *= mvv2e*mass[type[i]];
      size_peratom_cols == 1 ? evector[i] = energy : earray[i][activeke[0]] = energy;
      if (mask[i] & groupbit) kinetic[0] += energy;
    }
  
  //~ Translational kinetic energy is second in the enumeration
  if (activeke[1] >= 0)
    for (int i = 0; i < nlocal; i++) {
      energy = peratom_tke(v[i][0],v[i][1],v[i][2]);
      rmass ? energy *= mvv2e*rmass[i] : energy *= mvv2e*mass[type[i]];
      size_peratom_cols == 1 ? evector[i] = energy : earray[i][activeke[1]] = energy;
      if (mask[i] & groupbit) kinetic[1] += energy;
    }

  //~ Total kinetic energy is the third and final KE term
  if (activeke[2] >= 0)
    for (int i = 0; i < nlocal; i++) {
      energy = peratom_rke(omega[i][0],omega[i][1],omega[i][2],radius[i]);
      energy += peratom_tke(v[i][0],v[i][1],v[i][2]);
      rmass ? energy *= mvv2e*rmass[i] : energy *= mvv2e*mass[type[i]];
      size_peratom_cols == 1 ? evector[i] = energy : earray[i][activeke[2]] = energy;
      if (mask[i] & groupbit) kinetic[2] += energy;
    }
}

/* ---------------------------------------------------------------------- */

double ComputeEnergyGran::peratom_rke(double a, double b, double c, double r)
{
  double e = 0.2*r*r*(a*a + b*b + c*c);
  return e;
}

/* ---------------------------------------------------------------------- */

double ComputeEnergyGran::peratom_tke(double a, double b, double c)
{
  double e = 0.5*(a*a + b*b + c*c);
  return e;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeEnergyGran::memory_usage()
{
  double bytes = nmax * size_peratom_cols * sizeof(double);
  return bytes;
}

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

#include <mpi.h> //~ Added for MPI gathering of quantities on different procs
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fix_fluiddrag.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "respa.h"
#include "error.h"
#include "memory.h"
#include "group.h"
#include "modify.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CHUTE,SPHERICAL,GRADIENT,VECTOR};

/* ---------------------------------------------------------------------- */

FixFluidDrag::FixFluidDrag(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal fix fluiddrag command");

  dimension = domain->dimension;
  if (dimension == 3) numcols = 11;
  else numcols = 9;

  peratom_flag = 1;
  size_peratom_cols = numcols;
  peratom_freq = 1;
  nevery = 1; //~ Set how often to run the end_of_step function

  hydgrad = force->numeric(FLERR,arg[3]); //~ The hydraulic gradient
  compmethod = force->inumeric(FLERR,arg[4]); //~ Read in the computational method

  //~ Check that the latter is sensible
  if (compmethod < 0 || compmethod > 1)
    error->all(FLERR,"Only 0 and 1 are defined for the computational method in fix fluiddrag");

  fddata = NULL; //~ Added to initialise the array for data storage
  grow_arrays(atom->nmax);
  atom->add_callback(0); //~ Add a callback to grow the array if necessary

  MPI_Comm_rank(world,&me); //~ Identify the processor rank

  /*~ The specific weight of water is 'hard-coded' here for programming
    convenience and to simplify the input command. Make sure that SI
    units are being used; otherwise the numbers don't make sense.*/
  if (strcmp(update->unit_style,"si") != 0)
    error->all(FLERR,"fix fluiddrag is defined only for SI units");

  specificweight = 9810; //~ kg/(m2.s2)

  if (strcmp(arg[5],"chute") == 0) {
    if (narg != 7) error->all(FLERR,"Illegal fix fluiddrag command");
    style = CHUTE;
    phi = 0.0;
    theta = 180.0 - force->numeric(FLERR,arg[6]);
  } else if (strcmp(arg[5],"spherical") == 0) {
    if (narg != 8) error->all(FLERR,"Illegal fix fluiddrag command");
    style = SPHERICAL;
    phi = force->numeric(FLERR,arg[6]);
    theta = force->numeric(FLERR,arg[7]);
  } else if (strcmp(arg[5],"gradient") == 0) {
    if (narg != 10) error->all(FLERR,"Illegal fix fluiddrag command");
    style = GRADIENT;
    phi = force->numeric(FLERR,arg[6]);
    theta = force->numeric(FLERR,arg[7]);
    phigrad = force->numeric(FLERR,arg[8]);
    thetagrad = force->numeric(FLERR,arg[9]);
  } else if (strcmp(arg[5],"vector") == 0) {
    if (narg != 9) error->all(FLERR,"Illegal fix fluiddrag command");
    style = VECTOR;
    xdir = force->numeric(FLERR,arg[6]);
    ydir = force->numeric(FLERR,arg[7]);
    zdir = force->numeric(FLERR,arg[8]);
  } else error->all(FLERR,"Illegal fix fluiddrag command");

  PI = 4.0*atan(1.0);
  degree2rad = PI/180.0;

  if (style == CHUTE || style == SPHERICAL || style == GRADIENT) {
    if (dimension == 3) {
      xgrav = sin(degree2rad * theta) * cos(degree2rad * phi);
      ygrav = sin(degree2rad * theta) * sin(degree2rad * phi);
      zgrav = cos(degree2rad * theta);
    } else {
      xgrav = sin(degree2rad * theta);
      ygrav = cos(degree2rad * theta);
      zgrav = 0.0;
    }
  } else if (style == VECTOR) {
    if (dimension == 3) {
      double length = sqrt(xdir*xdir + ydir*ydir + zdir*zdir);
      xgrav = xdir/length;
      ygrav = ydir/length;
      zgrav = zdir/length;
    } else {
      double length = sqrt(xdir*xdir + ydir*ydir);
      xgrav = xdir/length;
      ygrav = ydir/length;
      zgrav = 0.0;
    }
  }

  time_origin = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

FixFluidDrag::~FixFluidDrag()
{
  atom->delete_callback(id,0); //~ Unregister callbacks from the Atom class
  memory->destroy(fddata); //~ Destroy the local data array
}

/* ---------------------------------------------------------------------- */

int FixFluidDrag::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFluidDrag::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  /*~ Firstly calculate the total volume of particles. It is 
    assumed that this does not change throughout the simulation.
    Also calculate the mean effective particle diameter and the
    largest particle diameter.*/
  double pvolume = 0.0; //~ Volume of particles on a processor
  double deff = 0.0; //~ The mean effective particle diameter
  double maxrad = 0.0; //~ The maximum particle radius
  double temppvolume;
  int ltag = 0; //~ The tag of the largest particle

  tagint *tag = atom->tag;
  int *mask = atom->mask;
  double *radius = atom->radius;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (dimension == 3) temppvolume = (4*PI/3)*radius[i]*radius[i]*radius[i];
      else temppvolume = PI*radius[i]*radius[i];

      pvolume += temppvolume;
      deff += temppvolume/(2.0*radius[i]);

      if (radius[i] > maxrad) {
	maxrad = radius[i];
	ltag = tag[i];
      }
    }
  
  //~ Gather pvolume and deff from all processors in totalpvolume
  totalpvolume = totaldeff = overallmaxrad = 0.0;
  MPI_Allreduce(&pvolume,&totalpvolume,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&deff,&totaldeff,1,MPI_DOUBLE,MPI_SUM,world);

  //~ Now ascertain which processor has the atom of largest radius and its tag
  int nprocs,*recvcounts,*brecvcounts,*displs,*bdispls;
  MPI_Comm_size(world,&nprocs);
  recvcounts = new int[nprocs];
  brecvcounts = new int[nprocs];
  displs = new int[nprocs];
  bdispls = new int[nprocs];

  for (int iproc = 0; iproc < nprocs; iproc++) {
    recvcounts[iproc] = 1;
    brecvcounts[iproc] = 2;
    displs[iproc] = iproc;
    bdispls[iproc] = 2*iproc;
  }

  int sendarray[2];
  sendarray[0] = me;
  sendarray[1] = ltag;

  //~ Gather the maximum radii for all processors in the allvals array
  double allvals[nprocs];
  int ballvals[2*nprocs];
  MPI_Allgatherv(&maxrad,1,MPI_DOUBLE,&allvals[0],recvcounts,displs,MPI_DOUBLE,world);
  MPI_Allgatherv(&sendarray[0],2,MPI_INT,&ballvals[0],brecvcounts,bdispls,MPI_INT,world);

  //~ Find the largest of the array members
  for (int iproc = 0; iproc < nprocs; iproc++) {
    if (allvals[iproc] > overallmaxrad) {
      overallmaxrad = allvals[iproc];
      procandtag[0] = ballvals[2*iproc];
      procandtag[1] = ballvals[2*iproc+1];
    }
  }

  delete [] recvcounts;
  delete [] brecvcounts;
  delete [] displs;
  delete [] bdispls;

  //~ Invert totaldeff and multiply by totalpvolume
  totaldeff = totalpvolume/totaldeff;

  //~ Calculate the pressure gradient from hydraulic gradient
  pgradient = specificweight*hydgrad;

  //~ Confirm that the largest atom is unique
  int counter,totalcounter;
  counter = 0;

  for (int i = 0; i < nlocal; i++)
    if ((mask[i] & groupbit) && radius[i] == overallmaxrad)
      counter++;

  MPI_Allreduce(&counter,&totalcounter,1,MPI_INT,MPI_SUM,world);

  if (compmethod == 0 && totalcounter != 1)
    error->all(FLERR,"More than one atom has the largest diameter in the system so computational method 0 is inappropriate");

  //~ Initialise the data storage array, firstly calculating some data
  double **x = atom->x;
  double totalvolume = 1.0;
  for (int i = 0; i < dimension; i++)
    totalvolume *= (domain->boxhi[i]-domain->boxlo[i]);
  double porosity = (totalvolume - totalpvolume)/totalvolume;
  double fdragmono = -1.0*pgradient*PI*totaldeff*totaldeff*totaldeff/(6.0*(1.0-porosity));

  for (int i = 0; i < nlocal; i++) {
    fddata[i][0] = tag[i];
    for (int q = 0; q < dimension; q++)
      fddata[i][q+1] = x[i][q];
    
    fddata[i][dimension+1] = radius[i];
    fddata[i][dimension+2] = totalvolume;
    fddata[i][dimension+3] = 0.0;
    fddata[i][dimension+4] = 0.0;
    if (dimension == 3) fddata[i][dimension+5] = 0.0;
    fddata[i][numcols-2] = porosity;
    fddata[i][numcols-1] = fdragmono;
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidDrag::post_force(int vflag)
{
  //~ Find the current porosity of the sample
  double totalvolume = 1.0;
  
  for (int i = 0; i < dimension; i++)
    totalvolume *= (domain->boxhi[i]-domain->boxlo[i]);
       
  double porosity = (totalvolume - totalpvolume)/totalvolume;
  double oneminusporosity = 1.0-porosity;

  //~ Calculate the drag force using the effective diameter
  double fdragmono = -1.0*pgradient*PI*totaldeff*totaldeff*totaldeff/(6.0*oneminusporosity);

  // update direction of gravity vector if gradient style

  if (style == GRADIENT) {
    double theta_current = degree2rad * 
      (theta + (update->ntimestep - time_origin)*update->dt*thetagrad*360.0);
    if (dimension == 3) {
      double phi_current = degree2rad * 
	(phi + (update->ntimestep - time_origin)*update->dt*phigrad*360.0);
      xgrav = sin(theta_current) * cos(phi_current);
      ygrav = sin(theta_current) * sin(phi_current);
      zgrav = cos(theta_current);
    } else {
      xgrav = sin(theta_current);
      ygrav = cos(theta_current);
    }
  }

  double rdiami,finteraction;
  double twooverdeff = 2.0/totaldeff; //~ Defined for efficiency
  double *radius = atom->radius; //~ Need the radii below
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag; //~ Use atom tags for ID
  double **x = atom->x; //~ Write out the coordinates for convenience

  //~ Now do the calculation, updating the forces as normal for all particles
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      rdiami = radius[i]*twooverdeff;
      finteraction = fdragmono*rdiami*rdiami*(porosity + oneminusporosity*rdiami);

      //~ Assume that finteraction is partitioned in proportion to x/y/zgrav
      f[i][0] += xgrav*finteraction;
      f[i][1] += ygrav*finteraction;
      if (dimension == 3) f[i][2] += zgrav*finteraction;

      //~ Write data to be stored to an array
      fddata[i][0] = tag[i];
      for (int q = 0; q < dimension; q++)
	fddata[i][q+1] = x[i][q];
	
      fddata[i][dimension+1] = radius[i];
      fddata[i][dimension+2] = totalvolume;
      fddata[i][dimension+3] = xgrav*finteraction;
      fddata[i][dimension+4] = ygrav*finteraction;
      if (dimension == 3) fddata[i][dimension+5] = zgrav*finteraction;
      fddata[i][numcols-2] = porosity;
      fddata[i][numcols-1] = fdragmono;
    }
}	       

/* ---------------------------------------------------------------------- */

void FixFluidDrag::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixFluidDrag::end_of_step()
{
  //~ Velocities are zeroed in this function
  
  /*~ Firstly find the force on, and velocity of, the largest particle.
    Note that because its position remains unchanged (because of the 
    good work of this function), it will remain on the same processor 
    so we don't have to track its movements from proc to proc.*/

  if (compmethod == 0) {
    double **v = atom->v;
    double *radius = atom->radius;
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    tagint *tag = atom->tag;
    int *type = atom->type;
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    double largestvel[3] = {0.0};
    double totallargestvel[3] = {0.0};
    int check = 0;

    /*~ Now update the velocities to reflect the fact that
      they will be updated later based on the forces*/
    if (me == procandtag[0]) {
      for (int i = 0; i < nlocal; i++)
      	if ((mask[i] & groupbit) && tag[i] == procandtag[1]) {
      	  for (int j = 0; j < 3; j++) {
      	    if (rmass) largestvel[j] = v[i][j] + f[i][j]*update->dt/(2.0*rmass[i]);
	    else largestvel[j] = v[i][j] + f[i][j]*update->dt/(2.0*mass[type[i]]);
	  }
	  check++;
	  break;
	}
      if (check != 1) error->all(FLERR,"The tag of the largest atom has been lost");
    }

    //~ All but one set of largestvel will be zero so can do a simple sum
    MPI_Allreduce(&largestvel[0],&totallargestvel[0],3,MPI_DOUBLE,MPI_SUM,world);

    //~ Finally subtract these velocities from the velocities of all atoms
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
    	for (int j = 0; j < 3; j++)
    	  v[i][j] -= totallargestvel[j];
  }
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixFluidDrag::grow_arrays(int nmax)
{
  memory->grow(fddata,nmax,numcols,"fluiddrag:fddata");
  array_atom = fddata;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixFluidDrag::copy_arrays(int i, int j, int delflag)
{
  for (int q = 0; q < numcols; q++)
    fddata[j][q] = fddata[i][q];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixFluidDrag::pack_exchange(int i, double *buf)
{
  for (int q = 0; q < numcols; q++)
    buf[q] = fddata[i][q];

  return numcols;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixFluidDrag::unpack_exchange(int nlocal, double *buf)
{
  for (int q = 0; q < numcols; q++)
    fddata[nlocal][q] = buf[q];

  return numcols;
}

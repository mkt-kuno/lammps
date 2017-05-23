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

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "fix_write_insurance_shear_history.h"
#include "atom.h"
#include "update.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "error.h"
#include "fix.h"
#include "memory.h"
#include "force.h"
#include "pair.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixWriteInsuranceShearHistory::FixWriteInsuranceShearHistory(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg) {}

/* ---------------------------------------------------------------------- */

FixWriteInsuranceShearHistory::~FixWriteInsuranceShearHistory() {}

/* ---------------------------------------------------------------------- */

int FixWriteInsuranceShearHistory::setmask() {return 0;}

/* ---------------------------------------------------------------------- */

void FixWriteInsuranceShearHistory::setup(int vflag)
{
  //~ In here, the data is written out for shear

  //~ Confirm that this is a serial implementation
  int nprocs;
  MPI_Comm_size(world,&nprocs);
  if (nprocs > 1) error->all(FLERR,"One core must be used with fix_write_insurance_shear_history");

  int i,j,ii,jj,inum,jnum;
  double delx,dely,delz,fx,fy,fz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *shear,*allshear,**firstshear;

  double **x = atom->x;
  double *radius = atom->radius;
  int *mask = atom->mask;
  tagint *tag = atom->tag; //~ Need to use atom tags for ID

  //~ Gain access to arrays in pairstyle using pair->extract
  Pair *pair;

  if (force->pair_match("gran/hooke/history",1)) 
    pair = force->pair_match("gran/hooke/history",1);
  else if (force->pair_match("gran/hertz/history",1))
    pair = force->pair_match("gran/hertz/history",1);
  else if (force->pair_match("gran/shm/history",1))
    pair = force->pair_match("gran/shm/history",1);
  else if (force->pair_match("gran/CM/history",1))
    pair = force->pair_match("gran/CM/history",1);
  else if (force->pair_match("gran/HMD/history",1))  // Added this [MO - 21 July 2014]
    pair = force->pair_match("gran/HMD/history",1);
  else if (force->pair_match("gran/CMD/history",1))  // Added this [MO - 18 November 2014]
    pair = force->pair_match("gran/CMD/history",1);
  else error->all(FLERR,"fix_write_insurance_shear_history not defined for the chosen pairstyle");

  int dim;  
  NeighList *list = (NeighList *) pair->extract("list",dim);
  NeighList *listhistory = (NeighList *) pair->extract("listhistory",dim);
  
  //~ Find the number of shear quantities
  int numshearquants = 3;

  //~ pair/gran/shm/history has 4 shear quantities
  //~ pair/gran/CM/history has 5 shear quantities
  //~ pair/gran/HMD/history has 26 shear quantities (also CMD)
  if (force->pair_match("shm",0)) numshearquants++;
  if (force->pair_match("gran/CM/history",1)) numshearquants += 2;
  if (force->pair_match("HMD",0)) numshearquants += 23;   
  if (force->pair_match("CMD",0)) numshearquants += 23;  

  /*~ Adding a rolling resistance model causes the number of
    shear quantities to be increased by 15 [KH - 29 July 2014]*/
  int *rolling = (int *) pair->extract("rolling",dim);
  if (*rolling) numshearquants += 15;

  // Added for D_spin model [MO - 13 November 2014]
  int *D_spin = (int *) pair->extract("D_spin",dim);
  if (*D_spin) numshearquants += 20;

  /*~ Per-contact energy tracing causes the number of shear quantities
    to increase by 4 [KH - 6 March 2014]*/
  int *trace_energy = (int *) pair->extract("trace_energy",dim);
  if (*trace_energy) numshearquants += 4;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firstshear = listhistory->firstdouble;
 
  /*~ Create a dynamic array for storing information regarding the contact
    forces*/
  int nrows = 1;
  int ncols = numshearquants+2;
  double **cfdata = NULL;
  memory->create(cfdata,nrows,ncols,"FixWriteInsuranceShearHistory:cfdata");

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;

    allshear = firstshear[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (!(mask[j] & groupbit)) continue;

      /*~ Ensure that shear information is written without duplication 
	when ghost atoms are present*/
      if (j >= atom->nlocal && tag[i] > tag[j]) continue;

      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      //~ Ensure that the particles are actually touching
      if (rsq > (radius[i] + radius[j])*(radius[i] + radius[j])) continue;

      shear = &allshear[numshearquants*jj];

      if (nrows > 1) //~ Extend the cfdata array
	memory->grow(cfdata,nrows,ncols,"FixWriteInsuranceShearHistory:cfdata");

      cfdata[nrows-1][0] = tag[i];
      cfdata[nrows-1][1] = tag[j];

      for (int q = 0; q < numshearquants; q++)
	cfdata[nrows-1][q+2] = shear[q];

      nrows++;
    }
  }

  //~ Now write the shear history to a file
  char buffer[30];
  int strstep = sprintf(buffer,"%d",update->ntimestep);
  int n = strlen("Shear_History_.txt") + 1 + strstep;
  char *filename = new char[n];
  strcpy(filename,"Shear_History_");
  strcat(filename,buffer);
  strcat(filename,".txt");
  FILE *fname = fopen(filename,"w");

  for (int i = 0; i < (nrows-1); i++) {
    for (int j = 0; j < ncols; j++) {
      if (j < 2) fprintf(fname,"%i\t",static_cast< int >(cfdata[i][j]));
      else if (j < ncols-1) fprintf(fname,"%1.16e\t",cfdata[i][j]);
      else fprintf(fname,"%1.16e\n",cfdata[i][j]);
    }
  }
  fclose(fname);

  //~ Destroy the array at this stage to free up memory
  memory->destroy(cfdata);
}

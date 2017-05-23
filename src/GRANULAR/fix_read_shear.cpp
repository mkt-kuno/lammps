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
#include <stdio.h>
#include <string.h>
#include <fstream>
#include "fix_read_shear.h"
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

FixReadShear::FixReadShear(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  /*~ This fix takes only one non-standard argument: the name
    of the file containing the shear data [KH - 23 May 2013]*/
  
  if (narg < 4) error->all(FLERR,"Illegal fix read shear command");
  
  //~ Read in the filename as a string
  int n = strlen(arg[3]) + 1;
  char *filename = new char[n];
  strcpy(filename,arg[3]);

  //~ Find the number of shear quantities
  numshearquants = 3;

  //~ pair/gran/shm/history has 4 shear quantities
  if (force->pair_match("shm",0)) numshearquants++;
  if (force->pair_match("gran/CM/history",1)) numshearquants += 2;
  if (force->pair_match("HMD",0)) numshearquants += 23; //Increased [MO - 14 November 2014]  
  if (force->pair_match("CMD",0)) numshearquants += 23; //Added [MO - 18 November 2014]  
  /*~ Adding a rolling resistance model causes the number of
    shear quantities to be increased by 15. Gain access to
    arrays in pairstyle using pair->extract [KH - 6 February 2014]*/
  Pair *pair;

  if (force->pair_match("gran/hooke/history",1)) 
    pair = force->pair_match("gran/hooke/history",1);
  else if (force->pair_match("gran/hertz/history",1))
    pair = force->pair_match("gran/hertz/history",1);
  else if (force->pair_match("gran/shm/history",1))
    pair = force->pair_match("gran/shm/history",1);
  else if (force->pair_match("gran/CM/history",1))
    pair = force->pair_match("gran/CM/history",1);
  else if (force->pair_match("gran/HMD/history",1)) // Added [MO - 21 July 2014]
    pair = force->pair_match("gran/HMD/history",1);
  else if (force->pair_match("gran/CMD/history",1)) // Added [MO - 18 November 2014]
    pair = force->pair_match("gran/CMD/history",1);
  else error->all(FLERR,"fix_read_shear not defined for the chosen pairstyle");

  int dim;
  int *rolling = (int *) pair->extract("rolling",dim);
  if (*rolling) numshearquants += 15;

  // added for D_spin function [MO - 13 November 2014]
  int *D_spin = (int *) pair->extract("D_spin",dim);
  if (*D_spin) numshearquants += 20;

  /*~ Per-contact energy tracing causes the number of shear quantities
    to increase by 4 [KH - 6 March 2014]*/
  int *trace_energy = (int *) pair->extract("trace_energy",dim);
  if (*trace_energy) numshearquants += 4;

  //~ Allocate memory for the sheardata array
  nrows = 1;
  int ncols = numshearquants+2;

  sheardata = NULL;
  memory->create(sheardata,nrows,ncols,"FixReadShear:sheardata");

  //~ Read in the shear data line-by-line to this array
  std::ifstream inputfile (filename, std::ios::in);
  inputfile.precision(20);

  if (!inputfile)
    error->all(FLERR,"Input file not found by fix read shear");
  else 
    while(!inputfile.eof()) {
      double tagi,tagj,shear0,shear1,shear2,shear3,shear4;

      if (numshearquants == 3)
	inputfile >> tagi >> tagj >> shear0 >> shear1 >> shear2;
      else if (numshearquants == 4) inputfile >> tagi >> tagj >> shear0 >> shear1 >> shear2 >> shear3;
      else inputfile >> tagi >> tagj >> shear0 >> shear1 >> shear2 >> shear3 >> shear4;
      
      if (nrows > 1) //~ Extend the sheardata array
	memory->grow(sheardata,nrows,ncols,"FixReadShear:sheardata");

      sheardata[nrows-1][0] = tagi;
      sheardata[nrows-1][1] = tagj;
      sheardata[nrows-1][2] = shear0;
      sheardata[nrows-1][3] = shear1;
      sheardata[nrows-1][4] = shear2;
      if (numshearquants == 4) sheardata[nrows-1][5] = shear3;
      if (numshearquants == 5) {
	sheardata[nrows-1][5] = shear3;
	sheardata[nrows-1][6] = shear4;
      }
      nrows++;
    }
}

/* ---------------------------------------------------------------------- */

FixReadShear::~FixReadShear() {}

/* ---------------------------------------------------------------------- */

int FixReadShear::setmask() 
{
 int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixReadShear::setup_pre_force(int vflag)
{
  /*~ Allocate the shear data to contacting particles in the
    neighbour list. This needs to be done after the neighbour
    list has been built, but before the contact forces are
    computed for the first time.*/

  int i,j,ii,jj,inum,jnum;
  double delx,dely,delz,rsq;
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
  else if (force->pair_match("gran/HMD/history",1))  // Added [MO - 21 July 2014]
    pair = force->pair_match("gran/HMD/history",1);
  else if (force->pair_match("gran/CMD/history",1))  // Added [MO - 18 November 2014]
    pair = force->pair_match("gran/CMD/history",1);
  else error->all(FLERR,"fix read shear not defined for the chosen pairstyle");

  int dim;
  NeighList *list = (NeighList *) pair->extract("list",dim);
  NeighList *listgranhistory = (NeighList *) pair->extract("listgranhistory",dim);
  
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firstshear = listgranhistory->firstdouble;
 
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

      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      //~ Ensure that the particles are actually touching
      if (rsq > (radius[i] + radius[j])*(radius[i] + radius[j])) continue;

      shear = &allshear[numshearquants*jj];

      /*~ Now find the shear information for these particles in the
	sheardata array*/
      
      for (int q = 0; q < (nrows-1); q++) {
	if (static_cast< int >(sheardata[q][0]) == tag[i] && static_cast< int >(sheardata[q][1]) == tag[j]) {
	  for (int r = 0; r < numshearquants; r++)
	    shear[r] = sheardata[q][r+2];
	  
	  break;
	} else if (static_cast< int >(sheardata[q][0]) == tag[j] && static_cast< int >(sheardata[q][1]) == tag[i]) {
	  for (int r = 0; r < 3; r++)
	    shear[r] = -1.0*sheardata[q][r+2];
	  
	  //~ shear[3] >= 0.0 always
	  if (numshearquants == 4) shear[3] = sheardata[q][5];
	  if (numshearquants == 5) {
	    shear[3] = sheardata[q][5];
	    shear[4] = sheardata[q][6];
	  }
	  break;
	}
      }  
    }
  }
 
  //~ Destroy the sheardata array to free up memory
  memory->destroy(sheardata);
}

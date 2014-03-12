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

#include "mpi.h"
#include "string.h"
#include "stdio.h"
#include "fix_shear_history.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "force.h"
#include "pair.h"
#include "update.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "stdlib.h"  //added in GM

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixShearHistory::FixShearHistory(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  restart_peratom = 1;
  create_attribute = 1;

  /*~ Read in the number of shear quantities if available 
    [KH - 21 November 2012]*/
  if (narg == 4) num_quants = force->inumeric(FLERR,arg[3]); // added in GM
  else num_quants = 3; //~ Assume the default value of 3 instead

  // perform initial allocation of atom-based arrays
  // register with atom class

  npartner = NULL;
  partner = NULL;

  //~ Number of shear quantities can vary [KH - 9 January 2014]
  shearpartner3 = NULL;
  shearpartner4 = NULL;
  shearpartner7 = NULL;
  shearpartner8 = NULL;
  shearpartner16 = NULL;
  shearpartner17 = NULL;
  shearpartner20 = NULL;
  shearpartner21 = NULL;

  grow_arrays(atom->nmax);
  atom->add_callback(0);
  atom->add_callback(1);

  ipage = NULL;

  //~ Also the case here [KH - 9 January 2014]
  dpage3 = NULL;
  dpage4 = NULL;
  dpage7 = NULL;
  dpage8 = NULL;
  dpage16 = NULL;
  dpage17 = NULL;
  dpage20 = NULL;
  dpage21 = NULL;

  pgsize = oneatom = 0;

  // initialize npartner to 0 so neighbor list creation is OK the 1st time

  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) npartner[i] = 0;
  maxtouch = 0;
}

/* ---------------------------------------------------------------------- */

FixShearHistory::~FixShearHistory()
{
  // unregister this fix so atom class doesn't invoke it any more

  atom->delete_callback(id,0);
  atom->delete_callback(id,1);

  // delete locally stored arrays

  memory->destroy(npartner);
  memory->sfree(partner);

  memory->sfree(shearpartner3); //~ [KH - 9 January 2014]
  memory->sfree(shearpartner4);
  memory->sfree(shearpartner7);
  memory->sfree(shearpartner8);
  memory->sfree(shearpartner16);
  memory->sfree(shearpartner17);
  memory->sfree(shearpartner20);
  memory->sfree(shearpartner21);

  delete [] ipage;

  delete [] dpage3; //~ [KH - 9 January 2014]
  delete [] dpage4;
  delete [] dpage7;
  delete [] dpage8;
  delete [] dpage16;
  delete [] dpage17;
  delete [] dpage20;
  delete [] dpage21;
}

/* ---------------------------------------------------------------------- */

int FixShearHistory::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= MIN_PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixShearHistory::init()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,
               "Pair style granular with history requires atoms have IDs");

  int dim;
  computeflag = (int *) pair->extract("computeflag",dim);

  allocate_pages();
}

/* ----------------------------------------------------------------------
  create pages if first time or if neighbor pgsize/oneatom has changed
  note that latter could cause shear history info to be discarded
------------------------------------------------------------------------- */

void FixShearHistory::allocate_pages()
{
  int create = 0;
  if (ipage == NULL) create = 1;
  if (pgsize != neighbor->pgsize) create = 1;
  if (oneatom != neighbor->oneatom) create = 1;

  if (create) {
    delete [] ipage;
    pgsize = neighbor->pgsize;
    oneatom = neighbor->oneatom;
    int nmypage = comm->nthreads;
    ipage = new MyPage<tagint>[nmypage];

    for (int i = 0; i < nmypage; i++)
      ipage[i].init(oneatom,pgsize);

    /*~ Use a switch-case structure to pick the correct
      dpage pointer depending on the value of num_quants
      [KH - 9 January 2014]*/
    switch (num_quants) {
    case 4: //~ 4 is the most likely num_quants
      delete [] dpage4;
      dpage4 = new MyPage<double[4]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage4[i].init(oneatom,pgsize);
      break;
    case 3: //~ 3 is next most likely
      delete [] dpage3;
      dpage3 = new MyPage<double[3]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage3[i].init(oneatom,pgsize);
      break;
    case 16:
      delete [] dpage16;
      dpage16 = new MyPage<double[16]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage16[i].init(oneatom,pgsize);
      break;
    case 17:
      delete [] dpage17;
      dpage17 = new MyPage<double[17]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage17[i].init(oneatom,pgsize);
      break;
    case 8:
      delete [] dpage8;
      dpage8 = new MyPage<double[8]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage8[i].init(oneatom,pgsize);
      break;
    case 7:
      delete [] dpage7;
      dpage7 = new MyPage<double[7]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage7[i].init(oneatom,pgsize);
      break;
    case 20:
      delete [] dpage20;
      dpage20 = new MyPage<double[20]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage20[i].init(oneatom,pgsize);
      break;
    case 21:
      delete [] dpage21;
      dpage21 = new MyPage<double[21]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage21[i].init(oneatom,pgsize);
      break;
    default:
      //~ If no cases matched, there is a problem
      error->all(FLERR,"Incorrect number of shear quantities");
    }
  }
}

/* ----------------------------------------------------------------------
   called by setup of run or minimize
   called by write_restart as input script command
   only invoke pre_exchange() if neigh list stores more current history info
     than npartner/partner arrays in this fix
   that will only be case if pair->compute() has been invoked since
     update of npartner/npartner
   this logic avoids 2 problems:
     run 100; write_restart; run 100
       setup_pre_exchange is called twice (by write_restart and 2nd run setup)
       w/out a neighbor list being created in between
     read_restart; run 100
       setup_pre_exchange called by run setup whacks restart shear history info
------------------------------------------------------------------------- */

void FixShearHistory::setup_pre_exchange()
{
  if (*computeflag) pre_exchange();
  *computeflag = 0;
}

/* ----------------------------------------------------------------------
   copy shear partner info from neighbor lists to atom arrays
   so can be migrated or stored with atoms
------------------------------------------------------------------------- */

void FixShearHistory::pre_exchange()
{
  int i,j,ii,jj,m,n,inum,jnum;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *touch,**firsttouch;
  double *shear,*allshear,**firstshear;

  // nlocal may include atoms added since last neigh build

  int nlocal = atom->nlocal;

  // zero npartner for all current atoms
  // clear 2 page data structures

  for (i = 0; i < nlocal; i++) npartner[i] = 0;

  ipage->reset();

  //~ Use a switch-case structure [KH - 9 January 2014]
  switch (num_quants) {
  case 4: //~ 4 is the most likely num_quants
    dpage4->reset();
    break;
  case 3: //~ 3 is next most likely
    dpage3->reset();
    break;
  case 16:
    dpage16->reset();
    break;
  case 17:
    dpage17->reset();
    break;
  case 8:
    dpage8->reset();
    break;
  case 7:
    dpage7->reset();
    break;
  case 20:
    dpage20->reset();
    break;
  case 21:
    dpage21->reset();
    break;
  }

  // 1st loop over neighbor list
  // calculate npartner for each owned atom
  // nlocal_neigh = nlocal when neigh list was built, may be smaller than nlocal

  tagint *tag = atom->tag;
  NeighList *list = pair->list;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firsttouch = list->listgranhistory->firstneigh;
  firstshear = list->listgranhistory->firstdouble;

  int nlocal_neigh = 0;
  if (inum) nlocal_neigh = ilist[inum-1] + 1;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    touch = firsttouch[i];

    for (jj = 0; jj < jnum; jj++) {
      if (touch[jj]) {
        npartner[i]++;
        j = jlist[jj];
        j &= NEIGHMASK;
        if (j < nlocal_neigh) npartner[j]++;
      }
    }
  }

  // get page chunks to store atom IDs and shear history for my atoms
  /*~ Use a switch-case structure at the highest level, rather than
    inside the for loop, for maximum efficiency [KH - 9 January 2014]*/
  switch (num_quants) {
  case 4: //~ 4 is the most likely num_quants
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner4[i] = dpage4->get(n);
      if (partner[i] == NULL || shearpartner4[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 3: //~ 3 is next most likely
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner3[i] = dpage3->get(n);
      if (partner[i] == NULL || shearpartner3[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 16:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner16[i] = dpage16->get(n);
      if (partner[i] == NULL || shearpartner16[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 17:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner17[i] = dpage17->get(n);
      if (partner[i] == NULL || shearpartner17[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 8:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner8[i] = dpage8->get(n);
      if (partner[i] == NULL || shearpartner8[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 7:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner7[i] = dpage7->get(n);
      if (partner[i] == NULL || shearpartner7[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 20:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner20[i] = dpage20->get(n);
      if (partner[i] == NULL || shearpartner20[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 21:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner21[i] = dpage21->get(n);
      if (partner[i] == NULL || shearpartner21[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  }

  // 2nd loop over neighbor list
  // store atom IDs and shear history for my atoms
  // re-zero npartner to use as counter for all my atoms

  for (i = 0; i < nlocal; i++) npartner[i] = 0;

  /*~ As before, use a switch-case structure at the highest level
    [KH - 9 January 2014]*/
  switch (num_quants) {
  case 4: //~ 4 is the most likely num_quants
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[4*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 4; kk++)
	    shearpartner4[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 4; kk++) {
	      /*~ Modified this when the rolling resistance shear
		parameters were stored in the shear array [KH - 5
		November 2013]*/
	      shearpartner4[j][m][kk] = -shear[kk];
	    }
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 3: //~ 3 is next most likely
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[3*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 3; kk++)
	    shearpartner3[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 3; kk++)
	      shearpartner3[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 16:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[16*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 16; kk++)
	    shearpartner16[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 16; kk++)
	      shearpartner16[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 17:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[17*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 17; kk++)
	    shearpartner17[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 17; kk++)
	      shearpartner17[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 8:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[8*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 8; kk++)
	    shearpartner8[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];

	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 4; kk++)
	      shearpartner8[j][m][kk] = -shear[kk];

	    for (int kk = 4; kk < 8; kk++)
	      shearpartner8[j][m][kk] = shear[kk]; //~ Positive
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 7:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[7*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 7; kk++)
	    shearpartner7[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];

	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 3; kk++)
	      shearpartner7[j][m][kk] = -shear[kk];

	    for (int kk = 3; kk < 7; kk++)
	      shearpartner7[j][m][kk] = shear[kk]; //~ Positive
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 20:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[20*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 20; kk++)
	    shearpartner20[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];

	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 3; kk++)
	      shearpartner20[j][m][kk] = -shear[kk];

	    for (int kk = 3; kk < 7; kk++)
	      shearpartner20[j][m][kk] = shear[kk]; //~ Positive

	    for (int kk = 7; kk < 20; kk++)
	      shearpartner20[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 21:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[21*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 21; kk++)
	    shearpartner21[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];

	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 4; kk++)
	      shearpartner21[j][m][kk] = -shear[kk];

	    for (int kk = 4; kk < 8; kk++)
	      shearpartner21[j][m][kk] = shear[kk]; //~ Positive

	    for (int kk = 8; kk < 21; kk++)
	      shearpartner21[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  }

  // set maxtouch = max # of partners of any owned atom
  // bump up comm->maxexchange_fix if necessary
  //~ Hard-coded 4 replaced with (num_quants+1) [KH - 9 January 2014]
  maxtouch = 0;
  for (i = 0; i < nlocal; i++) maxtouch = MAX(maxtouch,npartner[i]);
  comm->maxexchange_fix = MAX(comm->maxexchange_fix,(num_quants+1)*maxtouch+1);
}

/* ---------------------------------------------------------------------- */

void FixShearHistory::min_setup_pre_exchange()
{
  if (*computeflag) pre_exchange();
  *computeflag = 0;
}

/* ---------------------------------------------------------------------- */

void FixShearHistory::min_pre_exchange()
{
  pre_exchange();
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixShearHistory::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = nmax * sizeof(int);
  bytes += nmax * sizeof(int *);
  bytes += nmax * sizeof(double *);

  int nmypage = comm->nthreads;
  for (int i = 0; i < nmypage; i++) {
    bytes += ipage[i].size();

    /*~ Added a switch-case structure to prevent seg faults
      [KH - 9 January 2014]*/
    switch (num_quants) {
    case 4: //~ 4 is the most likely num_quants
      bytes += dpage4[i].size();
      break;
    case 3: //~ 3 is next most likely
      bytes += dpage3[i].size();
      break;
    case 16:
      bytes += dpage16[i].size();
      break;
    case 17:
      bytes += dpage17[i].size();
      break;
    case 8:
      bytes += dpage8[i].size();
      break;
    case 7:
      bytes += dpage7[i].size();
      break;
    case 20:
      bytes += dpage20[i].size();
      break;
    case 21:
      bytes += dpage21[i].size();
      break;
    }
  }

  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixShearHistory::grow_arrays(int nmax)
{
  memory->grow(npartner,nmax,"shear_history:npartner");
  partner = (tagint **) memory->srealloc(partner,nmax*sizeof(tagint *),
                                         "shear_history:partner");
  
  //~  Added more typedefs [KH - 9 January 2014]
  typedef double (*sptype3)[3];
  typedef double (*sptype4)[4];
  typedef double (*sptype7)[7];
  typedef double (*sptype8)[8];
  typedef double (*sptype16)[16];
  typedef double (*sptype17)[17];
  typedef double (*sptype20)[20];
  typedef double (*sptype21)[21];
  
  //~ Use a switch-case structure [KH - 9 January 2014]
  switch (num_quants) {
  case 4: //~ 4 is the most likely num_quants
    shearpartner4 = (sptype4 *) 
      memory->srealloc(shearpartner4,nmax*sizeof(sptype4),
		       "shear_history:shearpartner4");
    break;
  case 3: //~ 3 is next most likely
    shearpartner3 = (sptype3 *) 
      memory->srealloc(shearpartner3,nmax*sizeof(sptype3),
		       "shear_history:shearpartner3");
    break;
  case 16:
    shearpartner16 = (sptype16 *) 
      memory->srealloc(shearpartner16,nmax*sizeof(sptype16),
		       "shear_history:shearpartner16");
    break;
  case 17:
    shearpartner17 = (sptype17 *) 
      memory->srealloc(shearpartner17,nmax*sizeof(sptype17),
		       "shear_history:shearpartner17");
    break;
  case 8:
    shearpartner8 = (sptype8 *) 
      memory->srealloc(shearpartner8,nmax*sizeof(sptype8),
		       "shear_history:shearpartner8");
    break;
  case 7:
    shearpartner7 = (sptype7 *) 
      memory->srealloc(shearpartner7,nmax*sizeof(sptype7),
		       "shear_history:shearpartner7");
    break;
  case 20:
    shearpartner20 = (sptype20 *) 
      memory->srealloc(shearpartner20,nmax*sizeof(sptype20),
		       "shear_history:shearpartner20");
    break;
  case 21:
    shearpartner21 = (sptype21 *) 
      memory->srealloc(shearpartner21,nmax*sizeof(sptype21),
		       "shear_history:shearpartner21");
    break;
  }
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixShearHistory::copy_arrays(int i, int j, int delflag)
{
  // just copy pointers for partner and shearpartner
  // b/c can't overwrite chunk allocation inside ipage,dpage
  // incoming atoms in unpack_exchange just grab new chunks
  // so are orphaning chunks for migrating atoms
  // OK, b/c will reset ipage,dpage on next reneighboring

  npartner[j] = npartner[i];
  partner[j] = partner[i];

  //~ Use a switch-case structure [KH - 9 January 2014]
  switch (num_quants) {
  case 4: //~ 4 is the most likely num_quants
    shearpartner4[j] = shearpartner4[i];
    break;
  case 3:
    shearpartner3[j] = shearpartner3[i];
    break;
  case 16:
    shearpartner16[j] = shearpartner16[i];
    break;
  case 17:
    shearpartner17[j] = shearpartner17[i];
    break;
  case 8:
    shearpartner8[j] = shearpartner8[i];
    break;
  case 7:
    shearpartner7[j] = shearpartner7[i];
    break;
  case 20:
    shearpartner20[j] = shearpartner20[i];
    break;
  case 21:
    shearpartner21[j] = shearpartner21[i];
    break;
  }
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixShearHistory::set_arrays(int i)
{
  npartner[i] = 0;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixShearHistory::pack_exchange(int i, double *buf)
{
  // NOTE: how do I know comm buf is big enough if extreme # of touching neighs
  // Comm::BUFEXTRA may need to be increased

  int m = 0;
  buf[m++] = npartner[i];

  //~ Use a switch-case structure [KH - 9 January 2014]
  switch (num_quants) {
  case 4: //~ 4 is the most likely num_quants
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 4; k++)
	buf[m++] = shearpartner4[i][n][k];
    }
    break;
  case 3:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 3; k++)
	buf[m++] = shearpartner3[i][n][k];
    }
    break;
  case 16:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 16; k++)
	buf[m++] = shearpartner16[i][n][k];
    }
    break;
  case 17:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 17; k++)
	buf[m++] = shearpartner17[i][n][k];
    }
    break;
  case 8:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 8; k++)
	buf[m++] = shearpartner8[i][n][k];
    }
    break;
  case 7:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 7; k++)
	buf[m++] = shearpartner7[i][n][k];
    }
    break;
  case 20:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 20; k++)
	buf[m++] = shearpartner20[i][n][k];
    }
    break;
  case 21:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 21; k++)
	buf[m++] = shearpartner21[i][n][k];
    }
    break;
  }
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixShearHistory::unpack_exchange(int nlocal, double *buf)
{
  // allocate new chunks from ipage,dpage for incoming values

  int m = 0;
  npartner[nlocal] = static_cast<int> (buf[m++]);
  maxtouch = MAX(maxtouch,npartner[nlocal]);
  partner[nlocal] = ipage->get(npartner[nlocal]);

  //~ Use a switch-case structure [KH - 9 January 2014]
  switch (num_quants) {
  case 4: //~ 4 is the most likely num_quants
    shearpartner4[nlocal] = dpage4->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 4; k++)
	shearpartner4[nlocal][n][k] = buf[m++];
    }
    break;
  case 3:
    shearpartner3[nlocal] = dpage3->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 3; k++)
	shearpartner3[nlocal][n][k] = buf[m++];
    }
    break;
  case 16:
    shearpartner16[nlocal] = dpage16->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 16; k++)
	shearpartner16[nlocal][n][k] = buf[m++];
    }
    break;
  case 17:
    shearpartner17[nlocal] = dpage17->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 17; k++)
	shearpartner17[nlocal][n][k] = buf[m++];
    }
    break;
  case 8:
    shearpartner8[nlocal] = dpage8->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 8; k++)
	shearpartner8[nlocal][n][k] = buf[m++];
    }
    break;
  case 7:
    shearpartner7[nlocal] = dpage7->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 7; k++)
	shearpartner7[nlocal][n][k] = buf[m++];
    }
    break;
  case 20:
    shearpartner20[nlocal] = dpage20->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 20; k++)
	shearpartner20[nlocal][n][k] = buf[m++];
    }
    break;
  case 21:
    shearpartner21[nlocal] = dpage21->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 21; k++)
	shearpartner21[nlocal][n][k] = buf[m++];
    }
    break;
  }
  return m;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixShearHistory::pack_restart(int i, double *buf)
{
  int m = 0;
  buf[m++] = (num_quants+1)*npartner[i] + 2; // changed from 4 to num_quants+1 , modified GM
  buf[m++] = npartner[i];

  //~ Use a switch-case structure [KH - 9 January 2014]
  switch (num_quants) {
  case 4: //~ 4 is the most likely num_quants
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 4; k++)
	buf[m++] = shearpartner4[i][n][k];
    }
    break;
  case 3:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 3; k++)
	buf[m++] = shearpartner3[i][n][k];
    }
    break;
  case 16:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 16; k++)
	buf[m++] = shearpartner16[i][n][k];
    }
    break;
  case 17:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 17; k++)
	buf[m++] = shearpartner17[i][n][k];
    }
    break;
  case 8:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 8; k++)
	buf[m++] = shearpartner8[i][n][k];
    }
    break;
  case 7:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 7; k++)
	buf[m++] = shearpartner7[i][n][k];
    }
    break;
  case 20:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 20; k++)
	buf[m++] = shearpartner20[i][n][k];
    }
    break;
  case 21:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 21; k++)
	buf[m++] = shearpartner21[i][n][k];
    }
    break;
  }
  return m;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixShearHistory::unpack_restart(int nlocal, int nth)
{
  // ipage = NULL if being called from granular pair style init()

  if (ipage == NULL) allocate_pages();

  // skip to Nth set of extra values

  double **extra = atom->extra;

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  // allocate new chunks from ipage,dpage for incoming values

  npartner[nlocal] = static_cast<int> (extra[nlocal][m++]);
  maxtouch = MAX(maxtouch,npartner[nlocal]);
  partner[nlocal] = ipage->get(npartner[nlocal]);

  //~ Use a switch-case structure [KH - 9 January 2014]
  switch (num_quants) {
  case 4: //~ 4 is the most likely num_quants
    shearpartner4[nlocal] = dpage4->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 4; k++)
	shearpartner4[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 3:
    shearpartner3[nlocal] = dpage3->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 3; k++)
	shearpartner3[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 16:
    shearpartner16[nlocal] = dpage16->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 16; k++)
	shearpartner16[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 17:
    shearpartner17[nlocal] = dpage17->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 17; k++)
	shearpartner17[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 8:
    shearpartner8[nlocal] = dpage8->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 8; k++)
	shearpartner8[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 7:
    shearpartner7[nlocal] = dpage7->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 7; k++)
	shearpartner7[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 20:
    shearpartner20[nlocal] = dpage20->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 20; k++)
	shearpartner20[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 21:
    shearpartner21[nlocal] = dpage21->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 21; k++)
	shearpartner21[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  }
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixShearHistory::maxsize_restart()
{
  // maxtouch_all = max # of touching partners across all procs

  int maxtouch_all;
  MPI_Allreduce(&maxtouch,&maxtouch_all,1,MPI_INT,MPI_MAX,world);
  return (num_quants+1)*maxtouch_all + 2; //~ [KH - 8 January 2014]
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixShearHistory::size_restart(int nlocal)
{
  return (num_quants+1)*npartner[nlocal] + 2; // changed from 4 to num_quants+1 , modified GM
}

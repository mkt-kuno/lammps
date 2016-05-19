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
#include <string.h>
#include <stdio.h>
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
#include <stdlib.h>  //added in GM

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
  /*~~ shearpartner5 is added [MO - 4 June 2014] ~~*/ 
  shearpartner3 = NULL;
  shearpartner4 = NULL;
  shearpartner5 = NULL;
  shearpartner7 = NULL;
  shearpartner8 = NULL;
  shearpartner9 = NULL;
  shearpartner18 = NULL;
  shearpartner19 = NULL;
  shearpartner22 = NULL;
  shearpartner23 = NULL;
  // added [MO - 14 November 2014]
  shearpartner24 = NULL;
  shearpartner25 = NULL;
  shearpartner26 = NULL;
  shearpartner28 = NULL;
  shearpartner29 = NULL;
  shearpartner30 = NULL;
  shearpartner46 = NULL;
  shearpartner50 = NULL;

  grow_arrays(atom->nmax);
  atom->add_callback(0);
  atom->add_callback(1);

  ipage = NULL;

  //~ Also the case here [KH - 9 January 2014]
  /*~~ dpage5 is added [MO - 4 June 2014] ~~*/ 
  dpage3 = NULL;
  dpage4 = NULL;
  dpage5 = NULL;
  dpage7 = NULL;
  dpage8 = NULL;
  dpage9 = NULL;
  dpage18 = NULL;
  dpage19 = NULL;
  dpage22 = NULL;
  dpage23 = NULL;
  // added [MO - 14 November 2014]
  dpage24 = NULL;
  dpage25 = NULL;
  dpage26 = NULL;
  dpage28 = NULL;
  dpage29 = NULL;
  dpage30 = NULL;
  dpage46 = NULL;
  dpage50 = NULL;

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
  memory->sfree(shearpartner5); /*~~ 5 is added [MO - 4 June 2014] ~~*/ 
  memory->sfree(shearpartner7);
  memory->sfree(shearpartner8);
  memory->sfree(shearpartner9);
  memory->sfree(shearpartner18);
  memory->sfree(shearpartner19);
  memory->sfree(shearpartner22);
  memory->sfree(shearpartner23);
  // added [MO - 14 November 2014]
  memory->sfree(shearpartner24);
  memory->sfree(shearpartner25);
  memory->sfree(shearpartner26);
  memory->sfree(shearpartner28);
  memory->sfree(shearpartner29);
  memory->sfree(shearpartner30);
  memory->sfree(shearpartner46);
  memory->sfree(shearpartner50);

  delete [] ipage;

  delete [] dpage3; //~ [KH - 9 January 2014]
  delete [] dpage4;
  delete [] dpage5; /*~~ 5 is added [MO - 4 June 2014] ~~*/ 
  delete [] dpage7;
  delete [] dpage8;
  delete [] dpage9; /*~~ 9 is added [MO - 5 June 2014] ~~*/ 
  delete [] dpage18;
  delete [] dpage19;
  delete [] dpage22;
  delete [] dpage23;
  // 8 lines are added [MO - 14 November 2014]
  delete [] dpage24;
  delete [] dpage25;
  delete [] dpage26;
  delete [] dpage28;
  delete [] dpage29;
  delete [] dpage30;
  delete [] dpage46;
  delete [] dpage50;
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
    case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
      delete [] dpage5;
      dpage5 = new MyPage<double[5]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage5[i].init(oneatom,pgsize);
      break;
    case 18:
      delete [] dpage18;
      dpage18 = new MyPage<double[18]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage18[i].init(oneatom,pgsize);
      break;
    case 19:
      delete [] dpage19;
      dpage19 = new MyPage<double[19]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage19[i].init(oneatom,pgsize);
      break;
    case 8:
      delete [] dpage8;
      dpage8 = new MyPage<double[8]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage8[i].init(oneatom,pgsize);
      break;
    case 9: /*~~ 9 is added for the CM model [MO - 5 June 2014] ~~*/
      delete [] dpage9;
      dpage9 = new MyPage<double[9]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage9[i].init(oneatom,pgsize);
      break;
    case 7:
      delete [] dpage7;
      dpage7 = new MyPage<double[7]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage7[i].init(oneatom,pgsize);
      break;
    case 22:
      delete [] dpage22;
      dpage22 = new MyPage<double[22]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage22[i].init(oneatom,pgsize);
      break;
    case 23:
      delete [] dpage23;
      dpage23 = new MyPage<double[23]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage23[i].init(oneatom,pgsize);
      break;
      // 6 cases are added [MO - 14 November 2014]
    case 24: 
      delete [] dpage24;
      dpage24 = new MyPage<double[24]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage24[i].init(oneatom,pgsize);
      break;
    case 25:
      delete [] dpage25;
      dpage25 = new MyPage<double[25]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage25[i].init(oneatom,pgsize);
      break;
    case 26:
      delete [] dpage26;
      dpage26 = new MyPage<double[26]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage26[i].init(oneatom,pgsize);
      break;
    case 46: 
      delete [] dpage46;
      dpage46 = new MyPage<double[46]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage46[i].init(oneatom,pgsize);
      break;
    case 28:
      delete [] dpage28;
      dpage28 = new MyPage<double[28]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage28[i].init(oneatom,pgsize);
      break;
    case 29: 
      delete [] dpage29;
      dpage29 = new MyPage<double[29]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage29[i].init(oneatom,pgsize);
      break;
    case 30: 
      delete [] dpage30;
      dpage30 = new MyPage<double[30]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage30[i].init(oneatom,pgsize);
      break;
    case 50:
      delete [] dpage50;
      dpage50 = new MyPage<double[50]>[nmypage];
      for (int i = 0; i < nmypage; i++)
	dpage50[i].init(oneatom,pgsize);
      break;
    default:
      //~ If no cases matched, there is a problem
      error->all(FLERR,"Incorrect number of shear quantities");
    }
  }
}

/* ----------------------------------------------------------------------
   copy shear partner info from neighbor lists to atom arrays
   should be called whenever neighbor list stores current history info
     and need to have atoms store the info
   e.g. so atoms can migrate to new procs or between runs
     when atoms may be added or deleted (neighbor list becomes out-of-date)
   the next granular neigh list build will put this info back into neigh list
   called during run before atom exchanges
   called at end of run via post_run()
   do not call during setup of run (setup_pre_exchange)
     b/c there is no guarantee of a current neigh list (even on continued run)
   if run command does a 2nd run with pre = no, then no neigh list
     will be built, but old neigh list will still have the info
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
  case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
    dpage5->reset();
    break;
  case 18:
    dpage18->reset();
    break;
  case 19:
    dpage19->reset();
  case 8:
    dpage8->reset();
    break;
  case 9:
    dpage9->reset();
    break;
  case 7:
    dpage7->reset();
    break;
  case 22:
    dpage22->reset();
    break;
  case 23:
    dpage23->reset();
    break;
  // Added [MO - 14 November 2014]
  case 24:
    dpage24->reset();
    break;
  case 25:
    dpage25->reset();
  case 26:
    dpage26->reset();
    break;
  case 46:
    dpage46->reset();
    break;
  case 28:
    dpage28->reset();
    break;
  case 29:
    dpage29->reset();
    break;
  case 30:
    dpage30->reset();
    break;
  case 50:
    dpage50->reset();
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
   case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner5[i] = dpage5->get(n);
      if (partner[i] == NULL || shearpartner5[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 18:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner18[i] = dpage18->get(n);
      if (partner[i] == NULL || shearpartner18[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 19:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner19[i] = dpage19->get(n);
      if (partner[i] == NULL || shearpartner19[i] == NULL)
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
  case 9:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner9[i] = dpage9->get(n);
      if (partner[i] == NULL || shearpartner9[i] == NULL)
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
  case 22:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner22[i] = dpage22->get(n);
      if (partner[i] == NULL || shearpartner22[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 23:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner23[i] = dpage23->get(n);
      if (partner[i] == NULL || shearpartner23[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  // Added [MO - 14 November 2014]
  case 24:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner24[i] = dpage24->get(n);
      if (partner[i] == NULL || shearpartner24[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 25:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner25[i] = dpage25->get(n);
      if (partner[i] == NULL || shearpartner25[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 26:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner26[i] = dpage26->get(n);
      if (partner[i] == NULL || shearpartner26[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 46:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner46[i] = dpage46->get(n);
      if (partner[i] == NULL || shearpartner46[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 28:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner28[i] = dpage28->get(n);
      if (partner[i] == NULL || shearpartner28[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 29:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner29[i] = dpage29->get(n);
      if (partner[i] == NULL || shearpartner29[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 30:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner30[i] = dpage30->get(n);
      if (partner[i] == NULL || shearpartner30[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }
    break;
  case 50:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      n = npartner[i];
      partner[i] = ipage->get(n);
      shearpartner50[i] = dpage50->get(n);
      if (partner[i] == NULL || shearpartner50[i] == NULL)
	error->one(FLERR,"Shear history overflow, boost neigh_modify one");
    }  
    //
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
  case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[5*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 5; kk++)
	    shearpartner5[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 5; kk++) 
	      shearpartner5[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 18:  // hertz(3) + rolling(15)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[18*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 18; kk++)
	    shearpartner18[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 18; kk++)
	      shearpartner18[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 19:  // shm(4) + rolling(15)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[19*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 19; kk++)
	    shearpartner19[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 19; kk++)
	      shearpartner19[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 8:  // shm(4) + energy(4)
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
  case 9: // CM(5) + energy(4)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[9*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 9; kk++)
	    shearpartner9[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];

	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 5; kk++)
	      shearpartner9[j][m][kk] = -shear[kk];

	    for (int kk = 5; kk < 9; kk++)
	      shearpartner9[j][m][kk] = shear[kk]; //~ Positive
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 22:
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[22*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 22; kk++)
	    shearpartner22[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];

	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 3; kk++)
	      shearpartner22[j][m][kk] = -shear[kk];

	    for (int kk = 3; kk < 7; kk++)
	      shearpartner22[j][m][kk] = shear[kk]; //~ Positive

	    for (int kk = 7; kk < 22; kk++)
	      shearpartner22[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 23:  // shm(4) + energy(4) + rolling(15)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];
      
      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[23*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 23; kk++)
	    shearpartner23[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    
	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 4; kk++)
	      shearpartner23[j][m][kk] = -shear[kk];
	    
	    for (int kk = 4; kk < 8; kk++)
	      shearpartner23[j][m][kk] = shear[kk]; //~ Positive
	    
	    for (int kk = 8; kk < 23; kk++)
	      shearpartner23[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
    // Added [MO - 14 November 2014]
  case 24: // shm(4) + D_spin(20)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];
      
      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[24*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 24; kk++)
	    shearpartner24[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 24; kk++)
	      shearpartner24[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 25: // CM(5) + D_spin(20)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];
      
      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[25*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 25; kk++)
	    shearpartner25[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 25; kk++)
	      shearpartner25[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
   case 26: // HMD(26)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[26*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 26; kk++)
	    shearpartner26[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 26; kk++)
	      shearpartner26[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;    
  case 46: // HMD(26) + D_spin(20)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];

      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[46*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 46; kk++)
	    shearpartner46[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    for (int kk = 0; kk < 46; kk++)
	      shearpartner46[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 28: // shm(4) + energy(4) + D_spim(20)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];
      
      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[28*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 28; kk++)
	    shearpartner28[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    
	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 4; kk++)
	      shearpartner28[j][m][kk] = -shear[kk];
	    
	    for (int kk = 4; kk < 8; kk++)
	      shearpartner28[j][m][kk] = shear[kk]; //~ Positive
	    
	    for (int kk = 8; kk < 28; kk++)
	      shearpartner28[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 29: // CM(5) + energy(4) + D_spim(20)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];
      
      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[29*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 29; kk++)
	    shearpartner29[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    
	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 5; kk++)
	      shearpartner29[j][m][kk] = -shear[kk];
	    
	    for (int kk = 5; kk < 9; kk++)
	      shearpartner29[j][m][kk] = shear[kk]; //~ Positive
	    
	    for (int kk = 9; kk < 29; kk++)
	      shearpartner29[j][m][kk] = -shear[kk];
	    npartner[j]++;
	  }
	}
      }
    }
    break;
  case 30: // HMD(26) + energy(4)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];
      
      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[30*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 30; kk++)
	    shearpartner30[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    
	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 26; kk++)
	      shearpartner30[j][m][kk] = -shear[kk];
	    
	    for (int kk = 26; kk < 30; kk++)
	      shearpartner30[j][m][kk] = shear[kk]; //~ Positive
	     npartner[j]++;
	  }
	}
      }
    }
    break;
  case 50: // HMD(26) + energy(4) + D_spim(20)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jlist = firstneigh[i];
      allshear = firstshear[i];
      jnum = numneigh[i];
      touch = firsttouch[i];
      
      for (jj = 0; jj < jnum; jj++) {
	if (touch[jj]) {
	  shear = &allshear[50*jj];
	  j = jlist[jj];
	  j &= NEIGHMASK;
	  m = npartner[i];
	  partner[i][m] = tag[j];
	  for (int kk = 0; kk < 50; kk++)
	    shearpartner50[i][m][kk] = shear[kk];
	  npartner[i]++;
	  if (j < nlocal_neigh) {
	    m = npartner[j];
	    partner[j][m] = tag[i];
	    
	    /*~ This was modified to prevent energy terms
	      becoming negative which is not sensible
	      [KH - 11 March 2014]*/
	    for (int kk = 0; kk < 26; kk++)
	      shearpartner50[j][m][kk] = -shear[kk];
	    
	    for (int kk = 26; kk < 30; kk++)
	      shearpartner50[j][m][kk] = shear[kk]; //~ Positive
	    
	    for (int kk = 30; kk < 50; kk++)
	      shearpartner50[j][m][kk] = -shear[kk];
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

void FixShearHistory::min_pre_exchange()
{
  pre_exchange();
}

/* ---------------------------------------------------------------------- */

void FixShearHistory::post_run()
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
    case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
      bytes += dpage5[i].size();
      break;
    case 18:
      bytes += dpage18[i].size();
      break;
    case 19:
      bytes += dpage19[i].size();
      break;
    case 8:
      bytes += dpage8[i].size();
      break;
    case 9:
      bytes += dpage9[i].size();
      break;
    case 7:
      bytes += dpage7[i].size();
      break;
    case 22:
      bytes += dpage22[i].size();
      break;
    case 23:
      bytes += dpage23[i].size();
      break;
      // Added [MO - November 2014]
    case 24:
      bytes += dpage24[i].size();
      break;
    case 25:
      bytes += dpage25[i].size();
      break;
    case 26:
      bytes += dpage26[i].size();
      break;
    case 28:
      bytes += dpage28[i].size();
      break;
    case 29:
      bytes += dpage29[i].size();
      break;
    case 30:
      bytes += dpage30[i].size();
      break;
    case 46:
      bytes += dpage46[i].size();
      break;
    case 50:
      bytes += dpage50[i].size();
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
  typedef double (*sptype5)[5]; /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
  typedef double (*sptype7)[7];
  typedef double (*sptype8)[8];
  typedef double (*sptype9)[9];
  typedef double (*sptype18)[18];
  typedef double (*sptype19)[19];
  typedef double (*sptype22)[22];
  typedef double (*sptype23)[23];
  // Added [MO - 14 November 2014]
  typedef double (*sptype24)[24];
  typedef double (*sptype25)[25];
  typedef double (*sptype26)[26];
  typedef double (*sptype28)[28];
  typedef double (*sptype29)[29];
  typedef double (*sptype30)[30];
  typedef double (*sptype46)[46];
  typedef double (*sptype50)[50];
  
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
  case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
    shearpartner5 = (sptype5 *) 
      memory->srealloc(shearpartner5,nmax*sizeof(sptype5),
		       "shear_history:shearpartner5");
    break;
  case 18:
    shearpartner18 = (sptype18 *) 
      memory->srealloc(shearpartner18,nmax*sizeof(sptype18),
		       "shear_history:shearpartner18");
    break;
  case 19:
    shearpartner19 = (sptype19 *) 
      memory->srealloc(shearpartner19,nmax*sizeof(sptype19),
		       "shear_history:shearpartner19");
    break;
  case 8:
    shearpartner8 = (sptype8 *) 
      memory->srealloc(shearpartner8,nmax*sizeof(sptype8),
		       "shear_history:shearpartner8");
    break;
  case 9:
    shearpartner9 = (sptype9 *) 
      memory->srealloc(shearpartner9,nmax*sizeof(sptype9),
		       "shear_history:shearpartner9");
    break;
  case 7:
    shearpartner7 = (sptype7 *) 
      memory->srealloc(shearpartner7,nmax*sizeof(sptype7),
		       "shear_history:shearpartner7");
    break;
  case 22:
    shearpartner22 = (sptype22 *) 
      memory->srealloc(shearpartner22,nmax*sizeof(sptype22),
		       "shear_history:shearpartner22");
    break;
  case 23:
    shearpartner23 = (sptype23 *) 
      memory->srealloc(shearpartner23,nmax*sizeof(sptype23),
		       "shear_history:shearpartner23");
    break;
  // Added [MO - 14 November 2014]
  case 24:
    shearpartner24 = (sptype24 *) 
      memory->srealloc(shearpartner24,nmax*sizeof(sptype24),
		       "shear_history:shearpartner24");
    break;
  case 25:
    shearpartner25 = (sptype25 *) 
      memory->srealloc(shearpartner25,nmax*sizeof(sptype25),
		       "shear_history:shearpartner25");
    break;
  case 26:
    shearpartner26 = (sptype26 *) 
      memory->srealloc(shearpartner26,nmax*sizeof(sptype26),
		       "shear_history:shearpartner26");
    break;
  case 46:
    shearpartner46 = (sptype46 *) 
      memory->srealloc(shearpartner46,nmax*sizeof(sptype46),
		       "shear_history:shearpartner46");
    break;
  case 28:
    shearpartner28 = (sptype28 *) 
      memory->srealloc(shearpartner28,nmax*sizeof(sptype28),
		       "shear_history:shearpartner28");
    break;
  case 29:
    shearpartner29 = (sptype29 *) 
      memory->srealloc(shearpartner29,nmax*sizeof(sptype29),
		       "shear_history:shearpartner29");
    break;
  case 30:
    shearpartner30 = (sptype30 *) 
      memory->srealloc(shearpartner30,nmax*sizeof(sptype30),
		       "shear_history:shearpartner30");
    break;
  case 50:
    shearpartner50 = (sptype50 *) 
      memory->srealloc(shearpartner50,nmax*sizeof(sptype50),
		       "shear_history:shearpartner50");
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
  case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
    shearpartner5[j] = shearpartner5[i];
    break;
  case 18:
    shearpartner18[j] = shearpartner18[i];
    break;
  case 19:
    shearpartner19[j] = shearpartner19[i];
    break;
  case 8:
    shearpartner8[j] = shearpartner8[i];
    break;
  case 9:
    shearpartner9[j] = shearpartner9[i];
    break;
  case 7:
    shearpartner7[j] = shearpartner7[i];
    break;
  case 22:
    shearpartner22[j] = shearpartner22[i];
    break;
  case 23:
    shearpartner23[j] = shearpartner23[i];
    break;
  // Added [MO - 14 November 2014] 
  case 24:
    shearpartner24[j] = shearpartner24[i];
    break;
  case 25:
    shearpartner25[j] = shearpartner25[i];
    break;
  case 26:
    shearpartner26[j] = shearpartner26[i];
    break;
  case 46:
    shearpartner46[j] = shearpartner46[i];
    break;
  case 28:
    shearpartner28[j] = shearpartner28[i];
    break;
  case 29:
    shearpartner29[j] = shearpartner29[i];
    break;
  case 30:
    shearpartner30[j] = shearpartner30[i];
    break;
  case 50:
    shearpartner50[j] = shearpartner50[i];
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
  case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 5; k++)
	buf[m++] = shearpartner5[i][n][k];
    }
    break;
  case 18:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 18; k++)
	buf[m++] = shearpartner18[i][n][k];
    }
    break;
  case 19:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 19; k++)
	buf[m++] = shearpartner19[i][n][k];
    }
    break;
  case 8:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 8; k++)
	buf[m++] = shearpartner8[i][n][k];
    }
    break;
  case 9:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 9; k++)
	buf[m++] = shearpartner9[i][n][k];
    }
    break;
  case 7:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 7; k++)
	buf[m++] = shearpartner7[i][n][k];
    }
    break;
  case 22:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 22; k++)
	buf[m++] = shearpartner22[i][n][k];
    }
    break;
  case 23:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 23; k++)
	buf[m++] = shearpartner23[i][n][k];
    }
    break;
  // Added [MO - 14 November 2014]
  case 24:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 24; k++)
	buf[m++] = shearpartner24[i][n][k];
    }
    break;
 case 25:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 25; k++)
	buf[m++] = shearpartner25[i][n][k];
    }
    break;
 case 26:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 26; k++)
	buf[m++] = shearpartner26[i][n][k];
    }
    break;
 case 46:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 46; k++)
	buf[m++] = shearpartner46[i][n][k];
    }
    break;
 case 28:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 28; k++)
	buf[m++] = shearpartner28[i][n][k];
    }
    break;
 case 29:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 29; k++)
	buf[m++] = shearpartner29[i][n][k];
    }
    break;
 case 30:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 30; k++)
	buf[m++] = shearpartner30[i][n][k];
    }
    break;
 case 50:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 50; k++)
	buf[m++] = shearpartner50[i][n][k];
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
  case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
    shearpartner5[nlocal] = dpage5->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 5; k++)
	shearpartner5[nlocal][n][k] = buf[m++];
    }
    break;
  case 18:
    shearpartner18[nlocal] = dpage18->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 18; k++)
	shearpartner18[nlocal][n][k] = buf[m++];
    }
    break;
  case 19:
    shearpartner19[nlocal] = dpage19->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 19; k++)
	shearpartner19[nlocal][n][k] = buf[m++];
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
  case 9:
    shearpartner9[nlocal] = dpage9->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 9; k++)
	shearpartner9[nlocal][n][k] = buf[m++];
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
  case 22:
    shearpartner22[nlocal] = dpage22->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 22; k++)
	shearpartner22[nlocal][n][k] = buf[m++];
    }
    break;
  case 23:
    shearpartner23[nlocal] = dpage23->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 23; k++)
	shearpartner23[nlocal][n][k] = buf[m++];
    }
    break;
   // Added [MO - 14 November 2014]
  case 24:
    shearpartner24[nlocal] = dpage24->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 24; k++)
	shearpartner24[nlocal][n][k] = buf[m++];
    }
    break;
  case 25:
    shearpartner25[nlocal] = dpage25->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 2; k++)
	shearpartner25[nlocal][n][k] = buf[m++];
    }
    break;
  case 26:
    shearpartner26[nlocal] = dpage26->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 26; k++)
	shearpartner26[nlocal][n][k] = buf[m++];
    }
    break;
  case 46:
    shearpartner46[nlocal] = dpage46->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 46; k++)
	shearpartner46[nlocal][n][k] = buf[m++];
    }
    break;
  case 28:
    shearpartner28[nlocal] = dpage28->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 28; k++)
	shearpartner28[nlocal][n][k] = buf[m++];
    }
    break;
  case 29:
    shearpartner29[nlocal] = dpage29->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 29; k++)
	shearpartner29[nlocal][n][k] = buf[m++];
    }
    break;
  case 30:
    shearpartner30[nlocal] = dpage30->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 30; k++)
	shearpartner30[nlocal][n][k] = buf[m++];
    }
    break;
  case 50:
    shearpartner50[nlocal] = dpage50->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (buf[m++]);
      for (int k = 0; k < 50; k++)
	shearpartner50[nlocal][n][k] = buf[m++];
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
  case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 5; k++)
	buf[m++] = shearpartner5[i][n][k];
    }
    break;
  case 18:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 18; k++)
	buf[m++] = shearpartner18[i][n][k];
    }
    break;
  case 19:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 19; k++)
	buf[m++] = shearpartner19[i][n][k];
    }
    break;
  case 8:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 8; k++)
	buf[m++] = shearpartner8[i][n][k];
    }
    break;
  case 9:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 9; k++)
	buf[m++] = shearpartner9[i][n][k];
    }
    break;
  case 7:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 7; k++)
	buf[m++] = shearpartner7[i][n][k];
    }
    break;
  case 22:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 22; k++)
	buf[m++] = shearpartner22[i][n][k];
    }
    break;
  case 23:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 23; k++)
	buf[m++] = shearpartner23[i][n][k];
    }
    break;
  // Added [MO - 14 November 2014] 
  case 24:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 24; k++)
	buf[m++] = shearpartner24[i][n][k];
    }
    break;
 case 25:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 25; k++)
	buf[m++] = shearpartner25[i][n][k];
    }
    break;
 case 26:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 26; k++)
	buf[m++] = shearpartner26[i][n][k];
    }
    break;
 case 46:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 46; k++)
	buf[m++] = shearpartner46[i][n][k];
    }
    break;
 case 28:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 28; k++)
	buf[m++] = shearpartner28[i][n][k];
    }
    break;
 case 29:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 29; k++)
	buf[m++] = shearpartner29[i][n][k];
    }
    break;
 case 30:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 30; k++)
	buf[m++] = shearpartner30[i][n][k];
    }
    break;
 case 50:
    for (int n = 0; n < npartner[i]; n++) {
      buf[m++] = partner[i][n];
      for (int k = 0; k < 50; k++)
	buf[m++] = shearpartner50[i][n][k];
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
  case 5: /*~~ 5 is added for the CM model [MO - 4 June 2014] ~~*/
    shearpartner5[nlocal] = dpage5->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 5; k++)
	shearpartner5[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 18:
    shearpartner18[nlocal] = dpage18->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 18; k++)
	shearpartner18[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 19:
    shearpartner19[nlocal] = dpage19->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 19; k++)
	shearpartner19[nlocal][n][k] = extra[nlocal][m++];
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
  case 9:
    shearpartner9[nlocal] = dpage9->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 9; k++)
	shearpartner9[nlocal][n][k] = extra[nlocal][m++];
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
  case 22:
    shearpartner22[nlocal] = dpage22->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 22; k++)
	shearpartner22[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 23:
    shearpartner23[nlocal] = dpage23->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 23; k++)
	shearpartner23[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  // Added [MO - 14 November 2014]
  case 24:
    shearpartner24[nlocal] = dpage24->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 24; k++)
	shearpartner24[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
  case 25:
    shearpartner25[nlocal] = dpage25->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 25; k++)
	shearpartner25[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
case 26:
    shearpartner26[nlocal] = dpage26->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 26; k++)
	shearpartner26[nlocal][n][k] = extra[nlocal][m++];
    }
    break;case 46:
    shearpartner46[nlocal] = dpage46->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 46; k++)
	shearpartner46[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
case 28:
    shearpartner28[nlocal] = dpage28->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 28; k++)
	shearpartner28[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
case 29:
    shearpartner29[nlocal] = dpage29->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 29; k++)
	shearpartner29[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
case 30:
    shearpartner30[nlocal] = dpage30->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 30; k++)
	shearpartner30[nlocal][n][k] = extra[nlocal][m++];
    }
    break;
case 50:
    shearpartner50[nlocal] = dpage50->get(npartner[nlocal]);
    for (int n = 0; n < npartner[nlocal]; n++) {
      partner[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
      for (int k = 0; k < 50; k++)
	shearpartner50[nlocal][n][k] = extra[nlocal][m++];
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

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

#include "neighbor.h"
#include "neigh_list.h"
#include "atom.h"
#include "group.h"
#include "fix_shear_history.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   granular particles
   N^2 / 2 search for neighbor pairs with partial Newton's 3rd law
   shear history must be accounted for when a neighbor pair is added
   pair added to list if atoms i and j are both owned and i < j
   pair added if j is ghost (also stored by proc owning j)
------------------------------------------------------------------------- */

void Neighbor::granular_nsq_no_newton(NeighList *list)
{
  int i,j,m,n,nn,bitmask;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr,*touchptr;
  double *shearptr;

  NeighList *listgranhistory;
  int num_quants; // added in, modified GM
  int *npartner;
  tagint **partner;
  
  /*~ The number of shear quantities is not necessarily 3, but can
    be several different values as discussed in fix_shear_history.h
    [KH - 9 January 2014]*/
  /*~~ CM, HMD and CMD models are added where HMD and CMD use the
    same number of shear quantities. [MO - 18 November 2014]*/
  //~ ----------------- without energy tracing ----------------------
  double (**shearpartner3)[3]; //~ hooke/history or hertz/history
  double (**shearpartner4)[4]; //~ shm/history
  double (**shearpartner5)[5]; //~ CM/history
  double (**shearpartner18)[18]; //~ hooke/history or hertz/history with rolling  
  double (**shearpartner19)[19]; //~ shm/history with rolling
  double (**shearpartner26)[26]; //~ HMD/history
  double (**shearpartner24)[24]; //~ shm/history with D_spin
  double (**shearpartner25)[25]; //~ CM/history with D_spin
  double (**shearpartner46)[46]; //~ HMD/history with D_spin

  //~ ----------------- with energy tracing ----------------------
  double (**shearpartner7)[7]; //~ hooke/history or hertz/history
  double (**shearpartner8)[8]; //~ shm/history
  double (**shearpartner9)[9]; //~ CM/history
  double (**shearpartner22)[22]; //~ hooke/history or hertz/history with rolling
  double (**shearpartner23)[23]; //~ shm/history with rolling
  double (**shearpartner30)[30]; //~ HMD/history
  double (**shearpartner28)[28]; //~ shm/history with D_spin
  double (**shearpartner29)[29]; //~ CM/history with D_spin
  double (**shearpartner50)[50]; //~ HMD/history with D_spin

  int **firsttouch;
  double **firstshear;
  MyPage<int> *ipage_touch;
  MyPage<double> *dpage_shear;

  double **x = atom->x;
  double *radius = atom->radius;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  if (includegroup) {
    nlocal = atom->nfirst;
    bitmask = group->bitmask[includegroup];
  }

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  MyPage<int> *ipage = list->ipage;

  FixShearHistory *fix_history = list->fix_history;
  if (fix_history) {
    fix_history->nlocal_neigh = nlocal;
    fix_history->nall_neigh = nall;
    npartner = fix_history->npartner;
    partner = fix_history->partner;
    num_quants = fix_history->num_quants; // added in, modified GM
    listgranhistory = list->listgranhistory;
    firsttouch = listgranhistory->firstneigh;
    firstshear = listgranhistory->firstdouble;
    ipage_touch = listgranhistory->ipage;
    dpage_shear = listgranhistory->dpage;

    //~ Use a switch-case structure [KH - 9 January 2014]
    switch (num_quants) {
    case 4: //~ 4 is the most likely num_quants
      shearpartner4 = fix_history->shearpartner4;
      break;
    case 3: //~ 3 is next most likely
      shearpartner3 = fix_history->shearpartner3;
      break;
    case 5: 
      shearpartner5 = fix_history->shearpartner5;
      break;
    case 18:
      shearpartner18 = fix_history->shearpartner18;
      break;
    case 19:
      shearpartner19 = fix_history->shearpartner19;
      break;
      //~~  24, 25, 26, 46 were added [MO - 14 November 2014]
    case 24:
      shearpartner24 = fix_history->shearpartner24;
      break;
    case 25:
      shearpartner25 = fix_history->shearpartner25;
      break;
    case 26:
      shearpartner26 = fix_history->shearpartner26;
      break;
    case 46:
      shearpartner46 = fix_history->shearpartner46;
      break;
    case 8:
      shearpartner8 = fix_history->shearpartner8;
      break;
    case 9:
      shearpartner9 = fix_history->shearpartner9;
      break;
    case 7:
      shearpartner7 = fix_history->shearpartner7;
      break;
    case 22:
      shearpartner22 = fix_history->shearpartner22;
      break;
    case 23:
      shearpartner23 = fix_history->shearpartner23;
      break;
    //~~ 28, 29, 30, 50 were added [MO - 14 November 2014]
    case 28:
      shearpartner28 = fix_history->shearpartner28;
      break;
    case 29:
      shearpartner29 = fix_history->shearpartner29;
      break;
    case 30:
      shearpartner30 = fix_history->shearpartner30;
      break;
    case 50:
      shearpartner50 = fix_history->shearpartner50;
      break;  
    default:
      //~ If no cases matched, there is a problem
      error->all(FLERR,"Incorrect number of shear quantities");
    }
  }

  int inum = 0;
  ipage->reset();
  if (fix_history) {
    ipage_touch->reset();
    dpage_shear->reset();
  }

  for (i = 0; i < nlocal; i++) {
    n = 0;
    neighptr = ipage->vget();
    if (fix_history) {
      nn = 0;
      touchptr = ipage_touch->vget();
      shearptr = dpage_shear->vget();
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];

    // loop over remaining atoms, owned and ghost

    for (j = i+1; j < nall; j++) {
      if (includegroup && !(mask[j] & bitmask)) continue;
      if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      radsum = radi + radius[j];
      cutsq = (radsum+skin) * (radsum+skin);

      if (rsq <= cutsq) {
        neighptr[n] = j;

        if (fix_history) {
	  //~ Use a switch-case structure [KH - 9 January 2014]
	  switch (num_quants) {
	  case 4: //~ 4 is the most likely num_quants
	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 4; kk++)
		  shearptr[nn++] = shearpartner4[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 4; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 4; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 3: //~ 3 is next most likely
 	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 3; kk++)
		  shearptr[nn++] = shearpartner3[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 3; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 3; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 5: 
 	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 5; kk++)
		  shearptr[nn++] = shearpartner5[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 5; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 5; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 18:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 18; kk++)
		  shearptr[nn++] = shearpartner18[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 18; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 18; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 19:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 19; kk++)
		  shearptr[nn++] = shearpartner19[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 19; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 19; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	    //~~ 24, 25, 26, 46 were added [MO - 14 November 2014] 
	  case 24:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 24; kk++)
		  shearptr[nn++] = shearpartner24[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 24; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 24; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 25:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 25; kk++)
		  shearptr[nn++] = shearpartner25[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 25; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 25; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 26:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 26; kk++)
		  shearptr[nn++] = shearpartner26[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 26; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 26; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;  
	  case 46:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 46; kk++)
		  shearptr[nn++] = shearpartner46[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 46; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 46; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;    
	  case 8:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 8; kk++)
		  shearptr[nn++] = shearpartner8[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 8; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 8; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 9:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 9; kk++)
		  shearptr[nn++] = shearpartner9[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 9; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 9; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 7:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 7; kk++)
		  shearptr[nn++] = shearpartner7[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 7; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 7; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 22:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 22; kk++)
		  shearptr[nn++] = shearpartner22[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 22; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 22; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 23:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 23; kk++)
		  shearptr[nn++] = shearpartner23[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 23; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 23; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  //~~ 28, 29, 30, 50 were added [MO - 14 November 2014] 
	  case 28:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 28; kk++)
		  shearptr[nn++] = shearpartner28[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 28; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 28; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 29:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 29; kk++)
		  shearptr[nn++] = shearpartner29[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 29; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 29; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;
	  case 30:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 30; kk++)
		  shearptr[nn++] = shearpartner30[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 30; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 30; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break;  
	  case 50:
    	    if (rsq < radsum*radsum) {
	      for (m = 0; m < npartner[i]; m++)
		if (partner[i][m] == tag[j]) break;
	      if (m < npartner[i]) {
		touchptr[n] = 1;
		for (int kk = 0; kk < 50; kk++)
		  shearptr[nn++] = shearpartner50[i][m][kk];
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 50; kk++)
		  shearptr[nn++] = 0.0;
	      }
	    } else {
	      touchptr[n] = 0;
	      for (int kk = 0; kk < 50; kk++)
		shearptr[nn++] = 0.0;
	    }
	    break; 
	  }
        }

        n++;
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");

    if (fix_history) {
      firsttouch[i] = touchptr;
      firstshear[i] = shearptr;
      ipage_touch->vgot(n);
      dpage_shear->vgot(nn);
    }
  }

  list->inum = inum;
}

/* ----------------------------------------------------------------------
   granular particles
   N^2 / 2 search for neighbor pairs with full Newton's 3rd law
   shear history must be accounted for when a neighbor pair is added
   pair added to list if atoms i and j are both owned and i < j
   if j is ghost only me or other proc adds pair
   decision based on itag,jtag tests
------------------------------------------------------------------------- */

void Neighbor::granular_nsq_newton(NeighList *list)
{
  int i,j,m,n,nn,itag,jtag,bitmask;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr,*touchptr;
  double *shearptr;

  NeighList *listgranhistory;
  int *npartner;
  tagint **partner;
  double (**shearpartner)[3];
  int **firsttouch;
  double **firstshear;
  MyPage<int> *ipage_touch;
  MyPage<double> *dpage_shear;

  double **x = atom->x;
  double *radius = atom->radius;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  if (includegroup) {
    nlocal = atom->nfirst;
    bitmask = group->bitmask[includegroup];
  }

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  MyPage<int> *ipage = list->ipage;

  FixShearHistory *fix_history = list->fix_history;
  if (fix_history) {
    fix_history->nlocal_neigh = nlocal;
    fix_history->nall_neigh = nall;
    npartner = fix_history->npartner;
    partner = fix_history->partner;
    shearpartner = fix_history->shearpartner3;
    listgranhistory = list->listgranhistory;
    firsttouch = listgranhistory->firstneigh;
    firstshear = listgranhistory->firstdouble;
    ipage_touch = listgranhistory->ipage;
    dpage_shear = listgranhistory->dpage;
  }

  int inum = 0;
  ipage->reset();
  if (fix_history) {
    ipage_touch->reset();
    dpage_shear->reset();
  }

  for (i = 0; i < nlocal; i++) {
    n = 0;
    neighptr = ipage->vget();
    if (fix_history) {
      nn = 0;
      touchptr = ipage_touch->vget();
      shearptr = dpage_shear->vget();
    }

    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];

    // loop over remaining atoms, owned and ghost

    for (j = i+1; j < nall; j++) {
      if (includegroup && !(mask[j] & bitmask)) continue;

      if (j >= nlocal) {
        jtag = tag[j];
        if (itag > jtag) {
          if ((itag+jtag) % 2 == 0) continue;
        } else if (itag < jtag) {
          if ((itag+jtag) % 2 == 1) continue;
        } else {
          if (x[j][2] < ztmp) continue;
          if (x[j][2] == ztmp) {
            if (x[j][1] < ytmp) continue;
            if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
          }
        }
      }

      if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      radsum = radi + radius[j];
      cutsq = (radsum+skin) * (radsum+skin);

      if (rsq <= cutsq) {
        neighptr[n++] = j;

        if (fix_history) {
          if (rsq < radsum*radsum) {
            for (m = 0; m < npartner[i]; m++)
              if (partner[i][m] == tag[j]) break;
            if (m < npartner[i]) {
              touchptr[n] = 1;
              shearptr[nn++] = shearpartner[i][m][0];
              shearptr[nn++] = shearpartner[i][m][1];
              shearptr[nn++] = shearpartner[i][m][2];
            } else {
              touchptr[n] = 0;
              shearptr[nn++] = 0.0;
              shearptr[nn++] = 0.0;
              shearptr[nn++] = 0.0;
            }
          } else {
            touchptr[n] = 0;
            shearptr[nn++] = 0.0;
            shearptr[nn++] = 0.0;
            shearptr[nn++] = 0.0;
          }
        }

        n++;
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");

    if (fix_history) {
      firsttouch[i] = touchptr;
      firstshear[i] = shearptr;
      ipage_touch->vgot(n);
      dpage_shear->vgot(nn);
    }
  }

  list->inum = inum;
}

/* ----------------------------------------------------------------------
   granular particles
   binned neighbor list construction with partial Newton's 3rd law
   shear history must be accounted for when a neighbor pair is added
   each owned atom i checks own bin and surrounding bins in non-Newton stencil
   pair stored once if i,j are both owned and i < j
   pair stored by me if j is ghost (also stored by proc owning j)
------------------------------------------------------------------------- */

void Neighbor::granular_bin_no_newton(NeighList *list)
{
  int i,j,k,m,n,nn,ibin;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr,*touchptr;
  double *shearptr;

  NeighList *listgranhistory;
  int num_quants; // added in, modified GM
  int *npartner;
  tagint **partner;
  
  //~ As in granular_nsq_no_newton [KH - 9 January 2014]
  //~~ several lines were added [MO - 14 November 2014]
  //~ ----------------- without energy tracing ----------------------
  double (**shearpartner3)[3]; //~ hooke/history or hertz/history
  double (**shearpartner4)[4]; //~ shm/history
  double (**shearpartner5)[5]; //~ CM/history
  double (**shearpartner18)[18]; //~ hooke/history or hertz/history with rolling
  double (**shearpartner19)[19]; //~ shm/history with rolling
  double (**shearpartner26)[26]; //~ HMD/history [MO - 14 November 2014]
  double (**shearpartner24)[24]; //~ shm/history with D_spin [MO - 14 November 2014]
  double (**shearpartner25)[25]; //~ CM/history with D_spin  [MO - 14 November 2014]
  double (**shearpartner46)[46]; //~ HMD/history with D_spin [MO - 14 November 2014]

  //~ ----------------- with energy tracing ----------------------
  double (**shearpartner7)[7]; //~ hooke/history or hertz/history
  double (**shearpartner8)[8]; //~ shm/history
  double (**shearpartner9)[9]; //~ CM/history
  double (**shearpartner22)[22]; //~ hooke/history or hertz/history with rolling
  double (**shearpartner23)[23]; //~ shm/history with rolling
  double (**shearpartner30)[30]; //~ HMD/history [MO - 14 November 2014]
  double (**shearpartner28)[28]; //~ shm/history with D_spin [MO - 14 November 2014]
  double (**shearpartner29)[29]; //~ CM/history with D_spin  [MO - 14 November 2014]
  double (**shearpartner50)[50]; //~ HMD/history with D_spin [MO - 14 November 2014]	

  int **firsttouch;
  double **firstshear;
  MyPage<int> *ipage_touch;
  MyPage<double> *dpage_shear;

  // bin local & ghost atoms

  bin_atoms();

  // loop over each atom, storing neighbors

  double **x = atom->x;
  double *radius = atom->radius;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  if (includegroup) nlocal = atom->nfirst;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int nstencil = list->nstencil;
  int *stencil = list->stencil;
  MyPage<int> *ipage = list->ipage;

  FixShearHistory *fix_history = list->fix_history;
  if (fix_history) {
    fix_history->nlocal_neigh = nlocal;
    fix_history->nall_neigh = nlocal + atom->nghost;
    npartner = fix_history->npartner;
    partner = fix_history->partner;
    num_quants = fix_history->num_quants; // added in, modified GM
    listgranhistory = list->listgranhistory;
    firsttouch = listgranhistory->firstneigh;
    firstshear = listgranhistory->firstdouble;
    ipage_touch = listgranhistory->ipage;
    dpage_shear = listgranhistory->dpage;

    //~ Use a switch-case structure [KH - 9 January 2014]
    switch (num_quants) {
    case 4: //~ 4 is the most likely num_quants
      shearpartner4 = fix_history->shearpartner4;
      break;
    case 3: //~ 3 is next most likely
      shearpartner3 = fix_history->shearpartner3;
      break;
    case 5: 
      shearpartner5 = fix_history->shearpartner5;
      break;
    case 18:
      shearpartner18 = fix_history->shearpartner18;
      break;
    case 19:
      shearpartner19 = fix_history->shearpartner19;
      break;
    // 24, 25, 26, 46 were added [MO - 14 November 2014]
    case 24: 
      shearpartner24 = fix_history->shearpartner24;
      break;
    case 25:
      shearpartner25 = fix_history->shearpartner25;
      break;
    case 26:
      shearpartner26 = fix_history->shearpartner26;
      break;
    case 46:
      shearpartner46 = fix_history->shearpartner46;
      break;
      /////
    case 8:
      shearpartner8 = fix_history->shearpartner8;
      break;
    case 9:
      shearpartner9 = fix_history->shearpartner9;
      break;
    case 7:
      shearpartner7 = fix_history->shearpartner7;
      break;
    case 22:
      shearpartner22 = fix_history->shearpartner22;
      break;
    case 23:
      shearpartner23 = fix_history->shearpartner23;
      break;
    // 28, 29, 30, 50 were added [MO - 14 November 2014]
    case 28:
      shearpartner28 = fix_history->shearpartner28;
      break;
    case 29:
      shearpartner29 = fix_history->shearpartner29;
      break;
    case 30:
      shearpartner30 = fix_history->shearpartner30;
      break;
    case 50:
      shearpartner50 = fix_history->shearpartner50;
      break;
    default:
      //~ If no cases matched, there is a problem
      error->all(FLERR,"Incorrect number of shear quantities");
    }
  }

  int inum = 0;
  ipage->reset();
  if (fix_history) {
    ipage_touch->reset();
    dpage_shear->reset();
  }

  for (i = 0; i < nlocal; i++) {
    n = 0;
    neighptr = ipage->vget();
    if (fix_history) {
      nn = 0;
      touchptr = ipage_touch->vget();
      shearptr = dpage_shear->vget();
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];
    ibin = coord2bin(x[i]);

    // loop over all atoms in surrounding bins in stencil including self
    // only store pair if i < j
    // stores own/own pairs only once
    // stores own/ghost pairs on both procs

    for (k = 0; k < nstencil; k++) {
      for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
        if (j <= i) continue;
        if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        radsum = radi + radius[j];
        cutsq = (radsum+skin) * (radsum+skin);

        if (rsq <= cutsq) {
          neighptr[n] = j;

          if (fix_history) {
	    //~ Use a switch-case structure [KH - 9 January 2014]
	    switch (num_quants) {
	    case 4: //~ 4 is the most likely num_quants
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 4; kk++)
		    shearptr[nn++] = shearpartner4[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 4; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 4; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    case 3: //~ 3 is next most likely
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 3; kk++)
		    shearptr[nn++] = shearpartner3[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 3; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 3; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    case 5: 
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 5; kk++)
		    shearptr[nn++] = shearpartner5[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 5; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 5; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    case 18:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 18; kk++)
		    shearptr[nn++] = shearpartner18[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 18; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 18; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    case 19:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 19; kk++)
		    shearptr[nn++] = shearpartner19[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 19; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 19; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    // 24, 25, 26, 46 were added [MO - 14 November 2014]
	    case 24:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 24; kk++)
		    shearptr[nn++] = shearpartner24[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 24; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 24; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;  
	     case 25:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 25; kk++)
		    shearptr[nn++] = shearpartner25[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 25; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 25; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;  
	    case 26:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 26; kk++)
		    shearptr[nn++] = shearpartner26[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 26; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 26; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;    
	     case 46:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 46; kk++)
		    shearptr[nn++] = shearpartner46[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 46; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 46; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;    
	    ////
	    case 8:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 8; kk++)
		    shearptr[nn++] = shearpartner8[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 8; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 8; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    case 9:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 9; kk++)
		    shearptr[nn++] = shearpartner9[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 9; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 9; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    case 7:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 7; kk++)
		    shearptr[nn++] = shearpartner7[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 7; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 7; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    case 22:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 22; kk++)
		    shearptr[nn++] = shearpartner22[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 22; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 22; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    case 23:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 23; kk++)
		    shearptr[nn++] = shearpartner23[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 23; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 23; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    // 28, 29, 30, 50 were added [MO - 14 November 2014]
	    case 28:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 28; kk++)
		    shearptr[nn++] = shearpartner28[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 28; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 28; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;  
	     case 29:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 29; kk++)
		    shearptr[nn++] = shearpartner29[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 29; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 29; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;    
	    case 30:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 30; kk++)
		    shearptr[nn++] = shearpartner30[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 30; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 30; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;      
	     case 50:
	      if (rsq < radsum*radsum) {
		for (m = 0; m < npartner[i]; m++)
		  if (partner[i][m] == tag[j]) break;
		if (m < npartner[i]) {
		  touchptr[n] = 1;
		  for (int kk = 0; kk < 50; kk++)
		    shearptr[nn++] = shearpartner50[i][m][kk];
		} else {
		  touchptr[n] = 0;
		  for (int kk = 0; kk < 50; kk++)
		    shearptr[nn++] = 0.0;
		}
	      } else {
		touchptr[n] = 0;
		for (int kk = 0; kk < 50; kk++)
		  shearptr[nn++] = 0.0;
	      }
	      break;
	    }
          }

          n++;
        }
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");

    if (fix_history) {
      firsttouch[i] = touchptr;
      firstshear[i] = shearptr;
      ipage_touch->vgot(n);
      dpage_shear->vgot(nn);
    }
  }

  list->inum = inum;
}

/* ----------------------------------------------------------------------
   granular particles
   binned neighbor list construction with full Newton's 3rd law
   shear history must be accounted for when a neighbor pair is added
   each owned atom i checks its own bin and other bins in Newton stencil
   every pair stored exactly once by some processor
------------------------------------------------------------------------- */

void Neighbor::granular_bin_newton(NeighList *list)
{
  int i,j,k,m,n,nn,ibin;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr,*touchptr;
  double *shearptr;

  NeighList *listgranhistory;
  int *npartner;
  tagint **partner;
  double (**shearpartner)[3];
  int **firsttouch;
  double **firstshear;
  MyPage<int> *ipage_touch;
  MyPage<double> *dpage_shear;

  // bin local & ghost atoms

  bin_atoms();

  // loop over each atom, storing neighbors

  double **x = atom->x;
  double *radius = atom->radius;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  if (includegroup) nlocal = atom->nfirst;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int nstencil = list->nstencil;
  int *stencil = list->stencil;
  MyPage<int> *ipage = list->ipage;

  FixShearHistory *fix_history = list->fix_history;
  if (fix_history) {
    fix_history->nlocal_neigh = nlocal;
    fix_history->nall_neigh = nlocal + atom->nghost;
    npartner = fix_history->npartner;
    partner = fix_history->partner;
    shearpartner = fix_history->shearpartner3;
    listgranhistory = list->listgranhistory;
    firsttouch = listgranhistory->firstneigh;
    firstshear = listgranhistory->firstdouble;
    ipage_touch = listgranhistory->ipage;
    dpage_shear = listgranhistory->dpage;
  }

  int inum = 0;
  ipage->reset();
  if (fix_history) {
    ipage_touch->reset();
    dpage_shear->reset();
  }

  for (i = 0; i < nlocal; i++) {
    n = 0;
    neighptr = ipage->vget();
    if (fix_history) {
      nn = 0;
      touchptr = ipage_touch->vget();
      shearptr = dpage_shear->vget();
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];

    // loop over rest of atoms in i's bin, ghosts are at end of linked list
    // if j is owned atom, store it, since j is beyond i in linked list
    // if j is ghost, only store if j coords are "above and to the right" of i

    for (j = bins[i]; j >= 0; j = bins[j]) {
      if (j >= nlocal) {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp) {
          if (x[j][1] < ytmp) continue;
          if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
        }
      }

      if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      radsum = radi + radius[j];
      cutsq = (radsum+skin) * (radsum+skin);

      if (rsq <= cutsq) {
        neighptr[n] = j;

        if (fix_history) {
          if (rsq < radsum*radsum) {
            for (m = 0; m < npartner[i]; m++)
              if (partner[i][m] == tag[j]) break;
            if (m < npartner[i]) {
              touchptr[n] = 1;
              shearptr[nn++] = shearpartner[i][m][0];
              shearptr[nn++] = shearpartner[i][m][1];
              shearptr[nn++] = shearpartner[i][m][2];
            } else {
              touchptr[n] = 0;
              shearptr[nn++] = 0.0;
              shearptr[nn++] = 0.0;
              shearptr[nn++] = 0.0;
            }
          } else {
            touchptr[n] = 0;
            shearptr[nn++] = 0.0;
            shearptr[nn++] = 0.0;
            shearptr[nn++] = 0.0;
          }
        }

        n++;
      }
    }

    // loop over all atoms in other bins in stencil, store every pair

    ibin = coord2bin(x[i]);
    for (k = 0; k < nstencil; k++) {
      for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
        if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        radsum = radi + radius[j];
        cutsq = (radsum+skin) * (radsum+skin);

        if (rsq <= cutsq) {
          neighptr[n] = j;

          if (fix_history) {
            if (rsq < radsum*radsum) {
              for (m = 0; m < npartner[i]; m++)
                if (partner[i][m] == tag[j]) break;
              if (m < npartner[i]) {
                touchptr[n] = 1;
                shearptr[nn++] = shearpartner[i][m][0];
                shearptr[nn++] = shearpartner[i][m][1];
                shearptr[nn++] = shearpartner[i][m][2];
              } else {
                touchptr[n] = 0;
                shearptr[nn++] = 0.0;
                shearptr[nn++] = 0.0;
                shearptr[nn++] = 0.0;
              }
            } else {
              touchptr[n] = 0;
              shearptr[nn++] = 0.0;
              shearptr[nn++] = 0.0;
              shearptr[nn++] = 0.0;
            }
          }
          
          n++;
        }
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");

    if (fix_history) {
      firsttouch[i] = touchptr;
      firstshear[i] = shearptr;
      ipage_touch->vgot(n);
      dpage_shear->vgot(nn);
    }
  }

  list->inum = inum;
}

/* ----------------------------------------------------------------------
   granular particles
   binned neighbor list construction with Newton's 3rd law for triclinic
   shear history must be accounted for when a neighbor pair is added
   each owned atom i checks its own bin and other bins in triclinic stencil
   every pair stored exactly once by some processor
------------------------------------------------------------------------- */

void Neighbor::granular_bin_newton_tri(NeighList *list)
{
  int i,j,k,m,n,nn,ibin;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,cutsq;
  int *neighptr,*touchptr;
  double *shearptr;

  NeighList *listgranhistory;
  int *npartner;
  tagint **partner;
  double (**shearpartner)[3];
  int **firsttouch;
  double **firstshear;
  MyPage<int> *ipage_touch;
  MyPage<double> *dpage_shear;

  // bin local & ghost atoms

  bin_atoms();

  // loop over each atom, storing neighbors

  double **x = atom->x;
  double *radius = atom->radius;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  if (includegroup) nlocal = atom->nfirst;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int nstencil = list->nstencil;
  int *stencil = list->stencil;
  MyPage<int> *ipage = list->ipage;

  FixShearHistory *fix_history = list->fix_history;
  if (fix_history) {
    fix_history->nlocal_neigh = nlocal;
    fix_history->nall_neigh = nlocal + atom->nghost;
    npartner = fix_history->npartner;
    partner = fix_history->partner;
    shearpartner = fix_history->shearpartner3;
    listgranhistory = list->listgranhistory;
    firsttouch = listgranhistory->firstneigh;
    firstshear = listgranhistory->firstdouble;
    ipage_touch = listgranhistory->ipage;
    dpage_shear = listgranhistory->dpage;
  }

  int inum = 0;
  ipage->reset();
  if (fix_history) {
    ipage_touch->reset();
    dpage_shear->reset();
  }

  for (i = 0; i < nlocal; i++) {
    n = 0;
    neighptr = ipage->vget();
    if (fix_history) {
      nn = 0;
      touchptr = ipage_touch->vget();
      shearptr = dpage_shear->vget();
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];

    // loop over all atoms in bins in stencil
    // pairs for atoms j "below" i are excluded
    // below = lower z or (equal z and lower y) or (equal zy and lower x)
    //         (equal zyx and j <= i)
    // latter excludes self-self interaction but allows superposed atoms

    ibin = coord2bin(x[i]);
    for (k = 0; k < nstencil; k++) {
      for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp) {
          if (x[j][1] < ytmp) continue;
          if (x[j][1] == ytmp) {
            if (x[j][0] < xtmp) continue;
            if (x[j][0] == xtmp && j <= i) continue;
          }
        }

        if (exclude && exclusion(i,j,type[i],type[j],mask,molecule)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        radsum = radi + radius[j];
        cutsq = (radsum+skin) * (radsum+skin);

        if (rsq <= cutsq) {
          neighptr[n++] = j;

          if (fix_history) {
            if (rsq < radsum*radsum) {
              for (m = 0; m < npartner[i]; m++)
                if (partner[i][m] == tag[j]) break;
              if (m < npartner[i]) {
                touchptr[n] = 1;
                shearptr[nn++] = shearpartner[i][m][0];
                shearptr[nn++] = shearpartner[i][m][1];
                shearptr[nn++] = shearpartner[i][m][2];
              } else {
                touchptr[n] = 0;
                shearptr[nn++] = 0.0;
                shearptr[nn++] = 0.0;
                shearptr[nn++] = 0.0;
              }
            } else {
              touchptr[n] = 0;
              shearptr[nn++] = 0.0;
              shearptr[nn++] = 0.0;
              shearptr[nn++] = 0.0;
            }
          }

          n++;
        }
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");

    if (fix_history) {
      firsttouch[i] = touchptr;
      firstshear[i] = shearptr;
      ipage_touch->vgot(n);
      dpage_shear->vgot(nn);
    }
  }

  list->inum = inum;
}

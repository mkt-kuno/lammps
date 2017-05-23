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
#include <string.h>
#include <stdlib.h>
#include "compute_coord_gran.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeCoordGran::ComputeCoordGran(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  typelo(NULL), typehi(NULL), cvec(NULL), carray(NULL)
{
  if (narg < 3) error->all(FLERR,"Illegal compute coord/gran command");

  ncol = narg-3 + 1;
  int ntypes = atom->ntypes;
  typelo = new int[ncol];
  typehi = new int[ncol];

  if (narg == 3) {
    ncol = 1;
    typelo[0] = 1;
    typehi[0] = ntypes;
  } else {
    ncol = 0;
    int iarg = 3;
    while (iarg < narg) {
      force->bounds(FLERR,arg[iarg],ntypes,typelo[ncol],typehi[ncol]);
      if (typelo[ncol] > typehi[ncol])
        error->all(FLERR,"Illegal compute coord/gran command");
      ncol++;
      iarg++;
    }
  }

  peratom_flag = 1;
  if (ncol == 1) size_peratom_cols = 0;
  else size_peratom_cols = ncol;

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeCoordGran::~ComputeCoordGran()
{
  delete [] typelo;
  delete [] typehi;
  memory->destroy(cvec);
  memory->destroy(carray);
}

/* ---------------------------------------------------------------------- */

void ComputeCoordGran::init()
{
  if (force->pair == NULL)
    error->all(FLERR,"Compute coord/gran requires a pair style be defined");

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"coord/gran") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute coord/gran");
}

/* ---------------------------------------------------------------------- */

void ComputeCoordGran::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeCoordGran::compute_peratom()
{
  int i,j,m,ii,jj,inum,jnum,jtype,n;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *count;

  invoked_peratom = update->ntimestep;

  // grow coordination array if necessary

  if (atom->nmax > nmax) {
    if (ncol == 1) {
      memory->destroy(cvec);
      nmax = atom->nmax;
      memory->create(cvec,nmax,"coord/gran:cvec");
      vector_atom = cvec;
    } else {
      memory->destroy(carray);
      nmax = atom->nmax;
      memory->create(carray,nmax,ncol,"coord/gran:carray");
      array_atom = carray;
    }
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute coordination number(s) for each atom in group
  // use full neighbor list to count contacting atoms

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  double *radius = atom->radius; //~ Needed to be included [KH - 18 October 2011]

  if (ncol == 1) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (mask[i] & groupbit) {
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        n = 0;
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;
          
          jtype = type[j];
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;

	  //~ The calculation of coordination number was modified from the expression given in ComputeCoordAtom [KH - 7 December 2012]
	  if (rsq <= (radius[i] + radius[j])*(radius[i] + radius[j]) && jtype >= typelo[0] && jtype <= typehi[0]) n++;
        }
        
        cvec[i] = n;
      } else cvec[i] = 0.0;
    }

  } else {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      count = carray[i];
      for (m = 0; m < ncol; m++) count[m] = 0.0;

      if (mask[i] & groupbit) {
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        jlist = firstneigh[i];
        jnum = numneigh[i];


        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;
          
          jtype = type[j];
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;

	  //~ The statement below for calculating the coordination number was modified [KH - 11 October 2012]
	  if (rsq <= (radius[i] + radius[j])*(radius[i] + radius[j])) {
            for (m = 0; m < ncol; m++)
              if (jtype >= typelo[m] && jtype <= typehi[m])
                count[m] += 1.0;
          }
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeCoordGran::memory_usage()
{
  double bytes = ncol*nmax * sizeof(double);
  return bytes;
}

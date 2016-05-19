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
   Contributing author: Daniel Schwen
------------------------------------------------------------------------- */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "compute_laguerre_atom.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "comm.h"

#include <vector>
#include "voro++.hh"

using namespace LAMMPS_NS;
using namespace voro;

/* ---------------------------------------------------------------------- */

ComputeLaguerre::ComputeLaguerre(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 3 ) error->all(FLERR,"Illegal compute laguerre/atom command");
  memflag = verflag = volflag =  0;
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"membrane") == 0) {
      memflag = 1;
      size_peratom_cols = 3; // areas that intersect walls
      xleft = atof(arg[iarg+1]);
      xright = atof(arg[iarg+2]);
      yleft = atof(arg[iarg+3]);
      yright = atof(arg[iarg+4]);
      zleft = atof(arg[iarg+5]);
      zright = atof(arg[iarg+6]);
      iarg = iarg + 6;
    } else if (strcmp(arg[iarg],"volume") == 0) {
      volflag = 1;
      double *sublo = domain->sublo;
      double *subhi = domain->subhi;
      double *cut = comm->cutghost;
      xleft = sublo[0]-cut[0]; // as in compute voronoi/atom
      xright = subhi[0]+cut[0];
      yleft = sublo[1]-cut[1];
      yright = subhi[1]+cut[1];
      zleft = sublo[2]-cut[2];
      zright = subhi[2]+cut[2];
      size_peratom_cols = 1; // volumes of cells
      iarg = iarg +1;
    } else if (strcmp(arg[iarg],"vertices") == 0) {
      verflag = 1;
      size_peratom_cols = 501; // how many columns for vertices?
      iarg = iarg +1;
    } else error->all(FLERR,"Illegal compute laguerre/atom command");
    iarg++;
  }

  if (volflag ==1 & memflag ==1) error->all(FLERR,"Cannot do Laguerre for volume and membrane");

  peratom_flag = 1;

  nmax = 0;
  voro = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeLaguerre::~ComputeLaguerre()
{
  memory->destroy(voro);
}

/* ---------------------------------------------------------------------- */

void ComputeLaguerre::init()
{
  if (domain->dimension != 3)
    error->all(FLERR,"Compute laguerre/atom not allowed for 2d systems");
  if (domain->triclinic != 0)
    error->all(FLERR,"Compute laguerre/atom not allowed for triclinic boxes");

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"laguerre/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute laguerre/atom command");
}

/* ----------------------------------------------------------------------
   gather compute vector data from other nodes
------------------------------------------------------------------------- */

void ComputeLaguerre::compute_peratom()
{
  int i,maxk,kk,maxv,numv,facei,where,ivor;
  double areax,areay,areaz;

  invoked_peratom = update->ntimestep;

  // grow per atom array if necessary

  int nlocal = atom->nlocal;
  if (nlocal > nmax) {
    memory->destroy(voro);
    nmax = atom->nmax;
    memory->create(voro,nmax,size_peratom_cols,"laguerre/atom:voro");
    array_atom = voro;
  }

  // n = # of voro++ spatial hash cells
  // TODO: make square

  int nall = nlocal + atom->nghost;
  int n = int(floor( pow( double(nall)/8.0, 0.333333 ) ));
  n = (n==0) ? 1 : n;
  int m = int(floor( pow( double(nall)/4.0, 0.333333 ) ));
  m = (m==0) ? 1 : m;

  // initialize voro++ container
  // preallocates 8 atoms per cell
  // voro++ allocates more memory if needed

  container_poly conp(xleft,xright,yleft,yright,zleft,zright,
                n,n,n,false,false,false,8);


  ///should I use a wall class?
  /*wall_plane xl(1.0,0.0,0.0,xleft,-1);
  wall_plane xr(1.0,0.0,0.0,xright,-2);
  wall_plane yl(0.0,1.0,0.0,yleft,-3);
  wall_plane yr(0.0,1.0,0.0,yright,-4);
  wall_plane zl(0.0,0.0,1.0,zleft,-5);
  wall_plane zr(0.0,0.0,1.0,zright,-6);
  conp.add_wall(xl); //add a wall somehow
  conp.add_wall(yl);
  conp.add_wall(zl);
  conp.add_wall(xr);
  conp.add_wall(yr);
  conp.add_wall(zr);*/

  // pass coordinates for local and ghost atoms to voro++

  double **x = atom->x;
  double *radius = atom->radius;
  double maxradi=0.0;
  for (i = 0; i < nall; i++) {
     conp.put(i,x[i][0],x[i][1],x[i][2],radius[i]);  //could maybe pass on less atoms
     if (radius[i]>maxradi) maxradi=radius[i];
  }

  container_poly coni(xleft-2*maxradi,xright+2*maxradi,yleft-2*maxradi,yright+2*maxradi,zleft-2*maxradi,zright+2*maxradi,
                m,m,m,false,false,false,8);

  for (i = 0; i < nall; i++) {
     coni.put(i,x[i][0],x[i][1],x[i][2],radius[i]);
  }

  int idimag = -7; //id for imaginary particles
  double coorimag;

  // invoke voro++ and fetch results for owned atoms in group

  int *mask = atom->mask;
  std::vector<int> neigh,iverts,neighimag;
  std::vector<double> areas;
  std::vector<double> xyzverts;

 FILE *fpp,*fpp1;

 fpp = fopen("/home/gmarketo/LammpsWork/ForgitLAMMPS/ForGMmain/Membrane/CheckOldEdges.txt","w");
 fpp1 = fopen("/home/gmarketo/LammpsWork/ForgitLAMMPS/ForGMmain/Membrane/CheckNewEdges.txt","w");
 fprintf(fpp1,"i,x[i][0],x[i][1],x[i][2],neighimag[kk],thisis,x[thisis][0],x[thisis][1],x[thisis][2],    VORONOI VERTICES\n");

  voronoicell_neighbor c;
  c_loop_all cl(conp);
  if (cl.start()) do if (conp.compute_cell(c,cl)) {
    i = cl.pid();
    if (i < nlocal && (mask[i] & groupbit)) {
      for (int mmm = 0; mmm < size_peratom_cols; mmm++) voro[i][mmm] = 0.0; //initialise as zero
      if (volflag == 1) voro[i][0] = c.volume();
      else if (memflag == 1) {
        c.neighbors(neigh);
        maxk = neigh.size();
        ivor =3;
        //loop through neigh and calculate area if id<0
        for (kk = 0; kk < maxk; kk++) {
          if (neigh[kk]<0) {

            // Add imaginary atoms to coni!
            if (neigh[kk]==-1) {
                idimag=1000+i;
                coorimag=x[i][0]-1.01*radius[i];
                coni.put(idimag,coorimag,x[i][1],x[i][2],0.01*radius[i]);// check that imaginary grains not added twice!
            }
            if (neigh[kk]==-2) {
                idimag=2000+i;
                coorimag=x[i][0]+1.01*radius[i];
                coni.put(idimag,coorimag,x[i][1],x[i][2],0.01*radius[i]);
            }
            if (neigh[kk]==-3) {
                idimag=3000+i;
                coorimag=x[i][1]-1.01*radius[i];
                coni.put(idimag,x[i][0],coorimag,x[i][2],0.01*radius[i]);
            }
            if (neigh[kk]==-4) {
                idimag=4000+i;
                coorimag=x[i][1]+1.01*radius[i];
                coni.put(idimag,x[i][0],coorimag,x[i][2],0.01*radius[i]);
            }
            if (neigh[kk]==-5) {
                idimag=5000+i;
                coorimag=x[i][2]-1.01*radius[i];
                coni.put(idimag,x[i][0],x[i][1],coorimag,0.01*radius[i]);
            }
            if (neigh[kk]==-6) {
                idimag=6000+i;
                coorimag=x[i][2]+1.01*radius[i];
                coni.put(idimag,x[i][0],x[i][1],coorimag,0.01*radius[i]);
            }
        fprintf(fpp,"%d  %f %f %f %d %d\n",i,x[i][0],x[i][1],x[i][2],neigh[kk],idimag);


          }
        }
      }

    }
  } while (cl.inc());

 fclose(fpp);


// Loop over the container that also has in it the 'imaginary grains'
  voronoicell_neighbor cimag;
  c_loop_all climag(coni);
  if (climag.start()) do if (coni.compute_cell(cimag,climag)) {
    i = climag.pid();
    if (i < nlocal && (mask[i] & groupbit)) {
      for (int mmm = 0; mmm < size_peratom_cols; mmm++) voro[i][mmm] = 0.0; //initialise as zero
      if (volflag == 1) voro[i][0] = cimag.volume();
      else if (memflag == 1) {
        areax = 0.0;
        areay = 0.0;
        areaz = 0.0;
        cimag.neighbors(neighimag);
        maxk = neighimag.size();
        ivor =3;
        //loop through neighimag and get areas id<=-7
        for (kk = 0; kk < maxk; kk++) {
          if (neighimag[kk]>=1000) {
        int thisis=neighimag[kk]%1000; // the id of grain
        fprintf(fpp1,"%d %f %f %f %d %d %f %f %f ",i,x[i][0],x[i][1],x[i][2],neighimag[kk],thisis,x[thisis][0],x[thisis][1],x[thisis][2]);
            cimag.face_areas(areas);
            if (verflag == 1) cimag.face_vertices(iverts);
            /*if (neighimag[kk]==-1 || neighimag[kk]==-2) areax+=areas[kk];
            if (neighimag[kk]==-3 || neighimag[kk]==-4) areay+=areas[kk];
            if (neighimag[kk]==-5 || neighimag[kk]==-6) areaz+=areas[kk];*/

            if (verflag == 1) { // the vertices of the kk'th face are needed
              cimag.vertices(x[i][0],x[i][1],x[i][2],xyzverts);
              maxv = iverts.size();
              where = 0; // the location of the number of vertices for a face in the iverts vector
              for (facei = 0; facei < kk; facei++) {
                 numv = iverts[where];
                 where = where + numv + 1;
              } //where now stores where to find the number of vertices
              numv = iverts[where]; // for the kk'th face
              if (size_peratom_cols<ivor+3*numv+3) {
                printf("size_peratom_cols is %d and I will need %d+3x%d+3\n",size_peratom_cols,ivor,numv);
                printf("neighimag[kk]=%d\n",neighimag[kk]);
                error->all(FLERR,"Increase size_peratom_cols");
              }
              for (int mm = 0; mm < numv; mm++) { 
                 voro[i][ivor] = xyzverts[3*iverts[where+mm+1]];  //only write the ones that are interesting
                 voro[i][ivor+1] = xyzverts[3*iverts[where+mm+1]+1];
                 voro[i][ivor+2] = xyzverts[3*iverts[where+mm+1]+2];
        fprintf(fpp1,"%f %f %f ",voro[i][ivor],voro[i][ivor+1],voro[i][ivor+2]);
                 ivor+=3;
                 if (mm == numv-1) {
                   voro[i][ivor] = xyzverts[3*iverts[where+1]]; //rewrite the first one so as to close the surface
                   voro[i][ivor+1] = xyzverts[3*iverts[where+1]+1];
                   voro[i][ivor+2] = xyzverts[3*iverts[where+1]+2];
        fprintf(fpp1,"%f %f %f\n",voro[i][ivor],voro[i][ivor+1],voro[i][ivor+2]);
                   ivor+=3;
                 }
              }
            }

          }
        }
        voro[i][0] = areax;
        voro[i][1] = areay;
        voro[i][2] = areaz;
      }
    } else if (i < nlocal) {
        for (int ll = 0; ll < size_peratom_cols; ll++) voro[i][ll] = 0.0;
    }
  } while (climag.inc());


fclose(fpp1);
}



/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeLaguerre::memory_usage()
{
  double bytes = size_peratom_cols * nmax * sizeof(double);
  return bytes;
}

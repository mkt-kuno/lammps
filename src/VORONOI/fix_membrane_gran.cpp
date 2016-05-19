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
   Contributing authors: Leo Silbert (SNL), Gary Grest (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix_membrane_gran.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "modify.h"
#include "respa.h"
#include "memory.h"
#include "input.h"
#include "variable.h"
#include "math_const.h"
#include "error.h"
#include "comm.h"

#include <vector>
#include "voro++.hh"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace voro;

enum{XPLANE,YPLANE,ZPLANE,ZCYLINDER};    // XYZ PLANE need to be 0,1,2

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixMembraneGran::FixMembraneGran(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 15) error->all(FLERR,"Illegal fix membrane/gran command");

  if (!atom->sphere_flag)
    error->all(FLERR,"Fix membrane/gran requires atom style sphere");

/*  FROM FIX
  virial_flag = 1; // this fix can contribute to compute stress/atom

  restart_peratom = 1; set to 0
  create_attribute = 1;// 1 if fix stores attributes that need setting when a new atom is created

  vector_flag = 1;add these 3
  size_vector = 5;
  global_freq = 1;*/



  time_origin = update->ntimestep;


  xleft = atof(arg[3]);
  xright = atof(arg[4]);
  yleft = atof(arg[5]);
  yright = atof(arg[6]);
  zleft = atof(arg[7]);
  zright = atof(arg[8]);

  if (strcmp(arg[9],"NULL") == 0) xflag=0; 
  else {
    pressurex = atof(arg[9]);
    xflag=1;
  }
  if (strcmp(arg[10],"NULL") == 0) yflag=0; 
  else {
    pressurey = atof(arg[10]);
    yflag=1;
  }
  if (strcmp(arg[11],"NULL") == 0) zflag=0; 
  else {
    pressurez = atof(arg[11]);
    zflag=1;
  }

  distx = atof(arg[12]);
  disty = atof(arg[13]);
  distz = atof(arg[14]);

  /*FROM COMPUTE 
  peratom_flag = 1; also for fix // 0/1 if per-atom data is stored
  nmax = 0;
  voro = NULL;*/
  pstr = NULL;

}

/* ---------------------------------------------------------------------- */

FixMembraneGran::~FixMembraneGran()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);
  atom->delete_callback(id,1);

  // delete locally stored arrays

  delete [] pstr;
}

/* ---------------------------------------------------------------------- */

int FixMembraneGran::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMembraneGran::init()
{
  if (domain->dimension != 3)
    error->all(FLERR,"Fix membrane/gran only coded for 3d systems at the moment");

  dt = update->dt;

  // check variables for target pressure

  if (pstr) {
    pvar = input->variable->find(pstr);
    if (pvar < 0)
      error->all(FLERR,"Variable name for fix membrane/gran does not exist");
    if (!input->variable->equalstyle(pvar)) error->all(FLERR,"Variable for fix membrane/gran is invalid style");
  }

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixMembraneGran::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
}

/* ---------------------------------------------------------------------- */

void FixMembraneGran::post_force(int vflag)
{

  //FROM compute! Modify cause some things not needed!


  int i,maxk,kk,maxv,numv,facei,where,ivor;
  double areax,areay,areaz;

  //invoked_peratom = update->ntimestep;???

  // grow per atom array if necessary

  int nlocal = atom->nlocal;
  /*if (nlocal > nmax) {
    memory->destroy(voro);
    nmax = atom->nmax;
    memory->create(voro,nmax,size_peratom_cols,"laguerre/atom:voro");
    array_atom = voro;
  }*///remove, voro not needed

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

  // pass coordinates for local and ghost atoms to voro++

  double **x = atom->x;
  double *radius = atom->radius;
  double **f = atom->f;
  double maxradi=0.0;//maxradi not needed
  for (i = 0; i < nall; i++) {//use groupbit???
     conp.put(i,x[i][0],x[i][1],x[i][2],radius[i]);  //could maybe pass on less atoms
     if (radius[i]>maxradi) maxradi=radius[i];
  }

  container_poly coni(xleft-2*maxradi,xright+2*maxradi,yleft-2*maxradi,yright+2*maxradi,zleft-2*maxradi,zright+2*maxradi,
                m,m,m,false,false,false,8);

  for (i = 0; i < nall; i++) {
     coni.put(i,x[i][0],x[i][1],x[i][2],radius[i]);
  }//maybe remve coni if nplane function works??

  int idimag = nall + 1; //id for imaginary particles
  double coorimag;

  // invoke voro++ and fetch results for owned atoms in group

  int *mask = atom->mask;
  std::vector<int> neigh,iverts,neighimag;
  std::vector<double> areas,vnormal;
  std::vector<double> xyzverts;

 FILE *fpp,*fpp1,*fpp2,*fpp0;

 fpp = fopen("/home/gmarketo/LammpsWork/ForgitLAMMPS/ForGMmain/Membrane/FixOldEdges.txt","w");
 fpp1 = fopen("/home/gmarketo/LammpsWork/ForgitLAMMPS/ForGMmain/Membrane/FixNewEdges.txt","w");
 fpp2 = fopen("/home/gmarketo/LammpsWork/ForgitLAMMPS/ForGMmain/Membrane/FixAddForces.txt","w");
 fpp0 = fopen("/home/gmarketo/LammpsWork/ForgitLAMMPS/ForGMmain/Membrane/Areas0Normal.txt","w");
 fprintf(fpp2,"%%i,whichmem,neighimag[kk],fx,fy,fz,area[kk],nx,ny,nz\n");
 fprintf(fpp1,"%% i,x[i][0],x[i][1],x[i][2],neighimag[kk],thisis,x[thisis][0],x[thisis][1],x[thisis][2],whichmem,    VORONOI VERTICES\n");
 fprintf(fpp,"%% i,x[i][0],x[i][1],x[i][2],neigh[kk],idimag\n");
 fprintf(fpp0,"%%Vertices with zero normals\n");

 int memflag,verflag;
 memflag = verflag = 1;

  voronoicell_neighbor c;
  c_loop_all cl(conp);
  if (cl.start()) do if (conp.compute_cell(c,cl)) {
    i = cl.pid();
    if (i < nlocal && (mask[i] & groupbit)) {
      if (memflag == 1) {
        c.neighbors(neigh);
        maxk = neigh.size();
        ivor =3;
        //loop through neigh and add imaginary atoms if id<0 (edge particle)
        for (kk = 0; kk < maxk; kk++) {
          if (neigh[kk]<0) {

            // Add imaginary atoms to coni!
            if (neigh[kk]==-1) {
              if (xleft+distx>x[i][0]) {
                idimag=nall+1+i; // idimag assigned in such a way so that I can use integer division and modulo to get extra information
                coorimag=x[i][0]-2*radius[i];
                coni.put(idimag,coorimag,x[i][1],x[i][2],radius[i]);// check that imaginary grains not added twice!
              }
            }
            if (neigh[kk]==-2) {
              if (distx+x[i][0]>xright) {
                idimag=(nall+1)*2+i;
                coorimag=x[i][0]+2*radius[i];
                coni.put(idimag,coorimag,x[i][1],x[i][2],radius[i]);
              }
            }
            if (neigh[kk]==-3) {
              if (yleft+disty>x[i][1]) {
                idimag=(nall+1)*3+i;
                coorimag=x[i][1]-2*radius[i];
                coni.put(idimag,x[i][0],coorimag,x[i][2],radius[i]);
              }
            }
            if (neigh[kk]==-4) {
              if (disty+x[i][1]>yright) {
                idimag=(nall+1)*4+i;
                coorimag=x[i][1]+2*radius[i];
                coni.put(idimag,x[i][0],coorimag,x[i][2],radius[i]);
              }
            }
            if (neigh[kk]==-5) {
              if (zleft+distz>x[i][2]) {
                idimag=(nall+1)*5+i;
                coorimag=x[i][2]-2*radius[i];
                coni.put(idimag,x[i][0],x[i][1],coorimag,radius[i]);
              }
            }
            if (neigh[kk]==-6) {
              if (distz+x[i][2]>zright) {
                idimag=(nall+1)*6+i;
                coorimag=x[i][2]+2*radius[i];
                coni.put(idimag,x[i][0],x[i][1],coorimag,radius[i]);
              }
            }
        fprintf(fpp,"%d  %f %f %f %d %d\n",i,x[i][0],x[i][1],x[i][2],neigh[kk],idimag);
          }
        }
      }

    }
  } while (cl.inc());

 fclose(fpp);
 int thisis,whichmem;
 double magnorminv,nx,ny,nz,fx,fy,fz;

// Loop over the container that also has in it the 'imaginary grains'
  voronoicell_neighbor cimag;
  c_loop_all climag(coni);
  if (climag.start()) do if (coni.compute_cell(cimag,climag)) {
    i = climag.pid();
    if (i < nlocal && (mask[i] & groupbit)) {
      if (memflag == 1) {
        areax = 0.0;
        areay = 0.0;
        areaz = 0.0;
        fx = fy = fz = 0.0;
        cimag.neighbors(neighimag);
        maxk = neighimag.size();
        ivor =3;
        //loop through neighimag and get areas id<=-7
        for (kk = 0; kk < maxk; kk++) {
          if (neighimag[kk]>=nall + 1) {
            // first print out vertices information
            thisis=neighimag[kk]%(nall + 1); // the id of grain
            whichmem=neighimag[kk]/(nall + 1); // the id of membrane
            fprintf(fpp1,"%d %f %f %f %d %d %f %f %f %d ",i,x[i][0],x[i][1],x[i][2],neighimag[kk],thisis,x[thisis][0],x[thisis][1],x[thisis][2],whichmem);
            if (verflag == 1) { // the vertices of the kk'th face are needed
              cimag.face_vertices(iverts);
              cimag.vertices(x[i][0],x[i][1],x[i][2],xyzverts);
              maxv = iverts.size();
              where = 0; // the location of the number of vertices for a face in the iverts vector
              for (facei = 0; facei < kk; facei++) {
                 numv = iverts[where];
                 where = where + numv + 1;
              } //where now stores where to find the number of vertices
              numv = iverts[where]; // for the kk'th face
              for (int mm = 0; mm < numv; mm++) { 
                 fprintf(fpp1,"%f %f %f ",xyzverts[3*iverts[where+mm+1]],xyzverts[3*iverts[where+mm+1]+1],xyzverts[3*iverts[where+mm+1]+2]);
                 if (mm == numv-1) {
                  fprintf(fpp1,"%f %f %f\n",xyzverts[3*iverts[where+1]],xyzverts[3*iverts[where+1]+1],xyzverts[3*iverts[where+1]+2]);
                 }
              }
            }
            // then do the force calculation
            cimag.face_areas(areas);
            cimag.normals(vnormal);
            int sizeA= areas.size();
            int sizen= vnormal.size();
            if (sizeA*3!=sizen) printf("size of areas =%d and size of normals = %d\n",sizeA,sizen);
            magnorminv=1.0/sqrt(vnormal[kk*3]*vnormal[kk*3]+vnormal[kk*3+1]*vnormal[kk*3+1]+vnormal[kk*3+2]*vnormal[kk*3+2]);
            if (vnormal[kk*3]==0.0 && vnormal[kk*3+1]==0.0 && vnormal[kk*3+2]==0.0) {
            printf("Found zero normal");
            fprintf(fpp0,"%d %f \n",i,areas[kk]);
            }
            else {
             if (sqrt(vnormal[kk*3]*vnormal[kk*3]+vnormal[kk*3+1]*vnormal[kk*3+1]+vnormal[kk*3+2]*vnormal[kk*3+2])==0.0) {
              printf("timestep %d and area %e and normal [%e %e %e]\n",update->ntimestep,areas[kk],vnormal[kk*3],vnormal[kk*3+1],vnormal[kk*3+2]);
              error->all(FLERR,"Divide by zero in membrane!");
             }
             nx=vnormal[kk*3]*magnorminv;// why not vnormal[kk]???
             ny=vnormal[kk*3+1]*magnorminv;
             nz=vnormal[kk*3+2]*magnorminv;
             //add leverarm of fx,fy and fz!!!!
             if ((whichmem==1 || whichmem==2) && xflag==1) {
             /*  fx+=pressurex*areas[kk]*nx; // it seems that nx,ny,nz always points outwards
              fy+=pressurex*areas[kk]*ny;
              fz+=pressurex*areas[kk]*nz;
             } else if (whichmem==2) {*/
              fx-=pressurex*areas[kk]*nx;
              fy-=pressurex*areas[kk]*ny;
              fz-=pressurex*areas[kk]*nz;
             } else if ((whichmem==3 || whichmem==4) && yflag==1) {
              /*fx+=pressurey*areas[kk]*nx;
              fy+=pressurey*areas[kk]*ny;
              fz+=pressurey*areas[kk]*nz;//Check all these again!
             } else if (whichmem==4) {*/
              fx-=pressurey*areas[kk]*nx;
              fy-=pressurey*areas[kk]*ny;
              fz-=pressurey*areas[kk]*nz;
             } else if ((whichmem==5 || whichmem==6) && zflag==1) {
              /*fx+=pressurez*areas[kk]*nx;
              fy+=pressurez*areas[kk]*ny;
              fz+=pressurez*areas[kk]*nz;
             } else if (whichmem==6) {*/
              fx-=pressurez*areas[kk]*nx;
              fy-=pressurez*areas[kk]*ny;
              fz-=pressurez*areas[kk]*nz;
             } else if (xflag*yflag*zflag!=0) error->all(FLERR,"Error in fix/membrane/gran - what is whichmem?");
             fprintf(fpp2,"%d %d %d %e %e %e %e %e %e %e\n",i,whichmem,neighimag[kk],fx,fy,fz,areas[kk],nx,ny,nz);
            }
          }
        }
       f[i][0]+=fx;
       f[i][1]+=fy;
       f[i][2]+=fz;
      }
    } else if (i >= nlocal) {
        //printf("i = %d and nlocal= %d\n",i,nlocal);
        //error->warning(FLERR,"Why are there grains with id larger than nlocal in fix/membrane/gran?");
    }
  } while (climag.inc());


fclose(fpp1);
fclose(fpp2);
fclose(fpp0);



/*FROM FIX - useful?
  // virial setup
  if (vflag) v_setup(vflag);
  else evflag = 0;

  fwall[0] = fwall[1] = fwall[2] = 0.0; //per-processor force// fwall_all[0] = fwall_all[1] = fwall_all[2] = 0.0;

  shearupdate = 1;
  if (update->setupflag) shearupdate = 0;*/


}

/* ---------------------------------------------------------------------- */

void FixMembraneGran::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixMembraneGran::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = nmax * sizeof(int);
  bytes += 3*nmax * sizeof(double);
  //add vatom memory!
  return bytes;
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixMembraneGran::maxsize_restart()
{
  return 4;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixMembraneGran::size_restart(int nlocal)
{
  return 4;
}

/* ---------------------------------------------------------------------- */

void FixMembraneGran::reset_dt()
{
  dt = update->dt;
}

/* ---------------------------------------------------------------------- 
Allows the user to do a fix_modify at the input script and change the
parameters of the fix. Only allows a few things to be modified.
Returns the number of arguments read.
------------------------------------------------------------------------- */

int FixMembraneGran::modify_param(int narg, char **arg)
{
/*    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    int argsread=0;

    if (strcmp(arg[argsread],"gamman") == 0) {// do while loop instead?
      fprintf(screen, "changed wall gamman from %f to ",gamman);
      gamman=atof(arg[argsread+1]);
      argsread+=2;
      fprintf(screen, "%f\n",gamman);
    }
    else {
       fprintf(screen,"Argument %s not yet supported\n",arg[argsread]);
       error->all(FLERR,"Illegal fix modify membrane/gran command");
    }
    return argsread;*/
}


/* ---------------------------------------------------------------------- 
A function that calculates the outputs of the fix
1st output:low wall position
2nd output:high wall position
3rd-5th outputs:forces on atoms by wall
------------------------------------------------------------------------- */

double FixMembraneGran::compute_vector(int n)
{

 /* if (n == 0) return lo;
  if (n == 1) return hi;
  // only sum across procs one time?? //if (eflag == 0) {??
  MPI_Allreduce(fwall,fwall_all,3,MPI_DOUBLE,MPI_SUM,world);
  if (update->ntimestep == time_origin) error->warning(FLERR,"Force output by fix_wall_gran not computed properly");
  if (n>4) error->all(FLERR,"Illegal fix_wall_gran output");
  return fwall_all[n-2];*/
}

/* ---------------------------------------------------------------------- 
Adds the wall forces to the per-atom stress accessed through
compute stress/atom
------------------------------------------------------------------------- */

void FixMembraneGran::ev_tally_membrane(int i, double fx, double fy, double fz,
			double dx, double dy, double dz, double radi)
{

    double volume;

    //calculate stresses and assign them to vatom array
    //if (vflag_either) {
        if (vflag_atom) {
          //if (i < nlocal) {//i already lower than nlocal
            //volume of the particle
            if(domain->dimension == 3)
                volume = 4.0*MY_PI/3.0 * radi*radi*radi; //sphere
            else if (domain->dimension == 2)
                volume = MY_PI * radi*radi; //disk
            else
                error->all(FLERR,"Cannot read correct dimension");

            vatom[i][0] += dx * fx / volume;
            vatom[i][1] += dy * fy / volume;
            vatom[i][2] += dz * fz / volume;
            vatom[i][3] += dx * fy / volume;
            vatom[i][4] += dx * fz / volume;
            vatom[i][5] += dy * fz / volume;
          //}
        }
    //}

}

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

#include <stdlib.h>
#include <string.h>
#include "compute_stress_atom.h"
#include "atom.h"
#include "update.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "domain.h" //~ Added these two header files [KH - 1 November 2011]
#include <math.h>

using namespace LAMMPS_NS;

enum{NOBIAS,BIAS};

/* ---------------------------------------------------------------------- */

ComputeStressAtom::ComputeStressAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  id_temp(NULL), stress(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal compute stress/atom command");

  peratom_flag = 1;
  size_peratom_cols = 6;
  pressatomflag = 1;
  timeflag = 1;
  comm_reverse = 6;

  // store temperature ID used by stress computation
  // insure it is valid for temperature computation

  if (strcmp(arg[3],"NULL") == 0) id_temp = NULL;
  else {
    int n = strlen(arg[3]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[3]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute stress/atom temperature ID");
    if (modify->compute[icompute]->tempflag == 0)
      error->all(FLERR,
		 "Compute stress/atom temperature ID does not "
                 "compute temperature");
  }

  // process optional args

  if (narg == 4) {
    keflag = 1;
    pairflag = 1;
    bondflag = angleflag = dihedralflag = improperflag = 1;
    kspaceflag = 1;
    fixflag = 1;
  } else {
    keflag = 0;
    pairflag = 0;
    bondflag = angleflag = dihedralflag = improperflag = 0;
    kspaceflag = 0;
    fixflag = 0;
    int iarg = 4;
    while (iarg < narg) {
      if (strcmp(arg[iarg],"ke") == 0) keflag = 1;
      else if (strcmp(arg[iarg],"pair") == 0) pairflag = 1;
      else if (strcmp(arg[iarg],"bond") == 0) bondflag = 1;
      else if (strcmp(arg[iarg],"angle") == 0) angleflag = 1;
      else if (strcmp(arg[iarg],"dihedral") == 0) dihedralflag = 1;
      else if (strcmp(arg[iarg],"improper") == 0) improperflag = 1;
      else if (strcmp(arg[iarg],"kspace") == 0) kspaceflag = 1;
      else if (strcmp(arg[iarg],"fix") == 0) fixflag = 1;
      else if (strcmp(arg[iarg],"virial") == 0) {
        pairflag = 1;
        bondflag = angleflag = dihedralflag = improperflag = 1;
        kspaceflag = fixflag = 1;
      } else error->all(FLERR,"Illegal compute stress/atom command");
      iarg++;
    }
  }

  //~ Added the following warning [KH - 11 December 2012]
  if (force->pair_match("gran",0) && keflag)
    error->warning(FLERR,"Spurious kinetic energy terms may be included by ComputeStressAtom unless additional arguments are provided");

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeStressAtom::~ComputeStressAtom()
{
  delete [] id_temp;
  memory->destroy(stress);
}

/* ---------------------------------------------------------------------- */

void ComputeStressAtom::init()
{
  // set temperature compute, must be done in init()
  // fixes could have changed or compute_modify could have changed it

  if (id_temp) {
    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute stress/atom temperature ID");
    temperature = modify->compute[icompute];
    if (temperature->tempbias) biasflag = BIAS;
    else biasflag = NOBIAS;
  } else biasflag = NOBIAS;
}

/* ---------------------------------------------------------------------- */

void ComputeStressAtom::compute_peratom()
{
  int i,j;
  double onemass;

  invoked_peratom = update->ntimestep;

  //~ Modified to avoid conflict with FixMultistress [KH - 13 December 2012]
  if (update->vflag_atom != invoked_peratom && !force->pair_match("gran",0))
    error->all(FLERR,"Per-atom virial was not tallied on needed timestep");

  // grow local stress array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(stress);
    nmax = atom->nmax;
    memory->create(stress,nmax,6,"stress/atom:stress");
    array_atom = stress;
  }

  // npair includes ghosts if either newton flag is set
  //   b/c some bonds/dihedrals call pair::ev_tally with pairwise info
  // nbond includes ghosts if newton_bond is set
  // ntotal includes ghosts if either newton flag is set
  // KSpace includes ghosts if tip4pflag is set

  int nlocal = atom->nlocal;
  int npair = nlocal;
  int nbond = nlocal;
  int ntotal = nlocal;
  int nkspace = nlocal;
  if (force->newton) npair += atom->nghost;
  if (force->newton_bond) nbond += atom->nghost;
  if (force->newton) ntotal += atom->nghost;
  if (force->kspace && force->kspace->tip4pflag) nkspace += atom->nghost;

  // clear local stress array

  for (i = 0; i < ntotal; i++)
    for (j = 0; j < 6; j++)
      stress[i][j] = 0.0;

  // add in per-atom contributions from each force

  if (pairflag && force->pair) {
    double **vatom = force->pair->vatom;
    for (i = 0; i < npair; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (bondflag && force->bond) {
    double **vatom = force->bond->vatom;
    for (i = 0; i < nbond; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (angleflag && force->angle) {
    double **vatom = force->angle->vatom;
    for (i = 0; i < nbond; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (dihedralflag && force->dihedral) {
    double **vatom = force->dihedral->vatom;
    for (i = 0; i < nbond; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (improperflag && force->improper) {
    double **vatom = force->improper->vatom;
    for (i = 0; i < nbond; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  if (kspaceflag && force->kspace) {
    double **vatom = force->kspace->vatom;
    for (i = 0; i < nkspace; i++)
      for (j = 0; j < 6; j++)
        stress[i][j] += vatom[i][j];
  }

  // add in per-atom contributions from relevant fixes
  // skip if vatom = NULL
  // possible during setup phase if fix has not initialized its vatom yet
  // e.g. fix ave/spatial defined before fix shake,
  //   and fix ave/spatial uses a per-atom stress from this compute as input

  if (fixflag) {
    for (int ifix = 0; ifix < modify->nfix; ifix++)
      if (modify->fix[ifix]->virial_flag) {
        double **vatom = modify->fix[ifix]->vatom;
        if (vatom)
          for (i = 0; i < nlocal; i++)
            for (j = 0; j < 6; j++)
              stress[i][j] += vatom[i][j];
      }
  }

  // communicate ghost virials between neighbor procs

  if (force->newton || (force->kspace && force->kspace->tip4pflag))
    comm->reverse_comm_compute(this);

  // zero virial of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) {
      stress[i][0] = 0.0;
      stress[i][1] = 0.0;
      stress[i][2] = 0.0;
      stress[i][3] = 0.0;
      stress[i][4] = 0.0;
      stress[i][5] = 0.0;
    }

  // include kinetic energy term for each atom in group
  // apply temperature bias is applicable
  // mvv2e converts mv^2 to energy

  if (keflag) {
    double **v = atom->v;
    double *mass = atom->mass;
    double *rmass = atom->rmass;
    int *type = atom->type;
    double mvv2e = force->mvv2e;

    if (biasflag == NOBIAS) {
      if (rmass) {
	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    onemass = mvv2e * rmass[i];
	    stress[i][0] += onemass*v[i][0]*v[i][0];
	    stress[i][1] += onemass*v[i][1]*v[i][1];
	    stress[i][2] += onemass*v[i][2]*v[i][2];
	    stress[i][3] += onemass*v[i][0]*v[i][1];
	    stress[i][4] += onemass*v[i][0]*v[i][2];
	    stress[i][5] += onemass*v[i][1]*v[i][2];
	  }

      } else {
	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    onemass = mvv2e * mass[type[i]];
	    stress[i][0] += onemass*v[i][0]*v[i][0];
	    stress[i][1] += onemass*v[i][1]*v[i][1];
	    stress[i][2] += onemass*v[i][2]*v[i][2];
	    stress[i][3] += onemass*v[i][0]*v[i][1];
	    stress[i][4] += onemass*v[i][0]*v[i][2];
	    stress[i][5] += onemass*v[i][1]*v[i][2];
	  }
      }

    } else {

      // invoke temperature if it hasn't been already
      // this insures bias factor is pre-computed

      if (keflag && temperature->invoked_scalar != update->ntimestep)
	temperature->compute_scalar();

      if (rmass) {
	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    temperature->remove_bias(i,v[i]);
	    onemass = mvv2e * rmass[i];
	    stress[i][0] += onemass*v[i][0]*v[i][0];
	    stress[i][1] += onemass*v[i][1]*v[i][1];
	    stress[i][2] += onemass*v[i][2]*v[i][2];
	    stress[i][3] += onemass*v[i][0]*v[i][1];
	    stress[i][4] += onemass*v[i][0]*v[i][2];
	    stress[i][5] += onemass*v[i][1]*v[i][2];
	    temperature->restore_bias(i,v[i]);
	  }

      } else {
	for (i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    temperature->remove_bias(i,v[i]);
	    onemass = mvv2e * mass[type[i]];
	    stress[i][0] += onemass*v[i][0]*v[i][0];
	    stress[i][1] += onemass*v[i][1]*v[i][1];
	    stress[i][2] += onemass*v[i][2]*v[i][2];
	    stress[i][3] += onemass*v[i][0]*v[i][1];
	    stress[i][4] += onemass*v[i][0]*v[i][2];
	    stress[i][5] += onemass*v[i][1]*v[i][2];
	    temperature->restore_bias(i,v[i]);
	  }
      }
    }
  }

  // convert to stress*volume units = -pressure*volume
  /*~ The minus sign enforces a 'compression positive' sign
    convention for stress [KH - 4 August 2014]*/
  double nktv2p = -force->nktv2p;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      stress[i][0] *= nktv2p;
      stress[i][1] *= nktv2p;
      stress[i][2] *= nktv2p;
      stress[i][3] *= nktv2p;
      stress[i][4] *= nktv2p;
      stress[i][5] *= nktv2p;
    }
}

/* ---------------------------------------------------------------------- */

int ComputeStressAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = stress[i][0];
    buf[m++] = stress[i][1];
    buf[m++] = stress[i][2];
    buf[m++] = stress[i][3];
    buf[m++] = stress[i][4];
    buf[m++] = stress[i][5];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeStressAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    stress[j][0] += buf[m++];
    stress[j][1] += buf[m++];
    stress[j][2] += buf[m++];
    stress[j][3] += buf[m++];
    stress[j][4] += buf[m++];
    stress[j][5] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeStressAtom::memory_usage()
{
  double bytes = nmax*6 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   The purpose of this function is to calculate the mean stresses which
  are used in the servo control algorithm. Eq. 9.25 from COS book was used
  [KH - 1 November 2011]
------------------------------------------------------------------------- */

double *ComputeStressAtom::array_export()
{
  /*~ The static specifier ensures that the means exist for the duration of
    the program. Important to reinitialise the elements to 0 each time the
    function is called [KH - 2 November 2011]*/
  static double means[6];
  for (int i = 0; i < 6; i++)
    means[i] = 0.0;

  double PI = 4.0*atan(1.0);
  double pvolume; //~ Volume of any individual particle
  int *mask = atom->mask;
  double *radius = atom->radius;
  int nlocal = atom->nlocal;
 
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (domain->dimension == 3) {
	pvolume = (4.0*PI/3.0)*radius[i]*radius[i]*radius[i];
      } else pvolume = PI*radius[i]*radius[i];
      
      for (int j = 0; j < 6; j++)
 	means[j] += stress[i][j]*pvolume;
    }
  
  double totalvolume = 1.0; //~ Total volume enclosed by bounding box
  
  // modified to include presence of wall boundary [MO - 19 Aug 2015]
  for (int i = 0; i < domain->dimension; i++) {
    if (domain->periodicity[i] == 0 ) {
      totalvolume *= (domain->w_boxhi[i] - domain->w_boxlo[i]);
    }
    else totalvolume *= (domain->boxhi[i] - domain->boxlo[i]);
  }
  
  for (int i = 0; i < 6; i++)
    means[i] /= totalvolume;

  return means;
}

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

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_gran_hooke_history.h"
#include "atom.h"
#include "atom_vec.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "fix_shear_history.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "math_extra.h" //~ For rolling resistance [KH - 23 October 2013]
#include "fix_old_omega.h" //~ And these three too [KH - 6 November 2013]
#include "math_special.h"
#include "mpi.h"
#include "compute_energy_gran.h" //~ For energy tracing [KH - 19 February 2014]

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairGranHookeHistory::PairGranHookeHistory(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 1;
  no_virial_fdotr_compute = 1;
  history = 1;
  fix_history = NULL;

  /*~ Modified for rolling resistance model. The last 27(!) entries 
    in svector will be unused if rolling resistance model is inactive
    [KH - 29 July 2014]*/
  //~ Added four more for energy tracing [KH - 6 March 2014]
  single_extra = 45;
  svector = new double[single_extra]; //~ Changed to single_extra [KH - 25 October 2013]

  computeflag = 0;
  neighprev = 0;

  /*~ Initialise two integers used to limit the numbers of warnings 
    about failures to calculate contact stiffnesses in the rolling 
    resistance model [KH - 5 November 2013]*/
  lastwarning[0] = lastwarning[1] = -1000000;

  nmax = 0;
  mass_rigid = NULL;

  // set comm size needed by this Pair if used with fix rigid

  comm_forward = 1;

  /*~ Initialise the accumulated energy terms to zero. For the
    linear contact model, the shear strain is not calculated
    cumulatively, but it makes no difference to zero it here
    anyway [KH - 27 February 2014]*/
  dissipfriction = shearstrain = 0.0;
}

/* ---------------------------------------------------------------------- */

PairGranHookeHistory::~PairGranHookeHistory()
{
  delete [] svector;
  if (fix_history) modify->delete_fix("SHEAR_HISTORY");

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    delete [] onerad_dynamic;
    delete [] onerad_frozen;
    delete [] maxrad_dynamic;
    delete [] maxrad_frozen;
  }

  /*~ Delete the fix allocated for the rolling resistance model [KH -
    24 October 2013]*/
  if (rolling) modify->delete_fix("pair_oldomega");

  memory->destroy(mass_rigid);
}

/* ---------------------------------------------------------------------- */

void PairGranHookeHistory::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz,fx,fy,fz;
  double radi,radj,radsum,rsq,r,rinv,rsqinv;
  double vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double wr1,wr2,wr3;
  double vtr1,vtr2,vtr3,vrel;
  double mi,mj,meff,damp,ccel,tor1,tor2,tor3;
  double fn,fs,fs1,fs2,fs3;
  double shsqmag,shsqnew,shratio,rsht;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *touch,**firsttouch;
  double *shear,*allshear,**firstshear;

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  computeflag = 1;
  int shearupdate = 1;
  if (update->setupflag) shearupdate = 0;

  // update rigid body info for owned & ghost atoms if using FixRigid masses
  // body[i] = which body atom I is in, -1 if none
  // mass_body = mass of each rigid body

  if (fix_rigid && neighbor->ago == 0) {
    int tmp;
    int *body = (int *) fix_rigid->extract("body",tmp);
    double *mass_body = (double *) fix_rigid->extract("masstotal",tmp);
    if (atom->nmax > nmax) {
      memory->destroy(mass_rigid);
      nmax = atom->nmax;
      memory->create(mass_rigid,nmax,"pair:mass_rigid");
    }
    int nlocal = atom->nlocal;
    for (i = 0; i < nlocal; i++)
      if (body[i] >= 0) mass_rigid[i] = mass_body[body[i]];
      else mass_rigid[i] = 0.0;
    comm->forward_comm_pair(this);
  }

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double deltan,cri,crj;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firsttouch = listgranhistory->firstneigh;
  firstshear = listgranhistory->firstdouble;

  /*~ The following piece of code was added to determine whether or not
    any periodic boundaries, if present, are moving either via fix_
    multistress or fix_deform [KH - 9 November 2011]*/
  int velmapflag = 0;

  if (domain->xperiodic || domain->yperiodic || domain->zperiodic) {
    for (int q = 0; q < modify->nfix; q++)
      if (strcmp(modify->fix[q]->style,"multistress") == 0) {
	ierates = modify->fix[q]->param_export();
	velmapflag = 1;
	break;
      }
  
    if (velmapflag == 0)
      for (int q = 0; q < modify->nfix; q++) {
	if (strcmp(modify->fix[q]->style,"deform") == 0) {
	  ierates = modify->fix[q]->param_export();
	  velmapflag = 1;
	  break;
	}
      }
  }

  /*~ Ascertain whether or not energy tracing is active by checking
    for the presence of compute energy/gran and also the trace_energy
    flag which indicates per-contact tracing is active. If so, check if the
    tracked terms include those calculated in this pairstyle. The
    energy terms are updated only if necessary for efficiency.
    [KH - 6 March 2014]*/
  int pairenergy = trace_energy;
  if (!pairenergy)
    for (int q = 0; q < modify->ncompute; q++)
      if (strcmp(modify->compute[q]->style,"energy/gran") == 0) {
	pairenergy = ((ComputeEnergyGran *) modify->compute[q])->pairenergy;
	break;
      }

  //~ Initialise the strain energy terms to zero
  normalstrain = shearstrain = 0.0;

  /*~ The number of shear quantities is 18 if rolling resistance
    is active [KH - 29 July 2014]*/
  /*~ Another 4 shear quantities were added for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 3 + 15*rolling + 4*trace_energy;

  //~ Use tags to consider contacts only once [KH - 28 February 2014]
  tagint *tag = atom->tag; 

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    radi = radius[i];
    touch = firsttouch[i];
    allshear = firstshear[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      radj = radius[j];
      radsum = radi + radj;

      if (rsq >= radsum*radsum) {

        // unset non-touching neighbors

        touch[jj] = 0;
        shear = &allshear[numshearquants*jj];
	for (int q = 0; q < numshearquants; q++)
	  shear[q] = 0.0; //~ Added the 'for' loop [KH - 23 October 2013]

      } else {
        r = sqrt(rsq);
        rinv = 1.0/r;
        rsqinv = 1.0/rsq;
        deltan = radsum-r;
        cri = radi-0.5*deltan;
        crj = radj-0.5*deltan;

        // relative translational velocity

        vr1 = v[i][0] - v[j][0];
        vr2 = v[i][1] - v[j][1];
        vr3 = v[i][2] - v[j][2];

	/*~ These relative velocity components have to be updated for the
	  periodic boundaries [KH - 14 November 2011]*/
	if (velmapflag == 1) {
	  vr1 += (ierates[0]*delx + ierates[3]*dely + ierates[4]*delz);
	  vr2 += (ierates[3]*delx + ierates[1]*dely + ierates[5]*delz);
	  vr3 += (ierates[4]*delx + ierates[5]*dely + ierates[2]*delz);
	}

        // normal component

        vnnr = vr1*delx + vr2*dely + vr3*delz;
        vn1 = delx*vnnr * rsqinv;
        vn2 = dely*vnnr * rsqinv;
        vn3 = delz*vnnr * rsqinv;

        // tangential component

        vt1 = vr1 - vn1;
        vt2 = vr2 - vn2;
        vt3 = vr3 - vn3;

        // relative rotational velocity

	wr1 = (cri*omega[i][0] + crj*omega[j][0]) * rinv;
	wr2 = (cri*omega[i][1] + crj*omega[j][1]) * rinv;
	wr3 = (cri*omega[i][2] + crj*omega[j][2]) * rinv;

        // meff = effective mass of pair of particles
        // if I or J part of rigid body, use body mass
        // if I or J is frozen, meff is other particle

        if (rmass) {
          mi = rmass[i];
          mj = rmass[j];
        } else {
          mi = mass[type[i]];
          mj = mass[type[j]];
        }
        if (fix_rigid) {
          if (mass_rigid[i] > 0.0) mi = mass_rigid[i];
          if (mass_rigid[j] > 0.0) mj = mass_rigid[j];
        }

        meff = mi*mj / (mi+mj);
        if (mask[i] & freeze_group_bit) meff = mj;
        if (mask[j] & freeze_group_bit) meff = mi;

        // normal forces = Hookian contact + normal velocity damping

        damp = meff*gamman*vnnr*rsqinv;
        ccel = kn*(radsum-r)*rinv - damp;

        // relative velocities

        vtr1 = vt1 - (delz*wr2-dely*wr3);
        vtr2 = vt2 - (delx*wr3-delz*wr1);
        vtr3 = vt3 - (dely*wr1-delx*wr2);
        vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
        vrel = sqrt(vrel);

        // shear history effects

        touch[jj] = 1;
        shear = &allshear[numshearquants*jj];

        if (shearupdate) {
          shear[0] += vtr1*dt;
          shear[1] += vtr2*dt;
          shear[2] += vtr3*dt;
        }
        shsqmag = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];

        // rotate shear displacements

        rsht = shear[0]*delx + shear[1]*dely + shear[2]*delz;
        rsht *= rsqinv;
        if (shearupdate) {
          shear[0] -= rsht*delx;
          shear[1] -= rsht*dely;
          shear[2] -= rsht*delz;
          shsqnew = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];
          if (shsqnew!=0.0) {
              shratio=sqrt(shsqmag/shsqnew);
              shear[0] *= shratio; // conserve shear length
              shear[1] *= shratio;
              shear[2] *= shratio;
          }
        }

        // tangential forces = shear + tangential velocity damping

        fs1 = - (kt*shear[0] + meff*gammat*vtr1);
        fs2 = - (kt*shear[1] + meff*gammat*vtr2);
        fs3 = - (kt*shear[2] + meff*gammat*vtr3);

        // rescale frictional displacements and forces if needed

        fs = sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
        fn = xmu * fabs(ccel*r);

        if (fs > fn) {
          if (fs != 0.0) {
            shear[0] = (fn/fs) * (shear[0] + meff*gammat*vtr1/kt) -
              meff*gammat*vtr1/kt;
            shear[1] = (fn/fs) * (shear[1] + meff*gammat*vtr2/kt) -
              meff*gammat*vtr2/kt;
            shear[2] = (fn/fs) * (shear[2] + meff*gammat*vtr3/kt) -
              meff*gammat*vtr3/kt;
            fs1 *= fn/fs;
            fs2 *= fn/fs;
            fs3 *= fn/fs;
          } else fs1 = fs2 = fs3 = 0.0;
        }

        // forces & torques

        fx = delx*ccel + fs1;
        fy = dely*ccel + fs2;
        fz = delz*ccel + fs3;
        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;

        tor1 = rinv * (dely*fs3 - delz*fs2);
        tor2 = rinv * (delz*fs1 - delx*fs3);
        tor3 = rinv * (delx*fs2 - dely*fs1);
	torque[i][0] -= cri*tor1;
	torque[i][1] -= cri*tor2;
	torque[i][2] -= cri*tor3;

        if (j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
	  torque[j][0] -= crj*tor1;
	  torque[j][1] -= crj*tor2;
	  torque[j][2] -= crj*tor3;
        }

	double dur[3], dus[3], localdM[3], globaldM[3]; //~ Pass by reference
	double aveshearforce, slipdisp; //~ Required for energy
	double incdissipf, nstr, sstr;

	if (j < nlocal || tag[j] < tag[i]) {//~ Consider contacts only once

	  //~ Call function for rolling resistance model [KH - 25 October 2013]
	  if (rolling && shearupdate) {
	    /*~ The first '0' indicates that the rolling_resistance function is
	      called by the compute rather than the single function*/
	    rolling_resistance(0,i,j,numshearquants,delx,dely,delz,r,rinv,ccel,
			       fn,kt,torque,shear,dur,dus,localdM,globaldM);
	  }

	  //~ Add contributions to traced energy [KH - 19 February 2014]
	  if (pairenergy) {
	    /*~ Increment the friction energy only if the slip condition
	      is invoked*/
	    if (fs > fn && fn > 0.0) {
	      slipdisp = (fs-fn)/kt;
	      aveshearforce = 0.5*(fn + fs);

	      //~ slipdisp and aveshearforce are both positive
	      incdissipf = aveshearforce*slipdisp;
	      dissipfriction += incdissipf;
	      if (trace_energy) shear[3] += incdissipf;
	    }

	    /*~ Update the strain energy terms which don't need to
	      be calculated incrementally*/
	    nstr = 0.5*kn*deltan*deltan;
	    sstr = 0.5*(fs1*fs1 + fs2*fs2 + fs3*fs3)/kt;
	    normalstrain += nstr;
	    shearstrain += sstr;

	    if (trace_energy) {
	      shear[4] = nstr;
	      shear[5] = sstr;
	    }
	  }
	}

        if (evflag) ev_tally_gran(i,j,nlocal,fx,fy,fz,x[i][0],x[i][1],x[i][2],
                                 radius[i],x[j][0],x[j][1],x[j][2],radius[j]);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGranHookeHistory::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  onerad_dynamic = new double[n+1];
  onerad_frozen = new double[n+1];
  maxrad_dynamic = new double[n+1];
  maxrad_frozen = new double[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGranHookeHistory::settings(int narg, char **arg)
{
  if (narg != 6) error->all(FLERR,"Illegal pair_style command");

  kn = force->numeric(FLERR,arg[0]);
  if (strcmp(arg[1],"NULL") == 0) kt = kn * 2.0/7.0;
  else kt = force->numeric(FLERR,arg[1]);

  gamman = force->numeric(FLERR,arg[2]);
  if (strcmp(arg[3],"NULL") == 0) gammat = 0.5 * gamman;
  else gammat = force->numeric(FLERR,arg[3]);

  xmu = force->numeric(FLERR,arg[4]);
  dampflag = force->inumeric(FLERR,arg[5]);
  if (dampflag == 0) gammat = 0.0;

  if (kn < 0.0 || kt < 0.0 || gamman < 0.0 || gammat < 0.0 ||
      xmu < 0.0 || xmu > 10000.0 || dampflag < 0 || dampflag > 1)
    error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGranHookeHistory::coeff(int narg, char **arg)
{
  if (narg > 2) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGranHookeHistory::init_style()
{
  int i;

  // error and warning checks

  if (!atom->sphere_flag)
    error->all(FLERR,"Pair granular requires atom style sphere");
  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair granular requires ghost atoms store velocity");

  /*~ Have 18 shear quantities if rolling resistance is included
    [KH - 29 July 2014]*/
  /*~ Another 4 shear quantities are needed for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 3 + 15*rolling + 4*trace_energy;

  // need a granular neigh list and optionally a granular history neigh list

  int irequest = neighbor->request(this);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->gran = 1;
  if (history) {
    irequest = neighbor->request(this);
    neighbor->requests[irequest]->id = 1;
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->granhistory = 1;
    neighbor->requests[irequest]->dnum = numshearquants;
  }

  dt = update->dt;

  // if shear history is stored:
  // check if newton flag is valid
  // if first init, create Fix needed for storing shear history

  if (history && force->newton_pair == 1)
    error->all(FLERR,
               "Pair granular with shear history requires newton pair off");

  if (history && fix_history == NULL) {
    /*~ Even though the default of 3 is sufficient without rolling
      resistance and energy tracing, it is cleaner to always 
      specify the number of shear quantities (optional last arg) 
      [KH - 6 March 2014]*/

    char **fixarg = new char*[4]; //~ Increased from 3 to 4
    fixarg[0] = (char *) "SHEAR_HISTORY";
    fixarg[1] = (char *) "all";
    fixarg[2] = (char *) "SHEAR_HISTORY";

    //~ Carry out the necessary string conversion [KH - 6 March 2014]
    char nsq[5] = {0};
    sprintf(nsq,"%i",numshearquants);
    fixarg[3] = nsq; //~ Changed this condition
    modify->add_fix(4,fixarg,1); //~ Increased to 4
    delete [] fixarg;
    fix_history = (FixShearHistory *) modify->fix[modify->nfix-1];
    fix_history->pair = this;
  }

  /*~ If rolling resistance is active, implicitly set up a fix,
    fix_old_omega, to store values of omega from the preceding
    timestep [KH - 24 October 2013]*/
  if (rolling) add_old_omega_fix();

  // check for FixFreeze and set freeze_group_bit

  for (i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"freeze") == 0) break;
  if (i < modify->nfix) freeze_group_bit = modify->fix[i]->groupbit;
  else freeze_group_bit = 0;

  // check for FixRigid so can extract rigid body masses

  fix_rigid = NULL;
  for (i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->rigid_flag) break;
  if (i < modify->nfix) fix_rigid = modify->fix[i];

  // check for FixPour and FixDeposit so can extract particle radii

  int ipour;
  for (ipour = 0; ipour < modify->nfix; ipour++)
    if (strcmp(modify->fix[ipour]->style,"pour") == 0) break;
  if (ipour == modify->nfix) ipour = -1;

  int idep;
  for (idep = 0; idep < modify->nfix; idep++)
    if (strcmp(modify->fix[idep]->style,"deposit") == 0) break;
  if (idep == modify->nfix) idep = -1;

  // set maxrad_dynamic and maxrad_frozen for each type
  // include future FixPour and FixDeposit particles as dynamic

  int itype;
  for (i = 1; i <= atom->ntypes; i++) {
    onerad_dynamic[i] = onerad_frozen[i] = 0.0;
    if (ipour >= 0) {
      itype = i;
      onerad_dynamic[i] = 
        *((double *) modify->fix[ipour]->extract("radius",itype));
    }
    if (idep >= 0) {
      itype = i;
      onerad_dynamic[i] = 
        *((double *) modify->fix[idep]->extract("radius",itype));
    }
  }

  double *radius = atom->radius;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & freeze_group_bit)
      onerad_frozen[type[i]] = MAX(onerad_frozen[type[i]],radius[i]);
    else
      onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]],radius[i]);

  MPI_Allreduce(&onerad_dynamic[1],&maxrad_dynamic[1],atom->ntypes,
                MPI_DOUBLE,MPI_MAX,world);
  MPI_Allreduce(&onerad_frozen[1],&maxrad_frozen[1],atom->ntypes,
                MPI_DOUBLE,MPI_MAX,world);
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   optional granular history list
------------------------------------------------------------------------- */

void PairGranHookeHistory::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listgranhistory = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGranHookeHistory::init_one(int i, int j)
{
  if (!allocated) allocate();

  // cutoff = sum of max I,J radii for
  // dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

  double cutoff = maxrad_dynamic[i]+maxrad_dynamic[j];
  cutoff = MAX(cutoff,maxrad_frozen[i]+maxrad_dynamic[j]);
  cutoff = MAX(cutoff,maxrad_dynamic[i]+maxrad_frozen[j]);
  return cutoff;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGranHookeHistory::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++)
      fwrite(&setflag[i][j],sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGranHookeHistory::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGranHookeHistory::write_restart_settings(FILE *fp)
{
  fwrite(&kn,sizeof(double),1,fp);
  fwrite(&kt,sizeof(double),1,fp);
  fwrite(&gamman,sizeof(double),1,fp);
  fwrite(&gammat,sizeof(double),1,fp);
  fwrite(&xmu,sizeof(double),1,fp);
  fwrite(&dampflag,sizeof(int),1,fp);

  /*~ Added energy terms. Note that the total energy dissipated by
    friction and stored as shear strain, from all procs, is 
    calculated and stored on proc 0 [KH - 28 February 2014]*/
  double gatheredf = 0.0;
  MPI_Allreduce(&dissipfriction,&gatheredf,1,MPI_DOUBLE,MPI_SUM,world);
  fwrite(&gatheredf,sizeof(double),1,fp);

  double gatheredss = 0.0;
  MPI_Allreduce(&shearstrain,&gatheredss,1,MPI_DOUBLE,MPI_SUM,world);
  fwrite(&gatheredss,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGranHookeHistory::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&kn,sizeof(double),1,fp);
    fread(&kt,sizeof(double),1,fp);
    fread(&gamman,sizeof(double),1,fp);
    fread(&gammat,sizeof(double),1,fp);
    fread(&xmu,sizeof(double),1,fp);
    fread(&dampflag,sizeof(int),1,fp);

    /*~ Added energy terms. The total energy is read to the root
    proc and is NOT broadcast to all procs as only the total summed
    across all procs is of interest [KH - 28 February 2014]*/
    fread(&dissipfriction,sizeof(double),1,fp);
    fread(&shearstrain,sizeof(double),1,fp);
  }
  MPI_Bcast(&kn,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&kt,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&gamman,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&gammat,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&xmu,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&dampflag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

void PairGranHookeHistory::reset_dt()
{
  dt = update->dt;
}

/* ---------------------------------------------------------------------- */

double PairGranHookeHistory::single(int i, int j, int itype, int jtype,
                                    double rsq,
                                    double factor_coul, double factor_lj,
                                    double &fforce)
{
  double radi,radj,radsum;
  double r,rinv,rsqinv,delx,dely,delz;
  double vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3,wr1,wr2,wr3;
  double mi,mj,meff,damp,ccel;
  double vtr1,vtr2,vtr3,vrel;
  double fs1,fs2,fs3,fs,fn;
  double deltan,cri,crj;

  double *radius = atom->radius;
  radi = radius[i];
  radj = radius[j];
  radsum = radi + radj;

  double **x = atom->x;
  tagint *tag = atom->tag; //~ Write out the atom tags

  /*~ The number of optional entries in svector for energy
    tracing and/or rolling resistance [KH - 6 March 2014]*/
  int optionalq = 4*trace_energy + 27*rolling;

  if (rsq >= radsum*radsum) {
    fforce = 0.0;
    svector[0] = svector[1] = svector[2] = svector[3] = 0.0;
    //~ The tags, radii etc. will not be zero [KH - 10 January 2013]
    svector[4] = tag[i];
    svector[5] = tag[j];
    for (int q = 0; q < 3; q++)
      svector[q+6] = x[i][q];
    svector[9] = radi;
    for (int q = 0; q < 3; q++)
      svector[q+10] = x[j][q];
    svector[13] = radj;

    /*~ Four optional entries for energy tracing [KH - 6 March 2014]
      Order: energy dissipated by friction, normal component of
      strain energy, shear component of strain energy, other
      contributions (e.g., from rolling resistance model)

      27 optional entries for the rolling resistance model 
      [KH - 29 July 2014]. Order: 
      dUr[*], accumulated dUr[*], dUs[*], accumulated dUs[*], 
      localdM[*], accumulated localdM[*], globaldM[*], accumulated 
      globaldM[*], ksbar, (K+1)_rolling, (K+1)_twisting*/
    if (optionalq > 0)
      for (int q = 0; q < optionalq; q++) svector[q+14] = 0.0;

    return 0.0;
  }

  r = sqrt(rsq);
  rinv = 1.0/r;
  rsqinv = 1.0/rsq;
  deltan = radsum-r;
  cri = radi-0.5*deltan;
  crj = radj-0.5*deltan; 

  // relative translational velocity

  double **v = atom->v;
  vr1 = v[i][0] - v[j][0];
  vr2 = v[i][1] - v[j][1];
  vr3 = v[i][2] - v[j][2];

  // normal component

  delx = x[i][0] - x[j][0];
  dely = x[i][1] - x[j][1];
  delz = x[i][2] - x[j][2];

  //~ Add in the periodic boundary updating code [KH - 13 December 2012]
  if ((domain->xperiodic || domain->yperiodic || domain->zperiodic) &&
      domain->box_change == 1) {
    vr1 += (ierates[0]*delx + ierates[3]*dely + ierates[4]*delz);
    vr2 += (ierates[3]*delx + ierates[1]*dely + ierates[5]*delz);
    vr3 += (ierates[4]*delx + ierates[5]*dely + ierates[2]*delz);
  }

  vnnr = vr1*delx + vr2*dely + vr3*delz;
  vn1 = delx*vnnr * rsqinv;
  vn2 = dely*vnnr * rsqinv;
  vn3 = delz*vnnr * rsqinv;

  // tangential component

  vt1 = vr1 - vn1;
  vt2 = vr2 - vn2;
  vt3 = vr3 - vn3;

  // relative rotational velocity

  double **omega = atom->omega;
  wr1 = (cri*omega[i][0] + crj*omega[j][0]) * rinv;
  wr2 = (cri*omega[i][1] + crj*omega[j][1]) * rinv;
  wr3 = (cri*omega[i][2] + crj*omega[j][2]) * rinv;

  // meff = effective mass of pair of particles
  // if I or J part of rigid body, use body mass
  // if I or J is frozen, meff is other particle

  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;

  if (rmass) {
    mi = rmass[i];
    mj = rmass[j];
  } else {
    mi = mass[type[i]];
    mj = mass[type[j]];
  }
  if (fix_rigid) {
    // NOTE: insure mass_rigid is current for owned+ghost atoms?
    if (mass_rigid[i] > 0.0) mi = mass_rigid[i];
    if (mass_rigid[j] > 0.0) mj = mass_rigid[j];
  }

  meff = mi*mj / (mi+mj);
  if (mask[i] & freeze_group_bit) meff = mj;
  if (mask[j] & freeze_group_bit) meff = mi;

  // normal forces = Hookian contact + normal velocity damping

  damp = meff*gamman*vnnr*rsqinv;
  ccel = kn*(radsum-r)*rinv - damp;

  // relative velocities

  vtr1 = vt1 - (delz*wr2-dely*wr3);
  vtr2 = vt2 - (delx*wr3-delz*wr1);
  vtr3 = vt3 - (dely*wr1-delx*wr2);
  vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
  vrel = sqrt(vrel);

  // shear history effects
  // neighprev = index of found neigh on previous call
  // search entire jnum list of neighbors of I for neighbor J
  // start from neighprev, since will typically be next neighbor
  // reset neighprev to 0 as necessary

  int jnum = list->numneigh[i];
  int *touch = list->listgranhistory->firstneigh[i];
  double *allshear = list->listgranhistory->firstdouble[i];

  for (int jj = 0; jj < jnum; jj++) {
    neighprev++;
    if (neighprev >= jnum) neighprev = 0;
    if (touch[neighprev] == j) break;
  }

  /*~ The number of shear quantities is 18 if rolling resistance
    is active [KH - 29 July 2014]*/
  /*~ Another 4 shear quantities were added for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 3 + 15*rolling + 4*trace_energy;
  double *shear = &allshear[numshearquants*neighprev];

  // rotate shear displacements - not needed- shear already updated by compute!

  // tangential forces = shear + tangential velocity damping

  fs1 = - (kt*shear[0] + meff*gammat*vtr1);
  fs2 = - (kt*shear[1] + meff*gammat*vtr2);
  fs3 = - (kt*shear[2] + meff*gammat*vtr3);

  // rescale frictional displacements and forces if needed

  fs = sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
  fn = xmu * fabs(ccel*r);

  if (fs > fn) {
    if (fs != 0.0) {
      fs1 *= fn/fs;
      fs2 *= fn/fs;
      fs3 *= fn/fs;
      fs *= fn/fs;
    } else fs1 = fs2 = fs3 = fs = 0.0;
  }

  //~ Call function for rolling resistance model [KH - 30 October 2013]
  double dur[3], dus[3], localdM[3], globaldM[3]; //~ Pass by reference
  double **torque = atom->torque;
  if (rolling) {
    /*~ The first '1' indicates that the rolling_resistance function is
      called by the single function rather than the compute*/
    rolling_resistance(1,i,j,numshearquants,delx,dely,delz,r,rinv,ccel,
		       fn,kt,torque,shear,dur,dus,localdM,globaldM);
  }

  /*~ Some of the following are included only for convenience as
    the data could instead be obtained from a dump of the sphere
    coordinates [KH - 13 December 2011]*/
  fforce = ccel;
  svector[0] = fs1;
  svector[1] = fs2;
  svector[2] = fs3;
  svector[3] = ccel;
  svector[4] = tag[i];
  svector[5] = tag[j];
  for (int q = 0; q < 3; q++)
    svector[q+6] = x[i][q];
  svector[9] = radi;
  for (int q = 0; q < 3; q++)
    svector[q+10] = x[j][q];
  svector[13] = radj;

  //~ Add for energy tracing [KH - 6 March 2014]
  int nq = 4*trace_energy;
  if (trace_energy)
    for (int q = 0; q < 4; q++)
      svector[q+14] = shear[q+3];

  //~ Add for the rolling resistance model [KH - 30 October 2013]
  if (rolling) {
    for (int q = 0; q < 3; q++) {
      svector[q+nq+14] = dur[q];
      svector[q+nq+17] = shear[q+nq+3];
      svector[q+nq+20] = dus[q];
      svector[q+nq+23] = shear[q+nq+6];
      svector[q+nq+26] = localdM[q];
      svector[q+nq+29] = shear[q+nq+9];
      svector[q+nq+32] = globaldM[q];
      svector[q+nq+35] = shear[q+nq+12];
    }
    svector[nq+38] = shear[numshearquants-3]; //~ ksbar
    svector[nq+39] = shear[numshearquants-2]; //~ stored rolling (kappa+1)
    svector[nq+40] = shear[numshearquants-1]; //~ stored twisting (kappa+1)
  }

  return 0.0;
}

/* ---------------------------------------------------------------------- */

int PairGranHookeHistory::pack_forward_comm(int n, int *list, double *buf, 
                                            int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = mass_rigid[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairGranHookeHistory::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
    mass_rigid[i] = buf[m++];
}

/* ---------------------------------------------------------------------- */

void *PairGranHookeHistory::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"computeflag") == 0) return (void *) &computeflag;
  else if (strcmp(str,"list") == 0) return (void *) list;
  else if (strcmp(str,"listgranhistory") == 0) return (void *) listgranhistory;
  else if (strcmp(str,"rolling") == 0) return (void *) &rolling;
  else if (strcmp(str,"model_type") == 0) return (void *) &model_type;
  else if (strcmp(str,"rolling_delta") == 0) return (void *) &rolling_delta;
  else if (strcmp(str,"kappa") == 0) return (void *) &kappa;
  else if (strcmp(str,"post_limit_index") == 0) return (void *) &post_limit_index;
  else if (strcmp(str,"dissipfriction") == 0) return (void *) &dissipfriction;
  else if (strcmp(str,"normalstrain") == 0) return (void *) &normalstrain;
  else if (strcmp(str,"shearstrain") == 0) return (void *) &shearstrain;
  else if (strcmp(str,"trace_energy") == 0) return (void *) &trace_energy;
  return NULL;
}

/* ---------------------------------------------------------------------- */

void PairGranHookeHistory::rolling_resistance(int issingle, int i, int j, int numshearq, double delx, double dely, double delz, double r, double rinv, double ccel, double maxshear, double effectivekt, double **torque, double *shear, double *dur, double *dus, double *localdM, double *globaldM)
{
  /*~ This rolling resistance model was developed by Xin Huang during
    the summer and autumn of 2013. Note that the last two slots in 'shear'
    store the per-contact (kappa+1) values for rolling and twisting which
    controls the strength of the asperities; the third-last slot stores
    the most recent non-zero value of equivalent area tangential
    contact stiffness; the three preceding slots store the
    accumulated global rolling and twisting resistances; the three
    before those store the local accumulated resistances and the six 
    slots before those store the accumulated local dur and dus values.

    'issingle' indicates whether this function is called by the
    pairstyle compute or the single function. If the latter, don't
    update the torque or any other accumulated quantities.

    The input 'model_type' flag gives combinations of options using
    prime numbers. For example, if options a, b and c are assigned the
    integers 2, 3 and 5 (first 3 primes), then model_type == 6 means
    that a and b are active but not c; model_type == 10 means that a and
    c are active but not b; and model_type == 1 means that no options
    are active.

    The input 'kappa' controls the strength of the asperities; permissible
    values are 0-1.

    Defaults:
    * Common radius calculated from definition of Ai et al. (2012)
    
    List of available options:
    3  | Use common radius defined by Iwashita and Oda (1998, 2000)
    5  | Use common radius defined by Jiang et al. (2005)
    7  | Disable rolling resistance part of model
    11 | Disable twisting resistance part of model
  */

  /*~ If rolling_delta == 0, the rolling resistance model does nothing.
    For efficiency, set the relevant columns of the shear array to zero
    and exit from this function prematurely. Also set the increments,
    passed to this function by reference, to zero. As rolling_delta is a
    double, not an integer, compare with a tolerance*/
  double tolerance = 1.0e-20;
  if (rolling_delta < tolerance) {
    for (int q = 0; q < 15; q++) shear[numshearq-15+q] = 0.0;

    for (int q = 0; q < 3; q++)
      dur[q] = dus[q] = localdM[q] = globaldM[q] = 0.0;

    return;
  }

  /*~ Components of the unit vector along the contact normal. In the
    following, particle 1 is analogous to particle j and particle 2
    is taken as particle i. Hence since del* is calculated as (i-j),
    the signs are correct in the calculation of nx, ny and nz.*/
  double nx = delx*rinv;
  double ny = dely*rinv;
  double nz = delz*rinv;

  //~ nz == cos(theta): the rotation angle
  /*~ Calculate the four components of the unit quaternion, q. Compare
    with a tolerance to ensure no division by zero problems*/
  double sinthetaovertwo, magnxny;
  sinthetaovertwo = sqrt(0.5*(1.0 - nz));
  if (nx < 0.0) sinthetaovertwo *= -1.0;

  double q[4];
  q[0] = sqrt(0.5*(1.0 + nz)); //~ = cos(theta)/2
  
  magnxny = sqrt(nx*nx + ny*ny);
  if (magnxny < tolerance) q[1] = q[2] = 0.0;
  else {
    q[1] = -sinthetaovertwo*ny/magnxny;
    q[2] = sinthetaovertwo*nx/magnxny;
  }
  q[3] = 0.0;
  
  //~ Compute the rotation matrix, T, from this quaternion
  double T[3][3];
  MathExtra::quat_to_mat(q,T);

  /*~ Calculate the common radius for which several options are 
    available*/
  double commonradius,radsum,oneoverradsum;
  double *radius = atom->radius;
  radsum = radius[i] + radius[j];
  oneoverradsum = 1.0/radsum;

  //~ Default common radius defined by Ai et al. (2012)
  commonradius = radius[i]*radius[j]*oneoverradsum;

  //~ Use common radius defined by Iwashita and Oda (1998, 2000)
  if (model_type % 3 == 0) commonradius = 0.5*radsum;

  /*~ Use common radius defined by Jiang et al. (2005) which is twice
    the definition of Ai et al. (2012)*/
  if (model_type % 5 == 0) {
    if (model_type % 3 == 0)
      error->all(FLERR,"Cannot have two definitions of common radius active simultaneously");
    commonradius *= 2.0;
  }

  /*~ Obtain the omega values for the previous timestep from 
    FixOldOmega. This array also contains the corresponding x,
    y and z positions of the particle centroids (in columns 4
    to 6)*/
  double **oldomegas = ((FixOldOmega *) deffix)->oldomegas;
  double globaloldomegai[3], globaloldomegaj[3];
  
  for (int q = 0; q < 3; q++) {
    globaloldomegai[q] = oldomegas[i][q];
    globaloldomegaj[q] = oldomegas[j][q];
  }
  
  /*~ Transfer the old omega values to the local coordinate system
    using the rotation matrix T*/
  double localoldomegai[3], localoldomegaj[3];

  MathExtra::transpose_matvec(T,globaloldomegai,localoldomegai);
  MathExtra::transpose_matvec(T,globaloldomegaj,localoldomegaj);

  //~ Use a similar procedure for the difference in coordinates
  double globaldiffcoords[3], localdiffcoords[3];
  for (int q = 0; q < 3; q++)
    globaldiffcoords[q] = oldomegas[i][q+3] - oldomegas[j][q+3];
  MathExtra::transpose_matvec(T,globaldiffcoords,localdiffcoords);

  //~ Now find relative rotations, dthetar, in three directions 
  double PI, beta, dalpha[2], da[3], db[3], dthetar[3];
  
  /*~ beta is calculated using the difference of local z coordinates.
    Note that beta is evaluated as 'nan' if the argument is outside
    the range +/- 1, so check for this and round if necessary*/
  beta = asin(oneoverradsum*localdiffcoords[2]);
  if (fabs(localdiffcoords[2]) > radsum)
    beta = asin(localdiffcoords[2]/fabs(localdiffcoords[2])); //~ Retain sign

  PI = 4.0*atan(1.0);
  dalpha[0] = dalpha[1] = 0.5*PI - beta;

  //~ Correct the sign of dalpha, if necessary
  for (int q = 0; q < 2; q++)
    if (localdiffcoords[q] < 0.0 && dalpha[q] > 0.0)
      dalpha[q] *= -1.0;

  //~ Y-Z projection
  da[0] = radius[j]*(localoldomegaj[0]*dt - dalpha[0]);
  db[0] = radius[i]*(localoldomegai[0]*dt - dalpha[0]);

  //~ X-Z projection
  da[1] = radius[j]*(localoldomegaj[1]*dt - dalpha[1]);
  db[1] = radius[i]*(localoldomegai[1]*dt - dalpha[1]);

  //~ For spin around z axis, dalpha* == 0
  da[2] = radius[j]*localoldomegaj[2]*dt;
  db[2] = radius[i]*localoldomegai[2]*dt;

  for (int q = 0; q < 3; q++) {
    dur[q] = (radius[i]*da[q] - radius[j]*db[q])*oneoverradsum;
    dus[q] = da[q] + db[q];
    dthetar[q] = dur[q]/commonradius;
  }

  /*~ The equivalent area normal contact stiffness is found by dividing
    the magnitude of the normal contact force by the product of the normal 
    contact overlap and contact area. If division by zero, issue a warning.
    Also calculate some necessary quantities for later use*/
  int warnfrequency = 100; //~ How often to warn about stiffness calcs
  double un, deltai, B, delbyb, recipA, knbar, ksbar;
  
  un = fabs(radsum - r); //~ Normal contact overlap
  deltai = 0.5*un*rinv*(2.0*radius[j] - un);
  B = sqrt(deltai*(2.0*radius[i] - deltai)); //~ Radius of contact plane
  
  delbyb = rolling_delta*B;
  if (delbyb > tolerance) knbar = fabs(ccel*r/(un*PI*delbyb*delbyb));
  else {
    //~ Issue a warning and broadcast lastwarning int to all procs
    if (update->ntimestep-lastwarning[0] >= warnfrequency) {
      fprintf(screen,"Cannot estimate either contact stiffness in rolling resistance model on timestep "BIGINT_FORMAT"\n",update->ntimestep);
      lastwarning[0] = lastwarning[1] = update->ntimestep;
      MPI_Bcast(&lastwarning[0],2,MPI_INT,comm->me,world);
    }
    delbyb = tolerance; //~ Tiny value
    knbar = tolerance; //~ Assume a tiny stiffness value
  }
  recipA = 1.0/(PI*delbyb*delbyb); //~ Reciprocal of modified contact area

  /*~ The equivalent area tangential contact stiffness is equal to the
    effective tangential stiffness (kt for Hookean model; kt*polyhertz
    for Hertzian model) divided by the contact area. If kt is zero or
    if the contact area is very, very small, warn the user. Values of
    ksbar are stored in the last column of the shear array.*/
  if (kt < tolerance && update->beginstep == update->ntimestep 
      && comm->me == 0)
    error->warning(FLERR,"Using zero kt: tangential contact stiffness cannot be estimated in rolling resistance model");

  if (B > tolerance && effectivekt > tolerance) ksbar = effectivekt*recipA;
  else if (shear[numshearq-3] > tolerance) ksbar = fabs(shear[numshearq-3]);
  else {
    if (kt >= tolerance && update->ntimestep-lastwarning[1] >= warnfrequency) {
      fprintf(screen,"Cannot estimate tangential contact stiffness in rolling resistance model on timestep "BIGINT_FORMAT"\n",update->ntimestep);
      lastwarning[1] = update->ntimestep;
      MPI_Bcast(&lastwarning[1],1,MPI_INT,comm->me,world);
    }
    ksbar = tolerance; //~ Assume a tiny value
  }
  shear[numshearq-3] = ksbar; //~ Store ksbar in third-last column of shear array

  /*~ Calculate the maximum allowable accumulated global moment and
    store these in mlimit for convenience. The values of (kappa+1) 
    for the contact, stored for rolling and twisting in shear[numshearq-2]
    and shear[numshearq-1], respectively, are used in the calculation. If
    the contact is newly-formed, (kappa+1) will be 0.0 indicating that
    both stored values need to be initialised to their user-specified
    values [KH - 18 July 2014]*/
  double st[3], mlimit[3];
  if (fabs(shear[numshearq-1]) < 0.5) 
    shear[numshearq-1] = shear[numshearq-2] = kappa + 1.0;

  //~ Redefine commonradius for calculation of mlimit
  commonradius = radius[i]*radius[j]*oneoverradsum;

  mlimit[0] = mlimit[1] = (fabs(shear[numshearq-2])-1.0)*commonradius*ccel*r;
  mlimit[2] = (fabs(shear[numshearq-1])-1.0)*commonradius*maxshear;

  //~ Also create an 'st' array for convenience
  double deltaBpow = MathSpecial::powint(delbyb,4);
  st[0] = st[1] = -0.25*PI*deltaBpow*knbar;
  st[2] = -0.5*PI*deltaBpow*ksbar;

  //~ Calculate local increments of rolling resistance
  for (int q = 0; q < 3; q++) localdM[q] = st[q]*dthetar[q];
  if (model_type % 7 == 0) localdM[0] = localdM[1] = 0.0;
  if (model_type % 11 == 0) localdM[2] = 0.0;

  double scalefactor; //~ Denominator used for scaling resistances
  double acclocal[3], accglobal[3];
  for (int q = 0; q < 3; q++) {
    /*~ If the accumulated local resistances exceed the permissible
      limits, proportionally reduce the size of the incremental
      resistances and also scale the accumulated resistances 
      so that the accumulated local resistance will equal the 
      appropriate limit once the increment is added to it at the end
      of this function. The accumulated local rolling and twisting 
      resistances are stored in the ninth-last, eighth-last and
      seventh-last columns of the shear array*/
    scalefactor = fabs(shear[numshearq-9+q] + localdM[q]);
    if (scalefactor > mlimit[q]) {
      /*~ Updated the stored value of (kappa+1) for the 
	rolling	model only (not twisting). If post_limit_index
	== 0, set the stored value to 1; otherwise don't change
	it [KH - 18 July 2014]*/
      if (q < 2) {
	shear[numshearq-2] += (post_limit_index - 1.0)*(fabs(shear[numshearq-2]) - 1.0);
	mlimit[q] *= post_limit_index; //~ Control post-limit strength
      }
      localdM[q] *= mlimit[q]/scalefactor;
      shear[numshearq-9+q] *= mlimit[q]/scalefactor;
    }
    acclocal[q] = shear[numshearq-9+q]; // Fetch the accumulated local in one vector
  }

  /*~ Compute the global moment increments by multiplying the 
    transpose of the rotation matrix, T, by the local resistance
    increments. Do the same for the accumulated moment.*/
  MathExtra::matvec(T,localdM,globaldM);
  MathExtra::matvec(T,acclocal,accglobal);

  if (!issingle) {
    /*~ Now add the local resistance increments to the ninth-last, 
      eighth-last and seventh-last columns of the shear array. The
      accumulated global moment is in the sixth-last, fifth-last 
      and fourth-last columns. The accumulated values of dus are 
      stored in the three columns immediately before, and the 
      accumulated values of dur in the three columns immediately 
      before these.*/
    for (int q = 0; q < 3; q++) {
      shear[numshearq-6+q] = accglobal[q];
      shear[numshearq-9+q] += localdM[q];
      shear[numshearq-12+q] += dus[q];
      shear[numshearq-15+q] += dur[q];
    }

    /*~ Finally update the torque values for both i and j (the
      latter only if local to proc). The increments differ in sign*/
    for (int q = 0; q < 3; q++) {
      torque[i][q] -= accglobal[q];
      if (j < atom->nlocal) torque[j][q] += accglobal[q];
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairGranHookeHistory::add_old_omega_fix()
{
  //~ Check whether this fix is already present
  int oldomegafix = -1;
  for (int q = 0; q < modify->nfix; q++)
    if (strcmp(modify->fix[q]->style,"old_omega") == 0) oldomegafix = q;

  if (oldomegafix < 0) {//~ Fix not presently active
    char **newarg = new char*[3];
    newarg[0] = (char *) "pair_oldomega";
    newarg[1] = (char *) "all";
    newarg[2] = (char *) "old_omega";

    modify->add_fix(3,newarg,1);
    delete [] newarg;
  }
  
  //~ Set pointers for this newly-created fix
  int accfix = modify->find_fix("pair_oldomega");
  if (accfix < 0) error->all(FLERR,"Fix ID for fix old_omega does not exist");
  deffix = modify->fix[accfix];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairGranHookeHistory::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}

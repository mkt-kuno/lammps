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
#include <stdio.h>
#include <string.h>
#include "pair_gran_hertz_history.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "fix.h"
#include "fix_neigh_history.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "domain.h" //~ Two header files were added [KH - 9 November 2011]
#include "modify.h"
#include "compute_energy_gran.h" //~ For energy tracing [KH - 20 February 2014]

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairGranHertzHistory::PairGranHertzHistory(LAMMPS *lmp) :
  PairGranHookeHistory(lmp) {}

/* ---------------------------------------------------------------------- */

void PairGranHertzHistory::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz,fx,fy,fz;
  double radi,radj,radsum,rsq,r,rinv,rsqinv;
  double vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double wr1,wr2,wr3;
  double vtr1,vtr2,vtr3,vrel;
  double mi,mj,meff,damp,ccel,tor1,tor2,tor3;
  double fn,fs,fs1,fs2,fs3;
  double shsqmag,shsqnew,shratio,rsht,polyhertz;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *touch,**firsttouch;
  double *shear,*allshear,**firstshear;

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

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
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double deltan,cri,crj,oldshear[3];

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firsttouch = fix_history->firstflag;
  firstshear = fix_history->firstvalue;

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

  //~ Initialise the non-accumulated strain energy terms to zero
  normalstrain = 0.0;

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

        mi = rmass[i];
        mj = rmass[j];
        if (fix_rigid) {
          if (mass_rigid[i] > 0.0) mi = mass_rigid[i];
          if (mass_rigid[j] > 0.0) mj = mass_rigid[j];
        }

        meff = mi*mj / (mi+mj);
        if (mask[i] & freeze_group_bit) meff = mj;
        if (mask[j] & freeze_group_bit) meff = mi;

        // normal force = Hertzian contact + normal velocity damping

        damp = meff*gamman*vnnr*rsqinv;
        ccel = kn*(radsum-r)*rinv - damp;
        polyhertz = sqrt((radsum-r)*radi*radj / radsum);
        ccel *= polyhertz;

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
	  //~ Store the old values of shear [KH - 20 February 2014]
	  oldshear[0] = shear[0];
	  oldshear[1] = shear[1];
	  oldshear[2] = shear[2];

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

        fs1 = -polyhertz * (kt*shear[0] + meff*gammat*vtr1);
        fs2 = -polyhertz * (kt*shear[1] + meff*gammat*vtr2);
        fs3 = -polyhertz * (kt*shear[2] + meff*gammat*vtr3);

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

        if (newton_pair || j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
	  torque[j][0] -= crj*tor1;
	  torque[j][1] -= crj*tor2;
	  torque[j][2] -= crj*tor3;
        }

	double dur[3], dus[3], localdM[3], globaldM[3]; //~ Pass by reference
	double effectivekt = kt*polyhertz;
	double aveshearforce, slipdisp, oldsheardisp, incrementaldisp;
	double incdissipf, nstr, sstr;

	int consideronce = 0; //~ Consider contacts only once
	if (j < nlocal || tag[j] < tag[i]) consideronce = 1;

	//~ Call function for rolling resistance model [KH - 25 October 2013]
	if (rolling && shearupdate && consideronce) {
	  /*~ The first '0' indicates that the rolling_resistance function is
	    called by the compute rather than the single function*/
	  rolling_resistance(0,i,j,numshearquants,delx,dely,delz,r,rinv,ccel,
			     fn,effectivekt,torque,shear,dur,dus,localdM,
			     globaldM);
	}

	//~ Add contributions to traced energy [KH - 20 February 2014]
	if (pairenergy) {
	  /*~ Increment the friction energy only if the slip condition
	    is invoked*/
	  oldsheardisp = sqrt(oldshear[0]*oldshear[0] + oldshear[1]*oldshear[1] + oldshear[2]*oldshear[2]);
	  if (fs > fn && fn > 0.0) {
	    //~ current shear displacement = fn/effectivekt;
	    slipdisp = (fs-fn)/effectivekt;
	    aveshearforce = 0.5*(fn + polyhertz*kt*oldsheardisp);
	      
	    //~ slipdisp and aveshearforce are both positive
	    incdissipf = aveshearforce*slipdisp;
	    if (consideronce) dissipfriction += incdissipf;
	    if (trace_energy) shear[3] += incdissipf;
	  }

	  /*~ Update the normal contribution to strain energy which 
	    doesn't need to be calculated incrementally*/
	  nstr = 0.4*kn*polyhertz*deltan*deltan;
	  if (consideronce) normalstrain += nstr;
	  if (trace_energy) shear[4] = nstr;

	  //~ The shear component does require incremental calculation
	  if (shearupdate) {
	    incrementaldisp = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]) - oldsheardisp;
	    sstr = 0.5*incrementaldisp*(2.0*sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3)-effectivekt*incrementaldisp);
	    if (consideronce) shearstrain += sstr;
	    if (trace_energy) shear[5] += sstr;
	  }
	}

        if (evflag) ev_tally_gran(i,j,nlocal,fx,fy,fz,x[i][0],x[i][1],x[i][2],
                                 radius[i],x[j][0],x[j][1],x[j][2],radius[j]);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGranHertzHistory::settings(int narg, char **arg)
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

  // convert Kn and Kt from pressure units to force/distance^2

  kn /= force->nktv2p;
  kt /= force->nktv2p;
}

/* ---------------------------------------------------------------------- */

double PairGranHertzHistory::single(int i, int j, int itype, int jtype,
                                    double rsq,
                                    double factor_coul, double factor_lj,
                                    double &fforce)
{
  double radi,radj,radsum;
  double r,rinv,rsqinv,delx,dely,delz;
  double vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3,wr1,wr2,wr3;
  double mi,mj,meff,damp,ccel,polyhertz;
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
  int *mask = atom->mask;

  mi = rmass[i];
  mj = rmass[j];
  if (fix_rigid) {
    // NOTE: insure mass_rigid is current for owned+ghost atoms?
    if (mass_rigid[i] > 0.0) mi = mass_rigid[i];
    if (mass_rigid[j] > 0.0) mj = mass_rigid[j];
  }

  meff = mi*mj / (mi+mj);
  if (mask[i] & freeze_group_bit) meff = mj;
  if (mask[j] & freeze_group_bit) meff = mi;


  // normal force = Hertzian contact + normal velocity damping

  damp = meff*gamman*vnnr*rsqinv;
  ccel = kn*(radsum-r)*rinv - damp;
  polyhertz = sqrt((radsum-r)*radi*radj / radsum);
  ccel *= polyhertz;

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
  int *jlist = list->firstneigh[i];
  double *allshear = fix_history->firstvalue[i];

  for (int jj = 0; jj < jnum; jj++) {
    neighprev++;
    if (neighprev >= jnum) neighprev = 0;
    if (jlist[neighprev] == j) break;
  }

 /*~ The number of shear quantities is 18 if rolling resistance
    is active [KH - 29 July 2014]*/
  /*~ Another 4 shear quantities were added for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 3 + 15*rolling + 4*trace_energy;
  double *shear = &allshear[numshearquants*neighprev];

  // rotate shear displacements - not needed- shear already updated by compute!

  // tangential forces = shear + tangential velocity damping

  fs1 = -polyhertz * (kt*shear[0] + meff*gammat*vtr1);
  fs2 = -polyhertz * (kt*shear[1] + meff*gammat*vtr2);
  fs3 = -polyhertz * (kt*shear[2] + meff*gammat*vtr3);

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
  double effectivekt = kt*polyhertz;
  double **torque = atom->torque;
  if (rolling) {
    /*~ The first '1' indicates that the rolling_resistance function is
      called by the single function rather than the compute*/
    rolling_resistance(1,i,j,numshearquants,delx,dely,delz,r,rinv,ccel,
		       fn,effectivekt,torque,shear,dur,dus,localdM,globaldM);
  }

  /*~ Some of the following are included only for convenience as
    the data could instead be obtained from a dump of the sphere
    coordinates [KH - 13 December 2011]*/
  fforce = ccel;

  // set single_extra quantities

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

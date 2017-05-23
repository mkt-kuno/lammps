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
#include "pair_gran_HMD_history.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "fix.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "error.h"
#include "domain.h"
#include "modify.h" //~ This header file was added [KH - 9 November 2011]
#include "neigh_request.h" //~ Added four more header files [KH - 23 November 2012]
#include "fix.h"
#include "fix_shear_history.h"
#include "memory.h"
#include "compute_energy_gran.h" //~ For energy tracing [KH - 20 February 2014]

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairGranHMDHistory::PairGranHMDHistory(LAMMPS *lmp) :
  PairGranHookeHistory(lmp) {}

/* ---------------------------------------------------------------------- */

void PairGranHMDHistory::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fx,fy,fz;
  double radi,radj,radsum,rsq,r,rinv,rsqinv;
  double vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3;
  double wr1,wr2,wr3;
  double vtr1,vtr2,vtr3,vrel;
  double ccel,tor1,tor2,tor3;
  double fslim,fs,fs1,fs2,fs3;
  double shsqmag,shsqnew,shratio,rsht,polyhertz;
  double wspinx,wspiny,wspinz,shint0,shint1,shint2,omdel;
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
  double deltan,cri,crj;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firsttouch = list->listgranhistory->firstneigh;
  firstshear = list->listgranhistory->firstdouble;

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

  //***********
  dissipfriction = 0.0; // TEMPORARY [MO - 25 Sep 2015]
  //***********

  // Modified for HMD and CMD model that use 26 shear quantities [MO - 17 November 2014].
  int numshearquants = 26 + 20*D_spin + 4*trace_energy;

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
        shear = &allshear[numshearquants*jj]; // shear[] refers to a shear force here, not shear displacement
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

        // relative velocities

        vtr1 = vt1 - (delz*wr2-dely*wr3);
        vtr2 = vt2 - (delx*wr3-delz*wr1);
        vtr3 = vt3 - (dely*wr1-delx*wr2);
        vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
        vrel = sqrt(vrel);

        // shear history effects

        touch[jj] = 1;
        shear = &allshear[numshearquants*jj]; // shear[] refers to a shear force here, not shear displacement

        shsqmag = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];

        // rotate shear forces onto new contact plane conserving length

        rsht = shear[0]*delx + shear[1]*dely + shear[2]*delz;
        rsht *= rsqinv;
        if (shearupdate) {
          shear[0] -= rsht*delx;
          shear[1] -= rsht*dely;
          shear[2] -= rsht*delz;
          shsqnew = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2];
          if (shsqnew!=0.0) {
	    shratio=sqrt(shsqmag/shsqnew);
	    shear[0] *= shratio; // conserve shear force length
	    shear[1] *= shratio;
	    shear[2] *= shratio;
          }
        }

        // then perform rotation for rigid-body SPIN
        omdel=(omega[i][0]+omega[j][0])*delx+(omega[i][1]+omega[j][1])*dely+(omega[i][2]+omega[j][2])*delz;
        wspinx=0.5*rsqinv*delx*omdel;
        wspiny=0.5*rsqinv*dely*omdel;
        wspinz=0.5*rsqinv*delz*omdel;
        shint0 = shear[0];
        shint1 = shear[1];
        shint2 = shear[2];

        if (shearupdate) {
	  shear[0]=shint0+shint1*(-wspinz*dt)+shint2*wspiny*dt;
	  shear[1]=shint0*wspinz*dt+shint1+shint2*(-wspinx*dt);
	  shear[2]=shint0*(-wspiny*dt)+shint1*wspinx*dt+shint2;
        }
	
	//*************************************************************************************
	/* The main part of the HMD model is wrriten down below as of 13th August 2014 [MO]*/ 
	//*************************************************************************************

	double tolerance = 1.0e-20;
	if (THETA1 == 0 && xmu > tolerance) {
	  
	  // rotate shear displacement onto new contact plane conserving length
	  double shdsq_mag,shdsq_new,rshd,shdratio,shint5,shint6,shint7;
	  shdsq_mag = shear[5]*shear[5] + shear[6]*shear[6] + shear[7]*shear[7];
	  rshd = shear[5]*delx + shear[6]*dely + shear[7]*delz;
	  rshd *= rsqinv;
	  if (shearupdate) {
	    shear[5] -= rshd*delx;
	    shear[6] -= rshd*dely;
	    shear[7] -= rshd*delz;
	    shdsq_new = shear[5]*shear[5] + shear[6]*shear[6] + shear[7]*shear[7];
	    if (shdsq_new!=0.0) {
	      shdratio=sqrt(shdsq_mag/shdsq_new);
	      shear[5] *= shdratio; // conserve shear force length
	      shear[6] *= shdratio;
	      shear[7] *= shdratio;
	    }
	  }

	  // then perform rotation for rigid-body SPIN
	  shint5 = shear[5];
	  shint6 = shear[6];
	  shint7 = shear[7];
	  if (shearupdate) {
	    shear[5]=shint5+shint6*(-wspinz*dt)+shint7*wspiny*dt;
	    shear[6]=shint5*wspinz*dt+shint6+shint7*(-wspinx*dt);
	    shear[7]=shint5*(-wspiny*dt)+shint6*wspinx*dt+shint7;
	  }

	  ////////////////// 
	  // rotate shear displacement onto new contact plane conserving length
	  // Added for Tdisp_star1 [MO - 02 April 2015]
	  double shdsq_mag_star1,shdsq_new_star1,rshd_star1,shdratio_star1,shint22,shint23,shint24;
	  // make sure shear[22:24] = Tdisp1_star1,Tdisp2_star1 & Tdisp3_star1;
	  shdsq_mag_star1 = shear[22]*shear[22] + shear[23]*shear[23] + shear[24]*shear[24];
	  rshd_star1 = shear[22]*delx + shear[23]*dely + shear[24]*delz;
	  rshd_star1 *= rsqinv;
	  if (shearupdate) {
	    shear[22] -= rshd_star1*delx;
	    shear[23] -= rshd_star1*dely;
	    shear[24] -= rshd_star1*delz;
	    shdsq_new_star1 = shear[22]*shear[22] + shear[23]*shear[23] + shear[24]*shear[24];
	    if (shdsq_new_star1!=0.0) {
	      shdratio_star1=sqrt(shdsq_mag_star1/shdsq_new_star1);
	      shear[22] *= shdratio_star1; // conserve shear force length
	      shear[23] *= shdratio_star1;
	      shear[24] *= shdratio_star1;
	    }
	  }
	  // then perform rotation for rigid-body SPIN
	  shint22 = shear[22];
	  shint23 = shear[23];
	  shint24 = shear[24];
	  if (shearupdate) {
	    shear[22]=shint22+shint23*(-wspinz*dt)+shint24*wspiny*dt;
	    shear[23]=shint22*wspinz*dt+shint23+shint24*(-wspinx*dt);
	    shear[24]=shint22*(-wspiny*dt)+shint23*wspinx*dt+shint24;
	  }
	}
	
	//*****************************************************************************************
	/*To avoid the reverse of sign of shearquantities, 'fabs' is used for absolute quantities.
	  Attention should be paied for some flags which have 1 or -1 values. For these cases, the 
	  quantities were stored as 10000 instead of -1 and called with 'fabs', and changed to -1 
	  again at the next step.[MO - 15 December 2014] */
	//*****************************************************************************************
	// if xmu < tolerance or dTdisp < tolerance && T_step == 0, i.e. begining of simulation.
	//*****************************************************************************************

        // normal force = Hertzian contact 
	double R_star = 1.0 / (1.0/radi + 1.0/radj);  // equivalent radius of spheres
	double E_star = Geq/(1.0-Poiseq);             // equivalent Young's modulus at contact
	double G_star = Geq/(2.0*(2.0-Poiseq));       // equivalent shear modulus at contact
	double pi = 4.0*atan(1.0);                    // pi = 3.1415*******
	double overlap = radsum-r;                    // overlap of spheres
      	double overlap_old = fabs(shear[3]);                // overlap of spheres at the previous step 
	double a = sqrt(R_star*overlap);              // radius of contact circle
	double a_old = sqrt(R_star*overlap_old);      // radius of contact circle at the previous step 
	double N = kn*overlap*a;                      // normal force
	double N_old = kn*overlap_old*a_old;          // normal force at the previous step
	double dN = N - N_old;                        // change of normal force
	double K_N = 2.0*E_star*a;                    // normal contact stiffness (tangential, not secant)
	polyhertz = a;                                // re-use polyhertz to adjust the format for shm contact model
	ccel = kn*overlap*a*rinv;                     // previously, ccel = kn*overlap*polyhert*rinv;	
	//****************************************************************************************
	// The following codes introduce HMD model for tangential component.
	double T = 0.0;                               // bulk tangential force
	double T_old = shear[12];                     // bulk tangential force at the previous step (including its sign)
	double T_star1 = shear[8];                    // T*:  tangential force at the revasal step from loading to unloading 
	double T_star2 = shear[9];                    // T**: tangential force at the revasal step from unloading to re-loading 
	double Tdisp  = 0.0;                          // bulk tangential displacement	
	double Tdisp_mag = 0.0;                        // magnitude of bulk tangential displacement
	double Tdisp_old  = shear[4];                 // bulk tangential displacement at the previous step
	double dTdisp = 0.0;                          // increment of bulk displacement
       	double Tdisp1 = 0.0;                          // tangential displacement of x direction
	double Tdisp2 = 0.0;                          // tangential displacement of y direction
	double Tdisp3 = 0.0;                          // tangential displacement of z direction
	double Tdisp1_old = shear[5];                 // tangential displacement of x direction at the previous step
	double Tdisp2_old = shear[6];                 // tangential displacement of y direction at the previous step
	double Tdisp3_old = shear[7];                 // tangential displacement of z direction at the previous step     
	double inner_product = 0.0;                   // inter product between current and previous vectors of tangential displacement
	int T_step = 0;                               // the current loading case, e.g. 11 = N increasing T increasing
	int T_step_old = static_cast<int>(fabs(shear[15])); // the loading case at the previous step
	int slip_T = 0;
	int UFL = 0;                                  // UFL is the direction of tangential load
	int CTD = static_cast<int>(fabs(shear[14]));         // CTD is the direction of tangential displacement 
	if (CTD == 10000) CTD = -1;
	int CDF = static_cast<int>(fabs(shear[11]));         // CDF is the direction of tangential load-displacement system
	if (CDF == 10000) CDF = -1;
	double theta1 = 1.0;
	double theta2 = 1.0;
	double theta3 = 1.0;                  
	double theta_t = 1.0;
	double Tdisp1_star1 = shear[22];     // tangential displacement of x direction at T_star1
	double Tdisp2_star1 = shear[23];     // tangential displacement of y direction at T_star1
	double Tdisp3_star1 = shear[24];     // tangential displacement of z direction at T_star1
	double Tdisp_star1_mag;


	fslim = xmu * N;   // N = ccel*r 
	int    T_step_group = 0;
	int    T_step_group_old = 0;
	double Tdisp_DD = 0.0;
	double K_T = 0.0;       // tangential contact stiffness
	double dT = 0.0;        // increment of tangential contact force
	double fs_old = sqrt(shsqmag);	
	double dTdisp_mag = sqrt(vtr1*vtr1+vtr2*vtr2+vtr3*vtr3)*dt; 

	if (xmu < tolerance) {
	  K_T = 0.0;
	  T_step = 0;
	  CTD = 1;
	  CDF = 1;
	  T_step = 0; 
	  Tdisp1 = Tdisp2 = Tdisp3 = Tdisp = dTdisp = dTdisp_mag = 0.0;
	  T = shear[0] = shear[1] = shear[2] = 0.0;
	} 
	else if (THETA1 == 1) {
	  K_T = 8.0*G_star*a;
	  T_step = 1;
	  CTD = 1;
	  CDF = 1;
	  dTdisp = dTdisp_mag = 0.0;
	  // update T & Tdisps later
	} 
	else if (dTdisp_mag < 1.0e-15) {
	  // need to update historical parameters
	  K_T = 0.0;
	  T_step = T_step_old;
	  T = T_old;
	  Tdisp1 = Tdisp1_old;          
	  Tdisp2 = Tdisp2_old;
	  Tdisp3 = Tdisp3_old;
	  Tdisp = Tdisp_old;
	  dTdisp = dTdisp_mag = 0.0;
	}
	else {

	  //******************************************************************
	  // First, calculate the shear displacement
	  // Skip the follwing calculation if xmu or dTdisp is small
	  //******************************************************************
	  
	  Tdisp1 = Tdisp1_old + vtr1*dt;          
	  Tdisp2 = Tdisp2_old + vtr2*dt;
	  Tdisp3 = Tdisp3_old + vtr3*dt;
	  Tdisp_mag = sqrt(Tdisp1*Tdisp1 + Tdisp2*Tdisp2 + Tdisp3*Tdisp3);
	  
	  // inner_product becomes negative if tangential disp. becomes negative.
	  inner_product = Tdisp1*Tdisp1_star1 + Tdisp2*Tdisp2_star1 + Tdisp3*Tdisp3_star1;
	  Tdisp_star1_mag = sqrt(Tdisp1_star1*Tdisp1_star1 + Tdisp2_star1*Tdisp2_star1 + Tdisp3_star1*Tdisp3_star1);
	  
	  // modified [MO - 02 April 2015]
	  if (Tdisp_mag < tolerance && fabs(Tdisp_old) < tolerance && T_step_old == 0) {
	    CTD = 1;                                     // when there is no shear disp at all.
	    CDF = 1;
	  }
	  else if (Tdisp_mag > tolerance && fabs(Tdisp_old) < tolerance && T_step_old == 0) {
	    // this is for the first increment of tangential displacement
	    CTD = 1;                                   
	    CDF = 1;
	  }
	  else if (CDF == 1 && (Tdisp_star1_mag < tolerance))  CTD = 1;
	  else if (inner_product >= 0.0)                       CTD = 1;
	  else                                                 CTD = -1; 
	  
	  Tdisp = CDF * CTD * Tdisp_mag; // CDF is added here [MO - 02 April 2015]    
	  
	  // give the sign for tangential displacement based on the value of the inner product 
	  dTdisp = Tdisp - Tdisp_old;   // dTdisp includes direction of tangential displacement (not magnitude)
	  
	  //**************************************************************************************
	  // Update the Tdisp_DD
	  /* Fig.7 of Mindlin & Deresiewicz (1953) explain why this special case is needed.
	     Tdisp_DSC is the minimum tangential displacement to move onto a new loading curve of N+dN.
	     If the increment of the tangential displacement is less than Tdisp_DSC, the resultant tangential force
	     is less than theoretical value.
	     The maximum tangential contact stiffness (8.0*G_star*a) is used until the current tangential force 
	     catches up the the new loading curve. */
	  //*************************************************************************************
	  double Tdisp_DSC = 0.0;
	  if (a > tolerance) Tdisp_DSC = xmu*dN/(8.0*G_star*a);  // sign of Tdisp_DSC depends upon sign of dN;
	  	  
	  Tdisp_DD = fabs(shear[13]) - fabs(dTdisp) + Tdisp_DSC;
	  if (dN < 0) Tdisp_DD = fabs(shear[13]) + Tdisp_DSC;
	  int    special_DD = 0; 
	  if (Tdisp_DD < 0.0) Tdisp_DD = 0.0;  // Tdisp_DD <= 0.0 should be satisfied to move onto a new loading curve for N+dN.
	  else                special_DD = 1;  // Special case for DD value is now active
	  //*************************************************************************************
	  
	  // Identify the loading steps for tangential component
	  // for the first step of the special case	  
	  if (special_DD == 1){
	    theta_t = 1.0;
	    if (dN > 0.0 && T_step_old != 115 && T_step_old != 125 && T_step_old != 135 && T_step_old != 145){
	      if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance)          T_step = 115;
	      else if (CDF*dTdisp <  0.0 && fabs(T_star2) < tolerance)                                  T_step = 125; 
	      else if (CDF*dTdisp >= 0.0 && fabs(T_old) <= fabs(T_star1))                                   T_step = 135;
	      else if (CDF*dTdisp <  0.0 && fabs(T_old) <= fabs(T_star1) && fabs(T_star2) >= tolerance)      T_step = 145;
	      else T_step = T_step_old;
	      //fprintf(screen,"timestep %i Unexpected case occurred in zone B (HMD-balls). ERROR!!\n",update->ntimestep);
	      // if the special case was invoked in the previous step and is still active
	    }else if ((T_step_old == 115 || T_step_old == 135) && CDF*dTdisp >= 0.0){  
	      // still in step_115 or 135; if moved to unloading of T, go to the usual case
	      T_step = T_step_old;
	    }else if ((T_step_old == 125 || T_step_old == 145) && CDF*dTdisp < 0.0) {   
	      // still in step_125 or 145; if moved to re-loading of T, go to the usual case
	      T_step = T_step_old;                                           
	    }else if (T_step_old == 115 && CDF*dTdisp < 0.0){
	      // the first step from loading of special case to unloading of special case.
	      T_step = 125;
	    }else if (T_step_old == 125 && CDF*dTdisp >= 0.0){
	      // the first step from unloading of special case to reloading of special case.
	      T_step = 135;
	    }else if (T_step_old == 135 && CDF*dTdisp < 0.0){
	      // the first step from reloading  of special case to re-unloading of special case.
	      T_step = 145;
	    }else if (T_step_old == 145 && CDF*dTdisp > 0.0){
	      // the first step from re-unloading  of special case to re-reloading of special case.
	      T_step = 135;
	    }
	    else T_step = T_step_old;                                           
	  
	    //*************************************************************************************
	  }else {  // Usual cases 
	    // T loading: T_step = *1 
	    if (CDF*dTdisp >= 0.0 && fabs(T_star1) < tolerance && fabs(T_star2) < tolerance) { 
	      theta1 = 1.0-(CDF*T_old+xmu*dN)/(xmu*N);	
	      if (theta1 <= 0.0) theta1 = theta_t = tolerance;
	      else theta_t = pow(theta1, 1.0/3.0);
	      if       (dN >  tolerance)      T_step = 11;   // N increasing
	      else if  (dN < -tolerance)      T_step = 21;   // N decreasing
	      else                            T_step =  1;   // N constant
	    }
	    // T unloading: T_step = *2 
	    else if (CDF*dTdisp < 0.0 && fabs(T_star2) < tolerance) {
	      // For the first step from loading to unloading, T* is still null, which should be T* = T_old.
	      if  (fabs(T_star1) > tolerance) theta2 = 1.0-(CDF*(T_star1-T_old)+2.0*xmu*dN)/(2.0*xmu*N);
	      else                            theta2 = 1.0-(2.0*xmu*dN)/(2.0*xmu*N);
	      if (theta2 <= 0.0) theta2 = theta_t = tolerance;	  
	      else               theta_t  = pow(theta2, 1.0/3.0);
	      if      (dN >  tolerance) T_step = 12; 
	      else if (dN < -tolerance) T_step = 22;      
	      else                      T_step =  2;
	    }
	    // T re-loading: T_step = *3
	    else if (CDF*dTdisp >= 0.0 && fabs(T_old) <= fabs(T_star1)) {
	      // For the first step from unloading to re-loading, T** is still null, which should be T** = T_old.
	      if  (fabs(T_star2) > tolerance) theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu*dN)/(2.0*xmu*N);
	      else                            theta3 = 1.0-(2.0*xmu*dN)/(2.0*xmu*N);
	      if (theta3 <= 0.0) theta3 = theta_t = tolerance;
	      else               theta_t  = pow(theta3, 1.0/3.0);
	      if      (dN >  tolerance) T_step = 13; 
	      else if (dN < -tolerance) T_step = 23;     
	      else                      T_step = 3;
	    }
	    // T re-unloading: T_step = *4
	    else if (CDF*dTdisp < 0.0 && fabs(T_old) <= fabs(T_star1) && fabs(T_star2) >= tolerance) {
	      // This part is same with the above (T_step = *3) due to simplification.
	      theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu*dN)/(2.0*xmu*N);
	      if (theta3 <= 0.0) theta3 = theta_t = tolerance;
	      else               theta_t  = pow(theta3, 1.0/3.0);
	      if      (dN >  tolerance) T_step = 14;
	      else if (dN < -tolerance) T_step = 24; 
	      else                      T_step =  4;      
	    }
	    else {
	      T_step = T_step_old;
	      theta_t = 1.0;
	      //fprintf(screen,"timestep %i Unexpected case occurred in zone C (HMD-balls). ERROR!!\n",update->ntimestep);
	    }
	  }
      	
	  //*********************************************************************************************
	  // Classify the groups [MO - 19 March 2015]
	  if      (T_step == 0)                                                  T_step_group = 0;
	  else if (T_step == 1 || T_step == 11 || T_step == 21 || T_step == 115) T_step_group = 1;
	  else if (T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) T_step_group = 2;
	  else if (T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) T_step_group = 3;
	  else if (T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) T_step_group = 4;
	  else                                                                   T_step_group = 5; // error
	  // Classify the previos groups
	  if      (T_step_old == 0)                                                              T_step_group_old = 0;
	  else if (T_step_old == 1 || T_step_old == 11 || T_step_old == 21 || T_step_old == 115) T_step_group_old = 1;
	  else if (T_step_old == 2 || T_step_old == 12 || T_step_old == 22 || T_step_old == 125) T_step_group_old = 2;
	  else if (T_step_old == 3 || T_step_old == 13 || T_step_old == 23 || T_step_old == 135) T_step_group_old = 3;
	  else if (T_step_old == 4 || T_step_old == 14 || T_step_old == 24 || T_step_old == 145) T_step_group_old = 4;
	  else                                                                                   T_step_group_old = 5; // error
	
	  //***********************************************************************************
	  //UFL is neccesary for the calculation of tangential contact stiffness
	  if (T_step_group == 2 || T_step_group == 4) UFL = -1; // Added [MO - 21 Dec 2014]
	  else                                        UFL =  1;	
	  //***********************************************************************************
	  
	  // Calculate the tangential contact stiffness and the resultant tangential contact force 
	  if (theta_t > 1.0) theta_t = 1.0;
	  if (fabs(dTdisp) > 1.0e-15) K_T = 8.0*G_star*theta_t*a + CDF*UFL*xmu*(1.0-theta_t)*dN/dTdisp;
	  else K_T = 0.0;
	  dT = K_T * dTdisp; 
	  T = dT + T_old;
	  // rescale tangential force if full sliding takes place 	
	  if (fabs(T) > fslim) {
	    if (fabs(T) != 0.0) {
	      slip_T = 1;
	      T *= fslim/fabs(T);
	    } 
	  }
	} // End of If THETA == 0
	

	// update the tangential contact force by distributing dT into 3 components based on imcrese of relative rotational velosity
	if (shearupdate) {
	  shear[0] -= K_T * vtr1 * dt;  
	  shear[1] -= K_T * vtr2 * dt; 
	  shear[2] -= K_T * vtr3 * dt; 
	}
	
	// fabs(T) and fs_new are not always same for 3D simulation [MO - 03 April 2015] 
	fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
	
	if (fs > fslim) {
	  if (fs != 0.0 && xmu > tolerance) {
	    slip_T = 1;
	    if (shearupdate) {
	      shear[0] *= fslim/fs; 
	      shear[1] *= fslim/fs;
	      shear[2] *= fslim/fs;   
	    } 
	  } else shear[0] = shear[1] = shear[2] = 0.0;
	} 
	
	double fs_new = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
	if (fs_new > fslim) fs_new = fslim;

	double slip_degree = fs/fslim;
	// Added to convert HM to HMD model
	if (THETA1 == 1 && xmu > tolerance) {
	  if (K_T > tolerance && fs_new > tolerance) {
	    // consider as a virgin tangential loading
	    T = fs_new;
	    if (slip_degree >= 1.0)      theta_t = 0.0; //Tdisp = 1.5 * fslim / K_T; 
	    else if (slip_degree < 0.0)  theta_t = 1.0; //Tdisp = 0.0; 
	    else                         theta_t = pow((1.0-slip_degree),1.0/3.0); 
	    Tdisp = 1.5 * fslim / K_T * (1.0 - theta_t*theta_t);
	    Tdisp1 = Tdisp * shear[0]/fs_new;
	    Tdisp2 = Tdisp * shear[1]/fs_new;
	    Tdisp3 = Tdisp * shear[2]/fs_new;    

	    /*fprintf(screen,"timestep %i tagi %i tagj %i Kt %1.6e T %1.6e shear[0] %1.6e shear[1] %1.6e shear[2] %1.6e Tdisp %1.6e Tdisp1 %1.6e Tdisp2 %1.6e Tdisp3 %1.6e overlap %1.6e a %1.6e N %1.6e fslim %1.6e slip_degree %1.6e theta_t %1.6e \n",update->ntimestep,tag[i],tag[j],K_T,T,shear[0],shear[1],shear[2],Tdisp,Tdisp1,Tdisp2,Tdisp3,overlap,a,N,fslim,slip_degree,theta_t);*/

	  }
	  else T = Tdisp = Tdisp1 = Tdisp2 = Tdisp3 = 0.0;
	}

	
	// forces & torques
	
        fx = delx*ccel + shear[0];
        fy = dely*ccel + shear[1];
        fz = delz*ccel + shear[2];
        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;

        tor1 = rinv * (dely*shear[2] - delz*shear[1]);
        tor2 = rinv * (delz*shear[0] - delx*shear[2]);
        tor3 = rinv * (delx*shear[1] - dely*shear[0]);
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

	double nstr, sstr, incdissipf;
	double dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy;
	
	int consideronce = 0; //~ Consider contacts only once
	if (j < nlocal || tag[j] < tag[i]) consideronce = 1;

	//~~ Call function for twisting resistance model [MO - 04 November 2014]]
	if (D_spin && shearupdate && consideronce) {
	  Deresiewicz1954_spin(0,i,j,numshearquants,r,torque,shear,dspin_i,
			       dspin_stm,spin_stm,dM_i,dM,K_spin,theta_r,
			       M_limit,Geq,Poiseq,Dspin_energy,a,N);
	}	  
	//~ Add contributions to traced energy [KH - 20 February 2014]
	if (pairenergy) {
	  /*~ Update the normal contribution to strain energy which 
	    doesn't need to be calculated incrementally*/
	  nstr = 0.4*kn*a*overlap*overlap;                             // peviously, nstr = 0.4*kn*polyhertz*deltan*deltan;
	  if (consideronce) normalstrain += nstr;
	  if (trace_energy) shear[27] = nstr;
	  

	  //************
	  double effectivekt = 8.0*G_star*a;
	  if (xmu > tolerance) {
	    if (effectivekt > tolerance) incdissipf = 0.5*fs_new*fs_new/effectivekt; // this is "SHEAR STRAIN" not frictional dissipation
	    else incdissipf = 0.0;
	    if (consideronce) dissipfriction += incdissipf;
	    if (trace_energy) shear[26] = incdissipf; // not accumulated
	  }
	  //************


	  if (shearupdate) {
	    //~~ Update the spin contribution [MO - 13 November 2014]
	    if (D_spin) {
	      if (consideronce) spinenergy += Dspin_energy;
	      if (trace_energy) shear[29] += Dspin_energy;
	    }
	    /* Full sliding can be considered as accumulation of partial slip for HMD model.
	       Theoretically speaking, full sliding does not take place for HMD modle.
	       Thus, shear strain energy here is summation of shear strain energy + friction energy for shm model.*/ 
	    if (xmu > tolerance) {
	      if (THETA1 == 1) {
		if (fabs(K_T) > tolerance) sstr = 0.5*(fs_new + fs_old)*(fs - fs_old)/K_T;
		else sstr = 0.0;
	      }
	      else if (THETA1 == 0) {
		if (T < 0) fs_new *= -1.0; 
		if (T_old < 0) fs_old *= -1.0; 
		if (dTdisp < 0) dTdisp_mag *= -1.0; // not fabs(dTdisp)!!
		sstr = 0.5*dTdisp_mag*(fs_new + fs_old);
	      }
	      else sstr = 0.0;
	      if (consideronce) shearstrain += sstr;
	      if (trace_energy) shear[28] += sstr;
	    }
	  }
	}
	//*************************************************************************************
	// Update the T_star1 and T_star2 considering the change of the normal contact load
	int sp_Tstar1 = 0;
	if (THETA1 == 0 && xmu > tolerance) {
	  // xmu should be less than 1.0 to use the expression below [MO - 04 Jun 2015]
	  if (T_step_group != 1 && fabs(T_star1) > tolerance) T_star1 += CDF*xmu*dN;
	  if (T_step_group == 3 || T_step_group == 4)         T_star2 -= CDF*xmu*dN; 	
	  //***********************************************************************************
	  // Update T_star1 and T_star2
	  // CDF *= -1 should not be active if sp_Tstar1 = 1 is set. 
	  if (T_step_group == 1) T_star1 = T_star2 = 0.0;
	  else if (T_step_group == 2 && T_step_group_old == 1) {
	    // the first step from T_step = 1: loading to unloading
	    T_star1 = T; 
	    Tdisp_DD = 0.0;
	    sp_Tstar1 = 1;
	    // added to consider the sign of Tdisp correctly [MO - 01 Jun 2015]
	    Tdisp1_star1 = Tdisp1; //[MO - 01 Jun 2015] 
	    Tdisp2_star1 = Tdisp2; //[MO - 01 Jun 2015] 
	    Tdisp3_star1 = Tdisp3; //[MO - 01 Jun 2015] 
	    if (T*Tdisp < 0.0) {
	      Tdisp1_star1 *= - 1; //[MO - 01 Jun 2015] 
	      Tdisp2_star1 *= - 1; //[MO - 01 Jun 2015] 
	      Tdisp3_star1 *= - 1; //[MO - 01 Jun 2015] 
	    } 
	  }
	  else if (T_step_group == 3 && T_step_group_old == 2) {
	    // the first step from T_step = 2: unloading to re-loading
	    T_star2 = T;
	    Tdisp_DD = 0.0;
	    sp_Tstar1 = 1;
	    if (fabs(T_star2) > fabs(T_star1)) { // avoid initial flactuations [MO - 01 Jun 2015]
	      T_star1 = T_star2 = 0.0;
	      Tdisp1_star1 = Tdisp2_star1 = Tdisp3_star1 = 0.0; //[MO - 01 Jun 2015] 
	    }
	  } 
	  //**********************************************************************************
	  if ((T_step_group == 2 || T_step_group == 3 || T_step_group == 4) && sp_Tstar1 == 0) {
	    // Introduce the following rules to avoid numerical errors [MO - 30 March 2015]
	    // [1] CDF*(T*) > CDF*T > CDF*(T**), [2] |T*| > |T**| 
	    if (fabs(T) > fabs(T_star1) && (T_step_group == 2 || T_step_group == 3 || T_step_group == 4)) {
	      if (UFL == 1 && T*T_star1 >= 0.0) T_star1 = T_star2 = 0.0;
	      else if (UFL ==  1 && T*T_star1 <  0.0) T_star1 = - T;
	      else if (UFL == -1 && T*T_star1 >= 0.0) T_star1 =   T;
	      else if (UFL == -1 && T*T_star1 <  0.0) { 
		T_star1 = T_star2 = 0.0;	      
		CDF *= -1;
		// added to consider the sign of Tdisp correctly [MO - 01 Jun 2015]
		Tdisp1_star1 = Tdisp1; //[MO - 01 Jun 2015] 
		Tdisp2_star1 = Tdisp2; //[MO - 01 Jun 2015] 
		Tdisp3_star1 = Tdisp3; //[MO - 01 Jun 2015] 
		if (T*Tdisp < 0.0) {
		  Tdisp1_star1 *= - 1; //[MO - 01 Jun 2015] 
		  Tdisp2_star1 *= - 1; //[MO - 01 Jun 2015] 
		  Tdisp3_star1 *= - 1; //[MO - 01 Jun 2015] 
		} 
	      }
	    }
	    // CDF*T_star2 must be smaller than CDF*T
	    else if (CDF*T_star2 > CDF*T && (T_step_group == 3 || T_step_group == 4)) {
	      if      (UFL ==  1) T_star2 = T;
	      else if (UFL == -1) T_star2 = 0.0;
	    }
	    // |T_star2| must be smaller than |T_star1|
	    else if (fabs(T_star2) >= fabs(T_star1) && (T_step_group == 3 || T_step_group == 4)) {
	      if      (UFL ==  1 && T_star1*T_star2 >= 0.0) T_star1 = T_star2 = 0.0;
	      else if (UFL ==  1 && T_star1*T_star2 <  0.0) T_star2 = - T_star1;
	      else if (UFL == -1 && T_star1*T_star2 >= 0.0) T_star2 = 0.0;
	      else if (UFL == -1 && T_star1*T_star2 <  0.0) { //changed temporary [MO 01June2015]
		T_star1 = T_star2 = 0.0;
		CDF *= -1;
		// added to consider the sign of Tdisp correctly [MO - 01 Jun 2015]
		Tdisp1_star1 = Tdisp1; //[MO - 01 Jun 2015] 
		Tdisp2_star1 = Tdisp2; //[MO - 01 Jun 2015] 
		Tdisp3_star1 = Tdisp3; //[MO - 01 Jun 2015] 
		if (T*Tdisp < 0.0) {
		  Tdisp1_star1 *= - 1; //[MO - 01 Jun 2015] 
		  Tdisp2_star1 *= - 1; //[MO - 01 Jun 2015] 
		  Tdisp3_star1 *= - 1; //[MO - 01 Jun 2015] 
		}
	      } 
	    }
	  }
	}
	//***********************************************************************************

	//*************************************************************************************	    
	// Store the CTD and CDF as absolute values for the next step to avoid the sign changing [MO - 15 December 2014] 
	if (CTD == -1) CTD = 10000;
	if (CDF == -1) CDF = 10000;

	if (shearupdate) {      
	  // shear[0], shear[1] and shear[2] are tangential force of x, y and z directions, respectively.
	  shear[3] = overlap;           // previous overlap
	  shear[4] = Tdisp;             // previous tangential displacement
	  shear[5] = Tdisp1;            // tangential displacement of x direction in the previous step
	  shear[6] = Tdisp2;            // tangential displacement of y direction in the previous step
	  shear[7] = Tdisp3;            // tangential displacement of z direction in the previous step
	  shear[8] = T_star1;           // first reverse point of tangential load (T) from loading to unloading
	  shear[9] = T_star2;           // second reverse point of tangential load from unloading to re-loading
	  shear[10] = 0.0;           
	  shear[11] = CDF;              // CDF indicates wheather system is loading or unloading (int 1 or -1)       
	  shear[12] = T;                // tangential contact force 
	  shear[13] = Tdisp_DD;         // Tdisp_DD <= 0.0 should be satisfied to move on a new loading curve for N+dN    	   
	  shear[14] = CTD;              // +1 and -1 mean positive and negative shear disp, respectively (int 1 or -1)   
	  shear[15] = T_step;           // loading step
	  shear[16] = 0.0;      
	  shear[17] = a;
	  shear[18] = N;
	  shear[19] = 0.0; 
	  shear[20] = 0.0; 
	  shear[21] = ccel;  
	  shear[22] = Tdisp1_star1;
	  shear[23] = Tdisp2_star1;
	  shear[24] = Tdisp3_star1;
	  // shear[25] is empty.
	}
	//*******************************************************************************************
	
	
	if (evflag) ev_tally_gran(i,j,nlocal,fx,fy,fz,x[i][0],x[i][1],x[i][2],
				  radius[i],x[j][0],x[j][1],x[j][2],radius[j]);
      }
    }
  }

  //~ Accumulate per-processor energy terms [KH - 17 October 2014]
  //~~ Added for D_spin model [MO - 19 November 2014]
  gatheredf = gatheredss = gatheredse = 0.0;
  if (pairenergy) {
    MPI_Allreduce(&dissipfriction,&gatheredf,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&shearstrain,&gatheredss,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&spinenergy,&gatheredse,1,MPI_DOUBLE,MPI_SUM,world);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
    

/* ----------------------------------------------------------------------
  global settings
------------------------------------------------------------------------- */

void PairGranHMDHistory::settings(int narg, char **arg)
{

  if (narg != 4) error->all(FLERR,"Illegal pair_style command"); // increased from 3 to 4 [MO - 12 Sep 2015]
  if (domain->dimension == 2) error->all(FLERR,"PairGranHMDHistory formulae only valid for 3D simulations");

  Geq = force->numeric(FLERR,arg[0]);//0.5 * (Gi+Gj)
  Poiseq = force->numeric(FLERR,arg[1]);//0.5 * (Poisi+Poisj)
  xmu = force->numeric(FLERR,arg[2]);
  THETA1 = force->inumeric(FLERR,arg[3]);  // HMD with THETA1 = shm 

  if (Geq < 0.0 || Poiseq < 0.0 || xmu < 0.0 || Poiseq > 0.5 || (THETA1 != 1 && THETA1 != 0)) error->all(FLERR,"Illegal HMD pair parameter values");

  kn = 4.0*Geq / (3.0*(1.0-Poiseq)); // calculate these here to save effort in the compute
  kt = 4.0*Geq / (2.0-Poiseq);

  // convert Kn and Kt from pressure units to force/distance^2

  kn /= force->nktv2p;
  kt /= force->nktv2p;

}

/* ---------------------------------------------------------------------- */

double PairGranHMDHistory::single(int i, int j, int itype, int jtype,
				  double rsq,
				  double factor_coul, double factor_lj,
				  double &fforce)
{
  /*~ This is more straightforward than in the other granular 
    pairstyles as the shear forces are stored in shear and do
    not need to be recalculated [KH - 13 December 2012]*/

  double radi,radj,radsum;
  double r,rinv;
  double ccel,polyhertz;
  double fs1,fs2,fs3,fs,fn;

  double *radius = atom->radius;
  radi = radius[i];
  radj = radius[j];
  radsum = radi + radj;

  double **x = atom->x;
  tagint *tag = atom->tag; //~ Write out the atom tags

  /*The number of optional entries in svector for energy
    tracing and/or D_spin resistance [MO - 19 Novemeber 2014]*/
  int optionalq = 4*trace_energy + 27*D_spin;

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
      strain energy, shear component of strain energy, spin_energy.
      27 optional entries for the D_spin resistance model 
      [MO - 19 November 2014]*/
    if (optionalq > 0)
      for (int q = 0; q < optionalq; q++) svector[q+14] = 0.0;
    
    return 0.0;
  }

  r = sqrt(rsq);
  rinv = 1.0/r;

  // normal force = Hertzian contact

  ccel = kn*(radsum-r)*rinv;
  polyhertz = sqrt((radsum-r)*radi*radj / radsum);
  ccel *= polyhertz;

  // shear history effects
  // neighprev = index of found neigh on previous call
  // search entire jnum list of neighbors of I for neighbor J
  // start from neighprev, since will typically be next neighbor
  // reset neighprev to 0 as necessary

  int *jlist = list->firstneigh[i];
  int jnum = list->numneigh[i];
  double *allshear = list->listgranhistory->firstdouble[i];

  for (int jj = 0; jj < jnum; jj++) {
    neighprev++;
    if (neighprev >= jnum) neighprev = 0;
    if (jlist[neighprev] == j) break;
  }

  /*~ Another 4 shear quantities were added for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 26 + 20*D_spin + 4*trace_energy;
  double *shear = &allshear[numshearquants*neighprev];

  // Call function for spin resistance model [MO - 19 November 2014]
  double **torque = atom->torque;
  double dspin_i[3],dspin_stm,spin_stm,dM_i[3],dM,K_spin,theta_r,M_limit,Dspin_energy,a,N;
   
  //~~ Added for Deresiewicz1954_spin function [MO - 4 November 2014]
  if (D_spin) {
    a = fabs(shear[17]);
    N = fabs(shear[18]);
    Deresiewicz1954_spin(1,i,j,numshearquants,r,torque,shear,dspin_i,
			 dspin_stm,spin_stm,dM_i,dM,K_spin,theta_r,M_limit,Geq,Poiseq,Dspin_energy,a,N);
  }

  /*~ Some of the following are included only for convenience as
    the data could instead be obtained from a dump of the sphere
    coordinates [KH - 13 December 2011]*/
  fforce = ccel;
  svector[0] = shear[0];
  svector[1] = shear[1];
  svector[2] = shear[2];
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
      svector[q+14] = shear[q+26]; 
  
  //~ Add for the Deresiewicz1954_spin model [MO - 19 November 2014]
  if (D_spin) {
    for (int q = 0; q < 3; q++) {
      svector[q+nq+14] = dspin_i[q]; // dspin_i
      svector[q+nq+17] = shear[numshearquants-3+q]; // spin_i
      svector[q+nq+22] = dM_i[q];
      svector[q+nq+25] = shear[numshearquants-16+q];// M_i;
    }
    svector[nq+20] = dspin_stm; // system value 
    svector[nq+21] = spin_stm;  // system value
    svector[nq+28] = dM; // system value 
    svector[nq+29] = shear[numshearquants-4]; // M, system value
    svector[nq+30] = K_spin; 
    svector[nq+31] = theta_r; 
    svector[nq+32] = shear[numshearquants-12]; // step
    svector[nq+33] = M_limit; 
    svector[nq+34] = shear[numshearquants-5]; // M_star1
    svector[nq+35] = shear[numshearquants-6]; // M_star2
    svector[nq+36] = 0;
    svector[nq+37] = 0;
    svector[nq+38] = 0;
    svector[nq+39] = 0;
    svector[nq+40] = 0;
  } 

  return 0.0;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
   ------------------------------------------------------------------------- */

void PairGranHMDHistory::init_style()
{
  int i;

  // error and warning checks

  if (!atom->sphere_flag)
    error->all(FLERR,"Pair granular requires atom style sphere");
  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair granular requires ghost atoms store velocity");

  /* 20 shear quantities were added for D_spin [MO - 19 November 2014]*/
  /*~ Another 4 shear quantities are needed for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 26 + 20*D_spin + 4*trace_energy;

  // need a granular neigh list and optionally a granular history neigh list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->gran = 1;
  if (history) {
    irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->id = 1;
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->granhistory = 1;
    neighbor->requests[irequest]->dnum = numshearquants;
  }

  dt = update->dt;

  // if shear history is stored:
  // if first init, create Fix needed for storing shear history

  if (history && fix_history == NULL) {
    char dnumstr[16];
    sprintf(dnumstr,"%d",numshearquants); //~ Now variable [KH - 23 May 2017]
    char **fixarg = new char*[4];
    fixarg[0] = (char *) "SHEAR_HISTORY";
    fixarg[1] = (char *) "all";
    fixarg[2] = (char *) "SHEAR_HISTORY";
    fixarg[3] = dnumstr;
    modify->add_fix(4,fixarg,1);
    delete [] fixarg;
    fix_history = (FixShearHistory *) modify->fix[modify->nfix-1];
    fix_history->pair = this;
    neighbor->requests[irequest]->fix_history = fix_history;
  }

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
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairGranHMDHistory::write_restart(FILE *fp)
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

void PairGranHMDHistory::read_restart(FILE *fp)
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

void PairGranHMDHistory::write_restart_settings(FILE *fp)
{
  fwrite(&kn,sizeof(double),1,fp);
  fwrite(&kt,sizeof(double),1,fp);
  fwrite(&Geq,sizeof(double),1,fp);
  fwrite(&Poiseq,sizeof(double),1,fp);
  fwrite(&xmu,sizeof(double),1,fp);
  fwrite(&THETA1,sizeof(int),1,fp);

  //~ Added energy terms [KH - 28 February 2014]
  //~~ Added for D_spin model [MO - 13 November 2014]
  fwrite(&gatheredf,sizeof(double),1,fp);
  fwrite(&gatheredss,sizeof(double),1,fp);
  fwrite(&gatheredse,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGranHMDHistory::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&kn,sizeof(double),1,fp);
    fread(&kt,sizeof(double),1,fp);
    fread(&Geq,sizeof(double),1,fp);
    fread(&Poiseq,sizeof(double),1,fp);
    fread(&xmu,sizeof(double),1,fp);
    fread(&THETA1,sizeof(int),1,fp);    

    /*~ Added energy terms. The total energy is read to the root
      proc and is NOT broadcast to all procs as only the total summed
      across all procs is of interest [KH - 28 February 2014]*/
    //~~ Added for D_spin model [MO - 13 November 2014]
    fread(&dissipfriction,sizeof(double),1,fp);
    fread(&shearstrain,sizeof(double),1,fp);
    fread(&spinenergy,sizeof(double),1,fp);
  }
  MPI_Bcast(&kn,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&kt,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&Geq,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&Poiseq,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&xmu,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&THETA1,1,MPI_INT,0,world);
}

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
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
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

  // Modified for HMD and CMD model that temporary use 17 shear quantities [MO - 21 July 2014].
  int numshearquants = 16 + 4*trace_energy;

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
	

//*******************************************************************************************
	/* The main part of the HMD model is wrriten down below as of 13th August 2014 by Masahide [MO]*/ 

//***********************************************************************************************************************************
        // normal force = Hertzian contact 

	double R_star = 1.0 / (1.0/radi + 1.0/radj);  // equivalent radius of spheres
	double E_star = Geq/(1.0-Poiseq);             // equivalent Young's modulus at contact
	double G_star = Geq/(2.0*(2.0-Poiseq));       // equivalent shear modulus at contact
	double pi = 4.0*atan(1.0);                    // pi = 3.1415*******
	double overlap = radsum-r;                    // overlap of spheres
      	double overlap_old = shear[3];                // overlap of spheres at the previous step 
	double a = sqrt(R_star*overlap);              // radius of contact circle
	double a_old = sqrt(R_star*overlap_old);      // radius of contact circle at the previous step
                    
	//kn = 4.0*Geq / (3.0*(1.0-Poiseq))
	//kt = 4.0*Geq / (2.0-Poiseq)
	
	double N = kn*overlap*a;                      // normal force
	double N_old = kn*overlap_old*a_old;          // normal force at the previous step
	double dN = N - N_old;                        // change of normal force
	double K_N = 2.0*E_star*a;                    // normal contact stiffness (tangential, not secant)

	polyhertz = a;                                // re-use polyhertz to adjust the format for shm contact model
	ccel = kn*overlap*a*rinv;                     // previously, ccel = kn*overlap*polyhert*rinv;	

	
//***********************************************************************************************************************************
	// The following codes introduce HMD model for tangential component.
		 
	double T = 0.0;                               // bulk tangential force
	double T_old = shear[12];                     // bulk tangential force at the previous step (including its sign)
	double T_star1 = shear[8];                    // T*:  tangential force at the revasal step from loading to unloading 
	double T_star2 = shear[9];                    // T**: tangential force at the revasal step from unloading to re-loading 
	double T_star3 = shear[10];                   // 0.0 if not active 
	double T_star4 = shear[11];                   // 0.0 if not active
	double Tdisp  = 0.0;                          // bulk tangential displacement	
	double Tdisp_mag =0.0;                        // magnitude of bulk tangential displacement
	double Tdisp_old  = shear[4];                 // bulk tangential displacement at the previous step
	double dTdisp = 0.0;                          // increment of bulk displacement
	double Tdisp1 = 0.0;                          // tangential displacement of x direction
	double Tdisp2 = 0.0;                          // tangential displacement of y direction
	double Tdisp3 = 0.0;                          // tangential displacement of z direction
	double Tdisp1_old = shear[5];                 // tangential displacement of x direction at the previous step
	double Tdisp2_old = shear[6];                 // tangential displacement of y direction at the previous step
	double Tdisp3_old = shear[7];                 // tangential displacement of z direction at the previous step     
	double inter_product = 0.0;                   // inter product between current and previous vectors of tangential displacement
	int T_step = 0;                               // the current loading phase, e.g. 11 = N increasing T increasing
	int T_step_old = static_cast<int>(shear[15]); // the current loading phase at the previous step
	int CTD = static_cast<int>(shear[14]);        // CTD is the direction of tangential displacement 
	int CDF = static_cast<int>(shear[11]);        // CDF is the direction of tangential load-displacement system
	int UFL = 0;                                  // UFL is the direction of tangential load
	int zero = 0;                                 // zero = 1 when tangential displacement is zero to avoid a numerical error 
	
	
	// Firstly, calculate the shear displacement

	                                               
	if (shearupdate){                              // Update the tangential displacement  
	  Tdisp1 = Tdisp1_old + vtr1*dt;               // shear displacement =vtr*dt
	  Tdisp2 = Tdisp2_old + vtr2*dt;
	  Tdisp3 = Tdisp3_old + vtr3*dt;
	  Tdisp_mag = sqrt(Tdisp1*Tdisp1 + Tdisp2*Tdisp2 + Tdisp3*Tdisp3);
                         
	  if (Tdisp_mag <= 1.0e-20){                   // Tdisp = 0.0 cause numerial error later on, so avoid that here
	    Tdisp1 = 1.0e-25;
	    Tdisp2 = 0.0;
	    Tdisp3 = 0.0;
	    Tdisp_mag = 1.0e-25;
	    zero = 1;                                  // zero = 1 is a special case 
	  }
	}else{
	  Tdisp1 = Tdisp1_old;
	  Tdisp2 = Tdisp2_old;
	  Tdisp3 = Tdisp3_old;
	  Tdisp_mag = sqrt(Tdisp1*Tdisp1 + Tdisp2*Tdisp2 + Tdisp3*Tdisp3);
	}    
	
	// inter_product become negative if the direction of tangential loading changes
	inter_product = Tdisp1*Tdisp1_old + Tdisp2*Tdisp2_old + Tdisp3*Tdisp3_old;
	
	if (shearupdate){ 
	  if (Tdisp_mag <= 1.0e-20 && fabs(Tdisp_old) <= 1.0e-20){
	    CTD = 1;                                     // when there is no shear disp at all.
	    CDF = 1;
	  }
	  else if (Tdisp_mag > 1.0e-20 && fabs(Tdisp_old) <= 1.0e-20){    
	    CTD = 1;                                   // this is for the first increment of tangential displacement
	    CDF = 1;
	  }
	  else if (inter_product >= 0.0)                   CTD *= 1;
	  else if (inter_product < 0.0)                    CTD *= -1;
	  else fprintf(screen,"Unexpected case occurred in zone A. ERROR!!");

	  Tdisp =  CTD * Tdisp_mag;                      // give the sign for tangential displacement based on the value of the inter product 
	  dTdisp = Tdisp - Tdisp_old;                    // dTdisp includes direction of tangential displacement (not magnitude)
	}

//***********************************************************************************************************************************
	// Update the dTdisp_DD

	/* Fig.7 of Mindlin & Deresiewicz (1953) explain why this special case is needed.
	Tdisp_DSC is the minimum tangential displacement to move onto a new loading curve of N+dN.
	If the increment of the tangential displacement is less than Tdisp_DSC, the resultant tangential force is less than theoretical value.
	The maximum tangential contact stiffness (8.0*G_star*a) is used untii To catch upthe current tangential force catch up the the new loading curve. */
	
	double Tdisp_DSC = xmu*dN/(8.0*G_star*a);      // Tdisp_DSC is either positive or negative depending upon the sign of dN
	double Tdisp_DD = shear[13] - fabs(dTdisp) + Tdisp_DSC;
	int    special_DD = 0; 
	if (Tdisp_DD < 0.0) Tdisp_DD = 0.0;            // Tdisp_DD <= 0.0 should be satisfied to move onto a new loading curve for N+dN.
	else                special_DD = 1;            // Special case for DD value is now active               
	
//***********************************************************************************************************************************
	double theta1 = 1.0;
	double theta2 = 1.0;
	double theta3 = 1.0;                  
	double theta  = 1.0;                           // reduction ratio of tangential cotnact stiffness due to tangential load 

	if (xmu < 1.0e-20) theta = 1.0;
	else {
	  
//***********************************************************************************************************************************
	  // Idnetify the loading steps for tangential component

			
	  // for the first step of the special case

	  if (dN > 0.0 && special_DD == 1 && T_step_old != 115 && T_step_old != 125 && T_step_old != 135 && T_step_old != 145){
	    if (CDF*dTdisp >= 0.0 && fabs(T_star1) < 1.0e-20 && fabs(T_star2) < 1.0e-20)            T_step = 115;
	    else if (CDF*dTdisp <  0.0 && fabs(T_star2) < 1.0e-20)                                  T_step = 125; 
	    else if (CDF*dTdisp >= 0.0 && fabs(T_old) < fabs(T_star1))                              T_step = 135;
	    else if (CDF*dTdisp <  0.0 && fabs(T_old) <= fabs(T_star1) && fabs(T_star2)>= 1.0e-20)  T_step = 145;
	    else fprintf(screen,"Unexpected case occurs in zone B. ERROR!!");
	    theta = 1.0;
	    
	    // if the special case was invoked in the previous step and is still active
	    
	  }else if (special_DD == 1 && (T_step_old == 115 || T_step_old == 135) && CDF*dTdisp >= 0){  
	    // still in step_115 or 135; if moved to unloading of T, go to the usual case
	    T_step = T_step_old;
	    theta = 1.0;
	    
	  }else if (special_DD == 1  && (T_step_old == 125 || T_step_old == 145) && CDF*dTdisp < 0){   
	    // still in step_125 or 145; if moved to re-loading of T, go to the usual case
	    T_step = T_step_old;                                          
	    theta = 1.0;
	    
	  }else if (special_DD == 1 && T_step_old == 115 && CDF*dTdisp < 0){
	    // the first step from loading of special case to unloading of special case.
	    T_step = 125;
	    theta = 1.0;
	    
	  }else if (special_DD == 1 && T_step_old == 125 && CDF*dTdisp >= 0){
	    // the first step from unloading of special case to reloading of special case.
	    T_step = 135;
	    theta = 1.0;
	    
	}else if (special_DD == 1 && T_step_old == 135 && CDF*dTdisp < 0){
	    // the first step from reloading  of special case to re-unloading of special case.
	    T_step = 145;
	    theta = 1.0;
	    
	  }else {
	    
	    // Usual cases 
	    // T loading: T_step = *1 
	    
	    if (CDF*dTdisp >= 0.0 && fabs(T_star1) < 1.0e-20 && fabs(T_star2) < 1.0e-20){ 
	      theta1 = 1.0-(CDF*T_old+xmu*dN)/(xmu*N);	
	      if (theta1 <= 0.0) theta1 = theta = 0.0;
	      else theta = pow(theta1, 1.0/3.0);
	      if       (dN >  1.0e-20)      T_step = 11;   // N increasing
	      else if  (dN < -1.0e-20)      T_step = 21;   // N decreasing
	      else                          T_step =  1;   // N constant
	    }
	    
	    // T unloading: T_step = *2 
	    
	    else if (CDF*dTdisp < 0.0 && fabs(T_star2) < 1.0e-20){
	      // For the first step from loading to unloading, T* is still null, which should be T* = T_old.
	      if  (fabs(T_star1) > -1.0e-20) theta2 = 1.0-(CDF*(T_star1-T_old)+2.0*xmu*dN)/(2.0*xmu*N);
	      else                           theta2 = 1.0-(2.0*xmu*dN)/(2.0*xmu*N);
	      if (theta2 <= 0.0) theta2 = theta = 0.0;	  
	      else               theta  = pow(theta2, 1.0/3.0);
	      if      (dN >  1.0e-20) T_step = 12; 
	      else if (dN < -1.0e-20) T_step = 22;      
	      else                    T_step =  2;
	    }
	    
	    // T re-loading: T_step = *3
	    
	    else if (CDF*dTdisp >= 0.0 && fabs(T_old) < fabs(T_star1)){
	      // For the first step from unloading to re-loading, T** is still null, which should be T** = T_old.
	      if  (fabs(T_star2) > -1.0e-20) theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu*dN)/(2.0*xmu*N);
	      else                           theta3 = 1.0-(2.0*xmu*dN)/(2.0*xmu*N);
	      if (theta3 <= 0.0) theta3 = theta = 0.0;
	      else               theta  = pow(theta3, 1.0/3.0);
	      if      (dN >  1.0e-20) T_step = 13; 
	      else if (dN < -1.0e-20) T_step = 23;     
	      else                    T_step = 3;
	    }
	    
	    // T re-unloading: T_step = *4
	    
	    else if (CDF*dTdisp < 0.0 && fabs(T_old) < fabs(T_star1) && fabs(T_star2) >=  1.0e-20){
	      // This part is same with the above (T_step = *3) due to simplification.
	      theta3 = 1.0-(CDF*(T_old-T_star2)+2.0*xmu*dN)/(2.0*xmu*N);
	      if (theta3 <= 0.0) theta3 = theta = 0.0;
	      else               theta  = pow(theta3, 1.0/3.0);
	      if      (dN >  1.0e-20) T_step = 14;
	      else if (dN < -1.0e-20) T_step = 24; 
	      else                    T_step =  4;      
	    }  
	    else fprintf(screen,"Unexpected case occurred in zone C. ERROR!!");
	  }

//*********************************************************************************************
	  //UFL is neccesary for the calculation of tangential contact stiffness
	  
	  if (T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) UFL = -1;
	  else                                                              UFL =  1;
	}       // end of if function for xmu < 1.0e-20
	
//***********************************************************************************************************************************
	// Calculate the tangential contact stiffness and the resultant tangential contact force
  
	double K_T = 0.0;                             // tangential contact stiffness
	double dT = 0.0;                              // increment of tangential contact force
	double fs_ratio1;                             // ratio to re-scale the tangential force to avoid inconsistent magnitude of tangential force
	double fs_ratio2;                             // ratio to re-scale the tangential force if full sliding take places
	double K_R;                                   // ratio of tangential contact stiffness over one for shm model
	
	// update T_old considring the rotation of contact plane which was done prior to the calculation of contact force
	if (T_old >= 0) T_old =   sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
	else            T_old = - sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
	
	if (theta > 1.0) theta = 1.0;

	if (zero == 0) K_T = 8.0*G_star*theta*a + CDF*UFL*xmu*(1.0-theta)*dN/dTdisp;
	else             K_T = 0.0;                   // zero == 1 means dTdisp is zero and K_T becomes inf
	
	K_R = K_T / (8.0*G_star*a);

	if (shearupdate){
	    if (zero == 0){
	     	      
	      dT = K_T * dTdisp; 
	      T = dT + T_old;

	      // update the tangential contact force by distributing dT into 3 component based on imcrese of relative rotational velosity
	      shear[0] -= K_T * vtr1 * dt; 
	      shear[1] -= K_T * vtr2 * dt;
	      shear[2] -= K_T * vtr3 * dt;
	     
	      //rescale the component
	      fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
	      fs_ratio1 = fabs(T)/fs;
	      shear[0] *= fs_ratio1; 
	      shear[1] *= fs_ratio1;
	      shear[2] *= fs_ratio1; 
	    }
	    
	} else T = T_old;
	
//***********************************************************************************************************************************
        // rescale tangential force if full sliding takes place 

	fslim = xmu * N;                               // ccel*r = N
	fs = fabs(T);                                  // re-use fs for re-scale the tangential force         
	fs_ratio2 = fslim/fs;

        if (fs > fslim) {
          if (fs > 1.0e-20) {
	    if (shearupdate) {
	      T *= fs_ratio2;
	      shear[0] *= fs_ratio2; 
	      shear[1] *= fs_ratio2;
	      shear[2] *= fs_ratio2; 
	          
	    }
          } else shear[0] = shear[1] = shear[2] = 0.0;
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

        if (j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
	  torque[j][0] -= crj*tor1;
	  torque[j][1] -= crj*tor2;
	  torque[j][2] -= crj*tor3;
        }

	double dur[3], dus[3], localdM[3], globaldM[3]; //~ Pass by reference
	double effectivekt = K_T;
	double aveshearforce, slipdisp, oldshearforce, newshearforce;
	double incdissipf, nstr, sstr, incrementaldisp, rkt;
	
	if (j < nlocal || tag[j] < tag[i]) {//~ Consider contacts only once
	  
	  //~ Add contributions to traced energy [KH - 20 February 2014]
	  if (pairenergy) {
	    
	    /*~ Update the normal contribution to strain energy which 
	      doesn't need to be calculated incrementally*/
	    nstr = 0.4*kn*a*overlap*overlap;                             // peviously, nstr = 0.4*kn*polyhertz*deltan*deltan;
	    normalstrain += nstr;
	    if (trace_energy) shear[17] = nstr;

	    /* Full sliding can be considered as accumulation of partial slip for HMD model.
	       Theoretically speaking, full sliding does not take place for HMD modle.
	       Thus, shear strain energy here is summation of shear strain energy + friction energy for shm model.*/
	                                      
	    sstr = 0.5*dTdisp*(T + T_old);
	    shearstrain += sstr;
	    if (trace_energy) shear[18] += sstr;
	  }
	}
	
//*********************************************************************************************************
	// Update T_star1 and T_star2

	int sp_Tstar1 = 0;                         // CDF *= -1 should not be active if sp_Tstar1 = 1 is set.  
	
	if (T_step == 1 || T_step == 11 || T_step == 21 || T_step == 115){   // do not forget to include T_step == 115
	  T_star1 = 0.0;
	  T_star2 = 0.0;
	}

	// Update the T_star1 and T_star2 for new loading curve; move to loading step
	
	if ((T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) && fabs(T) > fabs(T_star1) && fabs(T_star1) > 1.0e-20){
	  T_star1 = 0.0;
	  T_star2 = 0.0;
	}
	
	// Update the T_star2 for new unloading curve; move to unloading step
	
	if ((T_step == 4 || T_step == 14 || T_step == 24 || T_step == 145) && fabs(T) < fabs(T_star2) && fabs(T_star2) > 1.0e-20){
	  T_star2 = 0.0;
	}
	
	if ((T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) && fabs(T_star1) <= 1.0e-20  && fabs(T_star2) <= 1.0e-20 && fabs(T_star2) <= 1.0e-20){
	  // the first step from T_step = 1; loading to unloading
	  T_star1 = T_old;
	  Tdisp_DD = 0.0;
	  sp_Tstar1 = 1;
	} 
	
	if ((T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) && fabs(T_star1) > fabs(T) && fabs(T_star2) <= 1.0e-20){  
	  // the first step from T_step = 2; unloading to re-loading
	  T_star2 = T_old;
	  Tdisp_DD = 0.0;
	} 


//*********************************************************************************************************
	// Update the T_star1 and T_star2 considering the change of the normal contact load
	
	if (T_step != 1 && T_step != 11 && T_step != 21 && T_step != 115 && fabs(T_star1)> 1.0e-20) T_star1 += CDF*xmu*dN;
       	if ((T_step == 3 || T_step == 13 || T_step == 23 || T_step == 135) && fabs(T_star2) > 1.0e-20) T_star2 -= CDF*xmu*dN;
	
 //*********************************************************************************************************


	//special case when |T| for unloading reaches -|T_star1|  // move on to a new loading curve in the opposite direction
	if ((T_step == 2 || T_step == 12 || T_step == 22 || T_step == 125) && fabs(T) > fabs(T_star1) && T_star1 > 1.0e-20){
	  
	  // fprintf(screen,"timestep %i dT %1.6e CDF*xmu*dN %1.6e  T  %1.6e T_old %1.6e T_star1 %1.6e \n \n",update->ntimestep,dT,CDF*xmu*dN,T,T_old,T_star1);

	  T_star1 = 0.0;
	  T_star2 = 0.0;

	  // avoid the case when T* > T > T*-xmu*dN 
	  if (sp_Tstar1 == 0)  CDF *= -1;  // the system is moved onto virgin loading curve
	}

//*********************************************************************************************************
	// output the values

	double lim = xmu*N;
	double slip = T/lim;
	double sp = Tdisp_DD;
	
	if (update->ntimestep % 1 == 0){
	  if (tag[i] == 1)
	    fprintf(screen,"step %i slip %1.6e K_R %1.6e T* %1.6e Tdisp %1.6e T %1.6e dTdisp %1.6e T_old %1.6e Tdisp_DD %1.6e lim %1.6e t %i \n",T_step,slip,K_R,T_star1,Tdisp,T,dTdisp,T_old,Tdisp_DD,-lim,update->ntimestep);
	}
	
//*******************************************************************************************	    
	/*~~ modified for MD model [MO - 17 July 201]4 ~~*/
	
	overlap_old = overlap;
	T_old = T;
	if (zero == 1)  Tdisp_old = Tdisp1_old = Tdisp2_old = Tdisp3_old = 0.0;
	else{
	  Tdisp_old  = Tdisp;
	  Tdisp1_old = Tdisp1;
	  Tdisp2_old = Tdisp2;
	  Tdisp3_old = Tdisp3;
	}
		
	// shear[0], shear[1] and shear[2] are tangential force of x, y and z directions, espectively.
	
	shear[3] = overlap_old;           // previous overlap
	shear[4] = Tdisp_old;             // previous tangential displacement
	shear[5] = Tdisp1_old;            // tangential displacement of x direction in the previous step
	shear[6] = Tdisp2_old;            // tangential displacement of y direction in the previous step
	shear[7] = Tdisp3_old;            // tangential displacement of z direction in the previous step
       	shear[8] = T_star1;               // max. tangential load in the history = first reverse point of tangential load (T) from loading to unloading
	shear[9] = T_star2;               // second reverse point of tangential load from unloading to re-loading
	shear[10] = 0.0;                  // T_star3; // third reverse point of tangential load from re-loading to re-unloading        [optional]
	shear[11] = CDF;                  // CDF indicates wheather system is loading or unloading //T_star4;       
                                          // T_star4; 4th reverse point of tangential load from re-unloading to re-re-loading    [optional]
	shear[12] = T_old;       
	shear[13] = Tdisp_DD;             // Tdisp_DD <= 0.0 should be satisfied to move on a new loading curve for N+dN 
	shear[14] = CTD;                  // +1 and -1 mean positive and negative shear disp, respectively.       
	shear[15] = T_step;
	      

//*******************************************************************************************

	if (evflag) ev_tally_gran(i,j,nlocal,fx,fy,fz,x[i][0],x[i][1],x[i][2],
				      radius[i],x[j][0],x[j][1],x[j][2],radius[j]);
      }
    }
  }
}
    

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGranHMDHistory::settings(int narg, char **arg)
{

  if (narg != 3) error->all(FLERR,"Illegal pair_style command");
  if (domain->dimension == 2) error->all(FLERR,"PairGranHMDHistory formulae only valid for 3D simulations");

  Geq = force->numeric(FLERR,arg[0]);//0.5 * (Gi+Gj)
  Poiseq = force->numeric(FLERR,arg[1]);//0.5 * (Poisi+Poisj)
  xmu = force->numeric(FLERR,arg[2]);

  if (Geq < 0.0 || Poiseq < 0.0 || xmu < 0.0 || Poiseq > 0.5) error->all(FLERR,"Illegal HMD pair parameter values");

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

  /*~ The number of optional entries in svector for energy
    tracing and/or rolling resistance [KH - 6 March 2014]*/
  int optionalq = 4*trace_energy;

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

      25 optional entries for the rolling resistance model 
      [KH - 25 October 2013]. Order: 
      dUr[*], accumulated dUr[*], dUs[*], accumulated dUs[*], 
      localdM[*], accumulated localdM[*], globaldM[*], accumulated 
      globaldM[*],ksbar*/
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
  int *touch = list->listgranhistory->firstneigh[i];
  double *allshear = list->listgranhistory->firstdouble[i];

  for (int jj = 0; jj < jnum; jj++) {
    neighprev++;
    if (neighprev >= jnum) neighprev = 0;
    if (touch[neighprev] == j) break;
  }

    /*~ Another 4 shear quantities were added for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 16 + 4*trace_energy;
  double *shear = &allshear[numshearquants*neighprev];

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
      svector[q+14] = shear[q+16]; //~ 4 rather than 3...

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

  /*~ Have 17 shear quantities if rolling resistance is included
    [KH - 24 October 2013]*/
  /*~ Another 4 shear quantities are needed for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 16 + 4*trace_energy;

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
    char **fixarg = new char*[4]; //~ Also increased [KH - 23 November 2012]
    fixarg[0] = (char *) "SHEAR_HISTORY";
    fixarg[1] = (char *) "all";
    fixarg[2] = (char *) "SHEAR_HISTORY";

    //~ Carry out the necessary string conversion [KH - 6 March 2014]
    char nsq[5] = {0};
    sprintf(nsq,"%i",numshearquants);
    fixarg[3] = nsq; //~ Changed this condition
    modify->add_fix(4,fixarg,suffix); //~ Increased to 4
    delete [] fixarg;
    fix_history = (FixShearHistory *) modify->fix[modify->nfix-1];
    fix_history->pair = this;
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

void PairGranHMDHistory::write_restart_settings(FILE *fp)
{
  fwrite(&kn,sizeof(double),1,fp);
  fwrite(&kt,sizeof(double),1,fp);
  fwrite(&Geq,sizeof(double),1,fp);
  fwrite(&Poiseq,sizeof(double),1,fp);
  fwrite(&xmu,sizeof(double),1,fp);

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

void PairGranHMDHistory::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&kn,sizeof(double),1,fp);
    fread(&kt,sizeof(double),1,fp);
    fread(&Geq,sizeof(double),1,fp);
    fread(&Poiseq,sizeof(double),1,fp);
    fread(&xmu,sizeof(double),1,fp);

    /*~ Added energy terms. The total energy is read to the root
    proc and is NOT broadcast to all procs as only the total summed
    across all procs is of interest [KH - 28 February 2014]*/
    fread(&dissipfriction,sizeof(double),1,fp);
    fread(&shearstrain,sizeof(double),1,fp);
  }
  MPI_Bcast(&kn,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&kt,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&Geq,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&Poiseq,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&xmu,1,MPI_DOUBLE,0,world);
}

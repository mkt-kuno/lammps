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
#include "pair_gran_CM_history.h"
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

PairGranCMHistory::PairGranCMHistory(LAMMPS *lmp) :
  PairGranHookeHistory(lmp) {}

/* ---------------------------------------------------------------------- */

void PairGranCMHistory::compute(int eflag, int vflag)
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

  /*~ 4 shear quantities were added for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 5 + 4*trace_energy;  /*~~ Updated to 5 [MO June 2014] ~~*/

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
	//#############################################################################################
	for (int q = 0; q < numshearquants; q++) 
	  //if overlap_max should be remembered after separating a contact
	  //replace this line by "for (int q = 0; q < numshearquants-2; q++)" 
	  //##################################################################
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

        shsqmag = shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]; //oldshearforce 

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

	/*~~ CM model for normal component (O'Donovan, 2013) [MO 2/Jun/2014] ~~*/ 
	
	double RMSf_eq = sqrt(2.0)*RMSf;                             // The equivalent roughness for the rough-rough contact
        double overlap = radsum-r;
        double R_star = 1.0 / (1.0/radi + 1.0/radj);
        double E_star = Geq/(1.0-Poiseq);
	double pi = 4.0*atan(1.0);
        double overlap_p = R_star*pow(3.0/4.0*pi*Hp/E_star,2.0);     // R_star*(3/4*pi*Hp/E_star)^2
	double P_GT = 100.0*RMSf_eq*E_star*pow(2.0*R_star*RMSf_eq,0.5);
	double overlap_GT = pow(3.0*P_GT/(4.0*sqrt(R_star)*E_star),2.0/3.0)+overlap_p; 
	double b = 2.0*E_star*sqrt(R_star*(overlap_GT-overlap_p))*overlap_GT/P_GT; 
	double overlap_max = shear[3];
	double overlap_old = shear[4];
	double effectivekt;
	double P_max;
	double overlap_shm;
	double overlap_effective;
	int step;


	/*~~Update the overlap which is the maximum in the history. [MO - 5 June 2014] ~~*/
	
	if (overlap >= overlap_max) overlap_max = overlap;
	
	/*~~ step 1*, 2* and 3* stand for loading, unloading and reloasing, respectively. 
	  step *4, *5 and *6 stand for asperity contact, Hertzian contact and zero force, respectively.
	  Possible conbinations are 14,15,25,26,35 and 36.  [MO - 18 Jun 2014]~~*/
	
	P_max = P_GT * pow(overlap_GT, -b) * pow(overlap_max, b);    // CM code was modified [MO - 24Jun2014]
	overlap_shm = pow(P_max/(kn*sqrt(R_star)),2.0/3.0);              
	overlap_effective = overlap_max - overlap_shm;      

	
	if (overlap >= overlap_GT){
	  if (overlap >= overlap_max - 1.0e-13) step = 15;
	  if (overlap < overlap_max) {
	    if (overlap >= overlap_old) step = 35;
	    if (overlap < overlap_old) step = 25;
	  }
	  ccel = kn*(overlap-overlap_p)*rinv;
	  polyhertz = sqrt((overlap-overlap_p)*R_star); //[MO - 18 July 2014] R_star is now used
	  ccel *= polyhertz;
	  if (shearupdate) {
	    effectivekt = polyhertz*kt;
	    shear[0] -= effectivekt*vtr1*dt;//shear displacement =vtr*dt
	    shear[1] -= effectivekt*vtr2*dt;
	    shear[2] -= effectivekt*vtr3*dt;
	  }
	}
	if (overlap < overlap_GT){
	  if (overlap >= overlap_max - 1.0e-13){ 
	    step = 14;
	    ccel = P_GT * pow(overlap_GT, -b) * pow(overlap, b)*rinv;        /*~~ Do not forget to add *rinv for LAMMPS ~~*/ 
	    if (shearupdate) {
	      effectivekt = 2.0*b*(1.0-Poiseq)/(2.0-Poiseq)*ccel*r*(1/overlap); // Corrected on 21 July 2014 [MO]
	      shear[0] -= effectivekt*vtr1*dt;//shear displacement =vtr*dt
	      shear[1] -= effectivekt*vtr2*dt;
	      shear[2] -= effectivekt*vtr3*dt;
	    } 
	  }
	  if (overlap < overlap_max - 1.0e-13){ 
	    if (overlap < overlap_effective){
	      if (overlap >= overlap_old) step = 36;
	      if (overlap < overlap_old) step = 26;
	      ccel = 0.0;                   /*~~ Particles are separated due to squahed asperities ~~*/
	      if (shearupdate) {
		effectivekt = 0.0;
		shear[0] -= effectivekt*vtr1*dt;//shear displacement =vtr*dt
		shear[1] -= effectivekt*vtr2*dt;
		shear[2] -= effectivekt*vtr3*dt;
	      }
	    }
	    if (overlap >= overlap_effective){
	      if (overlap >= overlap_old) step = 35;
	      if (overlap < overlap_old) step = 25;
	      ccel = kn*(overlap-overlap_effective)*rinv;
	      polyhertz = sqrt((overlap-overlap_effective)*R_star); // [MO - 18 July 2014] R_star is now used
	      ccel *= polyhertz;
	      if (shearupdate) {
		effectivekt = polyhertz*kt;
		shear[0] -= effectivekt*vtr1*dt;//shear displacement =vtr*dt
		shear[1] -= effectivekt*vtr2*dt;
		shear[2] -= effectivekt*vtr3*dt;
	      }
	    }
	  }
	}
	
     	/*~~ Update the overlap_old for the next step for CM model [MO - 5 June 2014] ~~*/	
	
	overlap_old = overlap;                     
	
	
	// rescale frictional forces if needed
	
	fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
	fslim = xmu * fabs(ccel*r); //replaced fn with fslim
	
	if (fs > fslim) {
	  if (fs != 0.0) {
	    if (shearupdate) {
	      shear[0] *= fslim/fs;
	      shear[1] *= fslim/fs;
	      shear[2] *= fslim/fs;
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
	double aveshearforce, slipdisp, oldshearforce, newshearforce;
	double incdissipf, nstr, sstr, incrementaldisp, rkt;
	double b1inv = 1.0/(b+1.0);
	
	if (j < nlocal || tag[j] < tag[i]) {//~ Consider contacts only once
		  
	  //~ Add contributions to traced energy [KH - 20 February 2014]
	  if (pairenergy) {
	   	    
	    /*~ Increment the friction energy only if the slip condition
	      is invoked*/
	    if (fs > fslim && fslim > 0.0) {
	      //~ current shear displacement = fslim/effectivekt;
	      slipdisp = (fs-fslim)/effectivekt;
	      aveshearforce = 0.5*(fs + fslim);
	      
	      //~ slipdisp and aveshearforce are both positive
	      incdissipf = aveshearforce*slipdisp;
	      dissipfriction += incdissipf;
	      if (trace_energy) shear[5] += incdissipf;
	    }
	    
	    /*~ Update the normal contribution to strain energy which 
	      doesn't need to be calculated incrementally*/
	    /*~~ However, the hysterisys should be considered using CM model [MO - 18 Jun 2014] ~~*/

	    if(step == 14){        // 14 stands for loading on asperity contact   [MO - 18 Jun 2014]
	      nstr = P_GT*b1inv*pow(overlap_GT,-b)*pow(overlap,b+1.0);
	      normalstrain += nstr;
	    }
	    if(step == 15){        // *5 stands for hertzian contact   [MO - 18 Jun 2014]
	    // overlap-overlap_p for Hertzian curve [MO - 18 Jun 2014]
	      nstr = P_GT*overlap_GT*b1inv-0.4*kn*sqrt(R_star)*(pow(overlap_GT-overlap_p,5.0/2.0)-pow(overlap-overlap_p,5.0/2.0));     
	      normalstrain += nstr;
	    }	    
	    if(step == 26 || step == 36){    // 2* and 3* stand for unloading and reloading. *6 means ccel = 0.0 which should not be used in this group [MO - 18 Jun 2014] 
	      if(overlap_max < overlap_GT){
		nstr = P_GT*b1inv*pow(overlap_GT,-b)*pow(overlap_max,b+1.0)-0.4*kn*sqrt(R_star)*pow(overlap_max - overlap_effective,5.0/2.0);
		normalstrain += nstr;
	      }
	      if(overlap_max >= overlap_GT){
		nstr = P_GT*overlap_GT*b1inv-0.4*kn*sqrt(R_star)*pow(overlap_GT-overlap_p,5.0/2.0);
	      	normalstrain += nstr;
	      }
	    }
	    if(step == 25 || step == 35){        // *5 stands for hertzian contact [MO - 18 Jun 2014]
	    // overlap-overlap_effective for Hertzian contact [MO - 18 Jun 2014]
	      if(overlap_max < overlap_GT){
		nstr = P_GT*b1inv*pow(overlap_GT,-b)*pow(overlap_max,b+1.0)-0.4*kn*sqrt(R_star)*(pow(overlap_max-overlap_effective,5.0/2.0)-pow(overlap-overlap_effective,5.0/2.0));      
		normalstrain += nstr;
	      }	    
	      if(overlap_max >= overlap_GT){
		nstr = P_GT*overlap_GT*b1inv-0.4*kn*sqrt(R_star)*(pow(overlap_GT-overlap_p,5.0/2.0)-pow(overlap-overlap_p,5.0/2.0));
		normalstrain += nstr;
	      }
	    }
	    
	    if (trace_energy) shear[6] = nstr;
	    

	    //~ The shear component does require incremental calculation
	    if (shearupdate) {
	      oldshearforce = sqrt(shsqmag);
	      newshearforce = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
	      if (effectivekt != 0.0){
	        // effectivekt = 0 should be avoided.
		incrementaldisp = (newshearforce - oldshearforce)/effectivekt;   
	      }else{incrementaldisp = 0.0;}  
	      // because no incremental shear force [MO - 21 July 2014]
	      sstr = 0.5*incrementaldisp*(newshearforce + oldshearforce);
	      shearstrain += sstr;
	      if (trace_energy) shear[7] += sstr;
	    }
	  }
	}
	
	
//*******************************************************************************************
	// output the values

	double fn = ccel*r;
	double dTdisp1 = vtr1*dt; 
	double dTdisp2 = vtr2*dt; 
	double dTdisp3 = vtr3*dt;
	
	fs = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]); 

	if (update->ntimestep % 1 == 0){
	  if (tag[i] == 1 && tag[j] == 2 || tag[i] == 2 && tag[j] == 1)
	    fprintf(screen,"overlap %1.6e N %1.6e Tdisp 0 T %1.6e t = %i t1 %1.6e t2 %1.6e t3 %1.6e fs %1.6e disp1 %1.6e disp2 %1.6e disp3 %1.6e \n",overlap,fn,newshearforce,update->ntimestep,shear[0],shear[1],shear[2],fs,dTdisp1,dTdisp2,dTdisp3);
	}
	
//*******************************************************************************************	

	//~ Assign current polyhertz value to shear[3] [KH - 23 November 2012]
	
	shear[3] = overlap_max;     /*~~ 2 quantities are added ~~*/
	shear[4] = overlap_old;
	
	if (evflag) ev_tally_gran(i,j,nlocal,fx,fy,fz,x[i][0],x[i][1],x[i][2],radius[i],x[j][0],x[j][1],x[j][2],radius[j]);
      }
    }
  }

  //~ Accumulate per-processor energy terms [KH - 17 October 2014]
  gatheredf = gatheredss = 0.0;
  if (pairenergy) {
    MPI_Allreduce(&dissipfriction,&gatheredf,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&shearstrain,&gatheredss,1,MPI_DOUBLE,MPI_SUM,world);
  }
}


/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGranCMHistory::settings(int narg, char **arg)
{

  if (narg != 5) error->all(FLERR,"Illegal pair_style command");
  if (domain->dimension == 2) error->all(FLERR,"PairGranCMHistory formulae only valid for 3D simulations");

  Geq = force->numeric(FLERR,arg[0]);//0.5 * (Gi+Gj)
  Poiseq = force->numeric(FLERR,arg[1]);//0.5 * (Poisi+Poisj)
  xmu = force->numeric(FLERR,arg[2]);
  RMSf = force->numeric(FLERR,arg[3]);//RMSf value for a particle where all the particles are assumed to have the same value
  Hp = force->numeric(FLERR,arg[4]);


  if (Geq < 0.0 || Poiseq < 0.0 || xmu < 0.0 || Poiseq > 0.5 || RMSf < 0.0 || Hp < 0.0) error->all(FLERR,"Illegal CM pair parameter values");

  kn = 4.0*Geq / (3.0*(1.0-Poiseq)); // calculate these here to save effort in the compute
  kt = 4.0*Geq / (2.0-Poiseq);

  // convert Kn and Kt from pressure units to force/distance^2

  kn /= force->nktv2p;
  kt /= force->nktv2p;

}

/* ---------------------------------------------------------------------- */

double PairGranCMHistory::single(int i, int j, int itype, int jtype,
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
    tracing [KH - 6 March 2014]*/
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
      contributions (e.g., from rolling resistance model)*/

    if (optionalq > 0)
      for (int q = 0; q < optionalq; q++) svector[q+14] = 0.0;
    
    return 0.0;
  }

  r = sqrt(rsq);
  rinv = 1.0/r;

  // normal force = CM contact


  /*~~ CM model for normal component (O'Donovan, 2013) [MO 2/Jun/2014] ~~*/ 

     
  double overlap = radsum-r;
  double R_star = 1.0 / (1.0/radi + 1.0/radj);
  double E_star = Geq/(1.0-Poiseq);
  double pi = 4.0*atan(1.0);
  double overlap_p = R_star*pow(3.0/4.0*pi*Hp/E_star,2.0);     // R_star*(3/4*pi*Hp/E_star)^2
  double P_GT = 100.0*RMSf*E_star*sqrt(2.0*R_star*RMSf);
  double overlap_GT = pow(3.0*P_GT/(4.0*sqrt(R_star)*E_star),2.0/3.0)+overlap_p; 
  double b = 2.0*E_star*sqrt(R_star*(overlap_GT-overlap_p))*overlap_GT/P_GT;

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

  /*~ 4 shear quantities were added for per-contact energy
    tracing [KH - 6 March 2014]*/
  int numshearquants = 5 + 4*trace_energy;   /*~~ [MO - 5 June 2014] ~~ */
  double *shear = &allshear[numshearquants*neighprev];

  double overlap_max = shear[3];
  double overlap_old = shear[4];	


  /*~~Update the overlap which is the maximum in the history. [MO - 5 June 2014] ~~*/
  
  if (overlap >= overlap_max) 
    overlap_max = overlap;
  
  
  /*~~ Case of Loading or Reloading~~*/
  
  if (overlap >= overlap_old){  
    if (overlap >= overlap_GT){
      ccel = kn*(overlap-overlap_p)*rinv;
      polyhertz = sqrt((overlap-overlap_p)*R_star);  // [MO - 18 July 2014] R_star is now used
      ccel *= polyhertz;
    }
    if (overlap < overlap_GT){
      if (overlap >= overlap_max - 1.0e-13){ 
	ccel = P_GT * pow(overlap_GT, -b) * pow(overlap, b)*rinv;        /*~~ Do not forget to add *rinv for LAMMPS ~~*/ 
      }
      if (overlap < overlap_max - 1.0e-13){ 
	if (overlap < overlap_p){
	  ccel = 0.0;                   /*~~ Particles are separated due to squahed asperities ~~*/
	}
	if (overlap >= overlap_p){
	  ccel = kn*(overlap-overlap_p)*rinv;
	  polyhertz = sqrt((overlap-overlap_p)*R_star);  // [MO - 18 July 2014] R_star is now used
	  ccel *= polyhertz;
	}
      }
    }
  }
	
  /*~~ Case of  Unloading ~~*/
  
  if  (overlap < overlap_old){
    if (overlap < overlap_p){
      ccel = 0.0;                   /*~~ Particles are separated due to squahed asperities ~~*/
    }
    else{
      ccel = kn*(overlap-overlap_p)*rinv;
      polyhertz = sqrt((overlap-overlap_p)*R_star);   // [MO - 18 July 2014] R_star is now used
      ccel *= polyhertz;
    }
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
  if (trace_energy)
    for (int q = 0; q < 4; q++)
      svector[q+14] = shear[q+5]; //~ 5 rather than 3...
  
  return 0.0;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGranCMHistory::init_style()
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
  int numshearquants = 5 + 4*trace_energy;  /*~~ updated to 5 [MO 04 June 2014] ~~*/

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
    modify->add_fix(4,fixarg,1); //~ Increased to 4
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

void PairGranCMHistory::write_restart(FILE *fp)
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

void PairGranCMHistory::read_restart(FILE *fp)
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

void PairGranCMHistory::write_restart_settings(FILE *fp)
{
  fwrite(&kn,sizeof(double),1,fp);
  fwrite(&kt,sizeof(double),1,fp);
  fwrite(&Geq,sizeof(double),1,fp);
  fwrite(&Poiseq,sizeof(double),1,fp);
  fwrite(&xmu,sizeof(double),1,fp);
  fwrite(&RMSf,sizeof(double),1,fp);
  fwrite(&Hp,sizeof(double),1,fp);

  //~ Added energy terms [KH - 28 February 2014]
  fwrite(&gatheredf,sizeof(double),1,fp);
  fwrite(&gatheredss,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGranCMHistory::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&kn,sizeof(double),1,fp);
    fread(&kt,sizeof(double),1,fp);
    fread(&Geq,sizeof(double),1,fp);
    fread(&Poiseq,sizeof(double),1,fp);
    fread(&xmu,sizeof(double),1,fp);
    fread(&RMSf,sizeof(double),1,fp);
    fread(&Hp,sizeof(double),1,fp);

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
  MPI_Bcast(&RMSf,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&Hp,1,MPI_DOUBLE,0,world);
}

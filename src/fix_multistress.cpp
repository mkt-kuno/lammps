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
   Contributing author: Kevin Hanley (Imperial)
------------------------------------------------------------------------- */

#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "atom.h"
#include "modify.h"
#include "fix_multistress.h"
#include "fix_deform.h"
#include "compute.h"
#include "error.h"
#include "update.h"
#include "domain.h"
#include "lattice.h"
#include "force.h"
#include "pair.h"
#include "integrate.h"
#include "fix_crushing.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG   1.0e20

/* ---------------------------------------------------------------------- */

FixMultistress::FixMultistress(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  /*~ The syntax for the fix multistress command is as follows, where
    those keywords in brackets are optional:
 
    fix ID group-ID multistress %_tolerance extra_steps {max_rate}
    {{x/xx}/{y/yy}/{z/zz}/xy/xz/yz (one or more)}
    {{stress target_stress}/
    {final lo hi (orthogonal) | final tilt (triclinic)}/
    {delta dlo dhi (orthogonal) | delta dtilt (triclinic)}/
    {scale factor (orthogonal)}/
    {vel V}/
    {erate R}/
    {cyclicmean mean amplitude period}/
    {cyclicdeviator mean amplitude period}/
    {constantb b boundary1 boundary2}}
    {linkvolstress {x y}/{y x}/{x z}/{z x}/{y z}/{z y}}
    {constantp {x y}/{y x}/{x z}/{z x}/{y z}/{z y}}
    {constantq {x y}/{y x}/{x z}/{z x}/{y z}/{z y}}
    {units lattice/box}
    
    If the final or delta options are used, your specifications
    may be modified. The lower boundaries do not move regardless of
    the user input. The strain rate will be correct though, e.g., if
    the cell has a height of 1 m and you specify delta as (-0.1 0.1),
    the cell will have final dimensions of 1.2 m as the upper boundary
    will move by 0.2 m.

    Lattice units are the default, as for fix_deform, but note that
    these are not fully tested (hence a warning is issued if lattice
    units are selected). 
    
    Variable (i.e., uncontrolled) volume is the default.

    Beware if strain control is required on a dimension and this fix is
    redefined at any stage. Take for example the 1 m high cell and set
    a strain rate of -0.2/s. Run for 1 s so that the cell has a height
    of 0.8 m. If fix_multistress is redefined with the same strain rate,
    the velocity of the boundary will change from 0.2 m/s to 0.16 m/s.
    This is NOT a bug - it is expected behaviour. To get the same
    velocity, you would need to increase the strain rate to -0.25/s.
    In this case, it is easiest to use "vel" as this is invariant across
    restarts (final, delta, scale and erate all need changing). You
    have been warned!

    [KH - 13 November 2012]*/

  /*~ Note that the order of the entries in any 6-element vectors is:
    x = 0; y = 1; z = 2; xy = 3; xz = 4; yz = 5*/

  restart_global = 1; //~ Global information is saved to the restart file
  nevery = 1; //~ Set how often to run the end_of_step function

  //~ Set flags to indicate that the dimensions of the box change
  no_change_box = 1;

  MPI_Comm_rank(world,&me); //~ Identify the processor rank

  //~ Create thermo output
  vector_flag = 1; //~ A compute_vector() function exists
  size_vector = 6; //~ Length of global vector
  global_freq = 1; //~ Frequency at which the data is available
  extvector = 1; //~ An intensive quantity is stored in the vector

  if (narg < 8 || narg > 36)
    error->all(FLERR,"Illegal fix multistress command");
  
  if (!atom->sphere_flag)
    error->all(FLERR,"Fix multistress requires atom style sphere");

  dimension = domain->dimension;
  triclinic = domain->triclinic;

  currstep = 0;
  instabcheck = 0; //~ Whether or not to run an optional instability check (1 == run)
  constructorflag = 1; //~ Indicates that the constructor has just been run

  for (int i = 0; i < 6; i++) {
    starget[i] = 0.0; //~ Target stresses input by user
    strflag[i] = defflag[i] = stabtestflag[i]
      = deltflag[i] = ictrlflag[i] = cyclicflag[i] = 0; //~ Initialise as flags
    erates[i] = 0.0; //~ Initialise the rates of change of the box boundaries
    maxrate[i] = -1.0; //~ Initialise the maxrates at negative values
    cyclicparam[0][i] = cyclicparam[1][i] = cyclicparam[2][i] = 0.0; //~ Cyclic loading parameters
  }

  /*~ Initialise linkvolstress - this exists if both the volume is maintained
    constant and if there is a link between the stresses in any two orthogonal
    directions. Also initialise constantpq - for maintaining either a constant
    p' or q and linking two other stresses together*/
  for (int i = 0; i < 3; i++) {
    linkvolstress[i] = 0; //~ Initialise as flags
    constantpq[i] = 0;
    constbflag[i] = 0;
  }

  lvstressflag = 0; //~ A flag to indicate whether linkvolstress is active
  constpflag = 0; //~ A flag to indicate whether constantp is active
  constqflag = 0; //~ A flag to indicate whether constantq is active
  constanteflag = 0; //~ A flag used for maintaining a constant e with fix crushing

  tolerance = force->numeric(FLERR,arg[3]); //~ Read in as a double
  if (tolerance <= 0 || tolerance >= 100) error->all(FLERR,"The tolerance must be a positive percentage");

  extrasteps = force->inumeric(FLERR,arg[4]);
  if (extrasteps <= 0) error->all(FLERR,"The number of extra steps must be positive");

  //~ Determine which sets of units are used - box or lattice (default)
  int iarg = 5;
  scaleflag = 1;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"units") == 0 && strcmp(arg[iarg+1],"box") == 0) {
      scaleflag = 0;
      break;
    }
    iarg++;
  }

  /*~ Issue a warning if lattice units are selected (not really tested and
    unlikely to work in the expected manner as a result)*/
  if (scaleflag == 1) error->warning(FLERR,"Lattice units have not been properly tested");

  //~ The maximum strain rate argument is optional (but recommended)
  maxrateflag = 0;
  iarg = 5;

  if (strncmp(arg[iarg],"x",1) != 0 && strncmp(arg[iarg],"y",1) != 0 &&
      strncmp(arg[iarg],"z",1) != 0) {
    for (int i = 0; i < 6; i++)
      maxrate[i] = fabs(force->numeric(FLERR,arg[iarg])); //~ Use the absolute value

    maxrateflag = 1;
    iarg++;
  }

  /*~ Initialise the number of cyclic loading steps, which may be replaced by the
    restart file value, if present, along with the initial volume bounded by the
    periodic boundaries and the initial mean effective and deviator stresses*/
  ncyclicsteps = 0;
  initialvolume = meaneffectivestress = deviatorstress = 0.0;

  /*~ Apply scaling if final, delta or vel are specified for strain control
    and box units are not specified explicitly.*/
  if (scaleflag && domain->lattice == NULL)
    error->all(FLERR,"Use of fix deform with undefined lattice");

  double xscale,yscale,zscale;
  if (scaleflag) {
    xscale = domain->lattice->xlattice;
    yscale = domain->lattice->ylattice;
    zscale = domain->lattice->zlattice;
  } else xscale = yscale = zscale = 1.0;

  double map[6]; //~ Mappings for indices 3 and 5 swapped compared to fix_deform
  map[0] = xscale; map[1] = yscale; map[2] = zscale;
  map[3] = xscale; map[4] = xscale; map[5] = yscale;

  /*~ addextra is an integer which is used to increase iarg by 1 due to 
    the additional argument for both final and delta with orthogonal boxes,
    and increase it by 2 if cyclicstress or constantb is used on a boundary*/
  int addextra = 0;

  /*~ One issue is that suitable values for beginstep and endstep are
    not set until the "Run::command" function is executed, which occurs
    after this constructor is run. Thus the variable "delt" cannot be
    set within this function. Temporarily set this equal to 1 and track
    which variables make use of delt. Then divide by the real delt value
    in the init function, when the actual values for beginstep and endstep
    will be available.*/
  int delt = 1;

  while (iarg < narg) {
    addextra = 0; //~ Reinitialise to 0

    if (strcmp(arg[iarg],"x") == 0 || strcmp(arg[iarg],"xx") == 0) {

      double xlostart = domain->boxlo[0];
      double xhistart = domain->boxhi[0];

      if (strcmp(arg[iarg+1],"stress") == 0) {
	starget[0] = force->numeric(FLERR,arg[iarg+2]);
	strflag[0] += 1; //~ Incremented to allow checking for multiple specifications
      } else if (strcmp(arg[iarg+1],"cyclicmean") == 0 || strcmp(arg[iarg+1],"cyclicdeviator") == 0) {
	for (int i = 0; i < 3; i++)
	  cyclicparam[i][0] = force->numeric(FLERR,arg[iarg+i+2]);

	addextra = 2;
	strflag[0] += 1;

	if (strcmp(arg[iarg+1],"cyclicmean") == 0) cyclicflag[0] = 1;
	else cyclicflag[0] = 2;
      } else if (strcmp(arg[iarg+1],"constantb") == 0) {
	cyclicparam[0][0] = force->numeric(FLERR,arg[iarg+2]); //~ Store the b value here
	addextra = 2;
	strflag[0] += 1;

	if (strcmp(arg[iarg+3],"y") == 0 && strcmp(arg[iarg+4],"z") == 0) constbflag[0] = 1;
	else if (strcmp(arg[iarg+3],"z") == 0 && strcmp(arg[iarg+4],"y") == 0) constbflag[0] = 2;
	else error->all(FLERR,"Incorrect constantb specification in fix multistress");
      } else {
	if (strcmp(arg[iarg+1],"erate") == 0) {
	  erates[0] = force->numeric(FLERR,arg[iarg+2]);
	} else if (strcmp(arg[iarg+1],"vel") == 0) {
	  erates[0] = map[0]*force->numeric(FLERR,arg[iarg+2])/(xhistart-xlostart);
	} else if (strcmp(arg[iarg+1],"scale") == 0) {
	  if (domain->triclinic == 1)
	    error->all(FLERR,"Scale is valid in fix multistress only for orthogonal boxes");
	  else {
	    erates[0] = (force->numeric(FLERR,arg[iarg+2])-1)/delt;
	    deltflag[0] = 1;
	  }
	} else if (strcmp(arg[iarg+1],"delta") == 0) {
	    erates[0] = map[0]*(force->numeric(FLERR,arg[iarg+3])-force->numeric(FLERR,arg[iarg+2]))/(delt*(xhistart-xlostart));
	    deltflag[0] = 1;
	    addextra = 1;
	} else if (strcmp(arg[iarg+1],"final") == 0) {
	  if (force->numeric(FLERR,arg[iarg+2]) >= force->numeric(FLERR,arg[iarg+3]))
	    error->all(FLERR,"Correct the order of the fix multistress arguments");

	  erates[0] = (xlostart-xhistart+map[0]*(force->numeric(FLERR,arg[iarg+3])-force->numeric(FLERR,arg[iarg+2])))/(delt*(xhistart-xlostart));
	  deltflag[0] = 1;
	  addextra = 1;
	} else error->all(FLERR,"Illegal fix multistress command");
	
	defflag[0] += 1;
      }
    } else if (strcmp(arg[iarg],"y") == 0 || strcmp(arg[iarg],"yy") == 0) {
      
      double ylostart = domain->boxlo[1];
      double yhistart = domain->boxhi[1];

      if (strcmp(arg[iarg+1],"stress") == 0) {
	starget[1] = force->numeric(FLERR,arg[iarg+2]);
	strflag[1] += 1; //~ Incremented to allow checking for multiple specifications
      } else if (strcmp(arg[iarg+1],"cyclicmean") == 0 || strcmp(arg[iarg+1],"cyclicdeviator") == 0) {
	for (int i = 0; i < 3; i++)
	  cyclicparam[i][1] = force->numeric(FLERR,arg[iarg+i+2]);

	addextra = 2;
	strflag[1] += 1;

	if (strcmp(arg[iarg+1],"cyclicmean") == 0) cyclicflag[1] = 1;
	else cyclicflag[1] = 2;
      } else if (strcmp(arg[iarg+1],"constantb") == 0) {
	cyclicparam[0][1] = force->numeric(FLERR,arg[iarg+2]); //~ Store the b value here
	addextra = 2;
	strflag[1] += 1;

	if (strcmp(arg[iarg+3],"x") == 0 && strcmp(arg[iarg+4],"z") == 0) constbflag[1] = 1;
	else if (strcmp(arg[iarg+3],"z") == 0 && strcmp(arg[iarg+4],"x") == 0) constbflag[1] = 2;
	else error->all(FLERR,"Incorrect constantb specification in fix multistress");
      } else {
	if (strcmp(arg[iarg+1],"erate") == 0) {
	  erates[1] = force->numeric(FLERR,arg[iarg+2]);
	} else if (strcmp(arg[iarg+1],"vel") == 0) {
	  erates[1] = map[1]*force->numeric(FLERR,arg[iarg+2])/(yhistart-ylostart);
	} else if (strcmp(arg[iarg+1],"scale") == 0) {
	  if (domain->triclinic == 1)
	    error->all(FLERR,"Scale is valid in fix multistress only for orthogonal boxes");
	  else {
	    erates[1] = (force->numeric(FLERR,arg[iarg+2])-1)/delt;
	    deltflag[1] = 1;
	  }
	} else if (strcmp(arg[iarg+1],"delta") == 0) {
	  erates[1] = map[1]*(force->numeric(FLERR,arg[iarg+3])-force->numeric(FLERR,arg[iarg+2]))/(delt*(yhistart-ylostart));
	  deltflag[1] = 1;
	  addextra = 1;
	} else if (strcmp(arg[iarg+1],"final") == 0) {
	  if (force->numeric(FLERR,arg[iarg+2]) >= force->numeric(FLERR,arg[iarg+3]))
	    error->all(FLERR,"Correct the order of the fix multistress arguments");
	  
	  erates[1] = (ylostart-yhistart+map[1]*(force->numeric(FLERR,arg[iarg+3])-force->numeric(FLERR,arg[iarg+2])))/(delt*(yhistart-ylostart));
	  deltflag[1] = 1;
	  addextra = 1;
	} else error->all(FLERR,"Illegal fix multistress command");
	
	defflag[1] += 1;
      }
    } else if (strcmp(arg[iarg],"z") == 0 || strcmp(arg[iarg],"zz") == 0) {
      
      double zlostart = domain->boxlo[2];
      double zhistart = domain->boxhi[2];

      if (strcmp(arg[iarg+1],"stress") == 0) {
	starget[2] = force->numeric(FLERR,arg[iarg+2]);
	strflag[2] += 1; //~ Incremented to allow checking for multiple specifications
      } else if (strcmp(arg[iarg+1],"cyclicmean") == 0 || strcmp(arg[iarg+1],"cyclicdeviator") == 0) {
	for (int i = 0; i < 3; i++)
	  cyclicparam[i][2] = force->numeric(FLERR,arg[iarg+i+2]);

	addextra = 2;
	strflag[2] += 1;

	if (strcmp(arg[iarg+1],"cyclicmean") == 0) cyclicflag[2] = 1;
	else cyclicflag[2] = 2;
      } else if (strcmp(arg[iarg+1],"constantb") == 0) {
	cyclicparam[0][2] = force->numeric(FLERR,arg[iarg+2]); //~ Store the b value here
	addextra = 2;
	strflag[2] += 1;

	if (strcmp(arg[iarg+3],"x") == 0 && strcmp(arg[iarg+4],"y") == 0) constbflag[2] = 1;
	else if (strcmp(arg[iarg+3],"y") == 0 && strcmp(arg[iarg+4],"x") == 0) constbflag[2] = 2;
	else error->all(FLERR,"Incorrect constantb specification in fix multistress");
      } else {
	if (strcmp(arg[iarg+1],"erate") == 0) {
	  erates[2] = force->numeric(FLERR,arg[iarg+2]);
	} else if (strcmp(arg[iarg+1],"vel") == 0) {
	  erates[2] = map[2]*force->numeric(FLERR,arg[iarg+2])/(zhistart-zlostart);
	} else if (strcmp(arg[iarg+1],"scale") == 0) {
	  if (domain->triclinic == 1)
	    error->all(FLERR,"Scale is valid in fix multistress only for orthogonal boxes");
	  else {
	    erates[2] = (force->numeric(FLERR,arg[iarg+2])-1)/delt;
	    deltflag[2] = 1;
	  }
	} else if (strcmp(arg[iarg+1],"delta") == 0) {
	  erates[2] = map[2]*(force->numeric(FLERR,arg[iarg+3])-force->numeric(FLERR,arg[iarg+2]))/(delt*(zhistart-zlostart));
	  deltflag[2] = 1;
	  addextra = 1;
	} else if (strcmp(arg[iarg+1],"final") == 0) {
	  if (force->numeric(FLERR,arg[iarg+2]) >= force->numeric(FLERR,arg[iarg+3]))
	    error->all(FLERR,"Correct the order of the fix multistress arguments");

	  erates[2] = (zlostart-zhistart+map[2]*(force->numeric(FLERR,arg[iarg+3])-force->numeric(FLERR,arg[iarg+2])))/(delt*(zhistart-zlostart));
	  deltflag[2] = 1;
	  addextra = 1;
	} else error->all(FLERR,"Illegal fix multistress command");
	
	defflag[2] += 1;
      }
    } else if (strcmp(arg[iarg],"xy") == 0 || strcmp(arg[iarg],"yx") == 0) {
      
      double xytiltstart = domain->xy;
      double ylostart = domain->boxlo[1];
      double yhistart = domain->boxhi[1];

      if (strcmp(arg[iarg+1],"stress") == 0) {
	starget[3] = force->numeric(FLERR,arg[iarg+2]);
	strflag[3] += 1;
      } else if (strcmp(arg[iarg+1],"cyclicmean") == 0) {
	for (int i = 0; i < 3; i++)
	  cyclicparam[i][3] = force->numeric(FLERR,arg[iarg+i+2]);

	addextra = 2;
	cyclicflag[3] = 1;
	strflag[3] += 1;
      } else {
	if (strcmp(arg[iarg+1],"erate") == 0) {
	  erates[3] = force->numeric(FLERR,arg[iarg+2]);
	} else if (strcmp(arg[iarg+1],"vel") == 0) {
	  erates[3] = map[3]*force->numeric(FLERR,arg[iarg+2])/(yhistart-ylostart);
	} else if (strcmp(arg[iarg+1],"scale") == 0) {
	  error->all(FLERR,"Scale is valid in fix multistress only for orthogonal boxes");
	} else if (strcmp(arg[iarg+1],"delta") == 0) {
	  erates[3] = map[3]*force->numeric(FLERR,arg[iarg+2])/(delt*(yhistart-ylostart));
	  deltflag[3] = 1;
	} else if (strcmp(arg[iarg+1],"final") == 0) {
	  erates[3] = (map[3]*force->numeric(FLERR,arg[iarg+2])-xytiltstart)/(delt*(yhistart-ylostart));
	  deltflag[3] = 1;
	} else error->all(FLERR,"Illegal fix multistress command");
	
	defflag[3] += 1;
      }
    } else if (strcmp(arg[iarg],"xz") == 0 || strcmp(arg[iarg],"zx") == 0) {
      
      double xztiltstart = domain->xz;
      double zlostart = domain->boxlo[2];
      double zhistart = domain->boxhi[2];

      if (strcmp(arg[iarg+1],"stress") == 0) {
	starget[4] = force->numeric(FLERR,arg[iarg+2]);
	strflag[4] += 1;
      } else if (strcmp(arg[iarg+1],"cyclicmean") == 0) {
	for (int i = 0; i < 3; i++)
	  cyclicparam[i][4] = force->numeric(FLERR,arg[iarg+i+2]);

	addextra = 2;
	cyclicflag[4] = 1;
	strflag[4] += 1;
      } else {
	if (strcmp(arg[iarg+1],"erate") == 0) {
	  erates[4] = force->numeric(FLERR,arg[iarg+2]);
	} else if (strcmp(arg[iarg+1],"vel") == 0) {
	  erates[4] = map[4]*force->numeric(FLERR,arg[iarg+2])/(zhistart-zlostart);
	} else if (strcmp(arg[iarg+1],"scale") == 0) {
	  error->all(FLERR,"Scale is valid in fix multistress only for orthogonal boxes");
	} else if (strcmp(arg[iarg+1],"delta") == 0) {
	  erates[4] = map[4]*force->numeric(FLERR,arg[iarg+2])/(delt*(zhistart-zlostart));
	  deltflag[4] = 1;
	} else if (strcmp(arg[iarg+1],"final") == 0) {
	  erates[4] = (map[4]*force->numeric(FLERR,arg[iarg+2])-xztiltstart)/(delt*(zhistart-zlostart));
	  deltflag[4] = 1;
	} else error->all(FLERR,"Illegal fix multistress command");
	
	defflag[4] += 1;
      }
    } else if (strcmp(arg[iarg],"yz") == 0 || strcmp(arg[iarg],"yz") == 0) {
      
      double yztiltstart = domain->yz;
      double zlostart = domain->boxlo[2];
      double zhistart = domain->boxhi[2];

      if (strcmp(arg[iarg+1],"stress") == 0) {
	starget[5] = force->numeric(FLERR,arg[iarg+2]);
	strflag[5] += 1;
      } else if (strcmp(arg[iarg+1],"cyclicmean") == 0) {
	for (int i = 0; i < 3; i++)
	  cyclicparam[i][5] = force->numeric(FLERR,arg[iarg+i+2]);

	addextra = 2;
	cyclicflag[5] = 1;
	strflag[5] += 1;
      } else {
	if (strcmp(arg[iarg+1],"erate") == 0) {
	  erates[5] = force->numeric(FLERR,arg[iarg+2]);
	} else if (strcmp(arg[iarg+1],"vel") == 0) {
	  erates[5] = map[5]*force->numeric(FLERR,arg[iarg+2])/(zhistart-zlostart);
	} else if (strcmp(arg[iarg+1],"scale") == 0) {
	  error->all(FLERR,"Scale is valid in fix multistress only for orthogonal boxes");
	} else if (strcmp(arg[iarg+1],"delta") == 0) {
	  erates[5] = map[5]*force->numeric(FLERR,arg[iarg+2])/(delt*(zhistart-zlostart));
	  deltflag[5] = 1;
	} else if (strcmp(arg[iarg+1],"final") == 0) {
	  erates[5] = (map[5]*force->numeric(FLERR,arg[iarg+2])-yztiltstart)/(delt*(zhistart-zlostart));
	  deltflag[5] = 1;
	} else error->all(FLERR,"Illegal fix multistress command");
	
	defflag[5] += 1;
      }
    } else if (strcmp(arg[iarg],"units") == 0) {
      iarg--; //~ Decrement since units has only one word following it
    } else if (strcmp(arg[iarg],"linkvolstress") == 0) {
      if (strcmp(arg[iarg+1],"x") == 0 || strcmp(arg[iarg+2],"x") == 0) linkvolstress[0] = 1;
      if (strcmp(arg[iarg+1],"y") == 0 || strcmp(arg[iarg+2],"y") == 0) linkvolstress[1] = 1;
      if (strcmp(arg[iarg+1],"z") == 0 || strcmp(arg[iarg+2],"z") == 0) linkvolstress[2] = 1;
      lvstressflag = 1;
      if ((linkvolstress[0] + linkvolstress[1] + linkvolstress[2]) != 2)
	error->all(FLERR,"Illegal fix multistress command: linkvolstress is defined incorrectly");
    } else if (strcmp(arg[iarg],"constantp") == 0) {
      if (strcmp(arg[iarg+1],"x") == 0 || strcmp(arg[iarg+2],"x") == 0) constantpq[0] = 1;
      if (strcmp(arg[iarg+1],"y") == 0 || strcmp(arg[iarg+2],"y") == 0) constantpq[1] = 1;
      if (strcmp(arg[iarg+1],"z") == 0 || strcmp(arg[iarg+2],"z") == 0) constantpq[2] = 1;
      constpflag = 1;
      if ((constantpq[0] + constantpq[1] + constantpq[2]) != 2)
	error->all(FLERR,"Illegal fix multistress command: constantp is defined incorrectly");
    } else if (strcmp(arg[iarg],"constantq") == 0) {
      if (strcmp(arg[iarg+1],"x") == 0 || strcmp(arg[iarg+2],"x") == 0) constantpq[0] = 1;
      if (strcmp(arg[iarg+1],"y") == 0 || strcmp(arg[iarg+2],"y") == 0) constantpq[1] = 1;
      if (strcmp(arg[iarg+1],"z") == 0 || strcmp(arg[iarg+2],"z") == 0) constantpq[2] = 1;
      constqflag = 1;
      if ((constantpq[0] + constantpq[1] + constantpq[2]) != 2)
	error->all(FLERR,"Illegal fix multistress command: constantq is defined incorrectly");
    } else error->all(FLERR,"Illegal fix multistress command");

    iarg += (3 + addextra);
  }

  //~ Set cstressflag to indicate cyclic manipulation of the stresses
  if (cyclicflag[0] > 0 || cyclicflag[1] > 0 || cyclicflag[2] > 0 ||
      cyclicflag[3] > 0 || cyclicflag[4] > 0 || cyclicflag[5] > 0) cstressflag = 1;
  else cstressflag = 0;

  //~ Set constbctrl to indicate that the b value is being controlled
  if ((constbflag[0] != 0 && (constbflag[1] + constbflag[2] > 0)) ||
      (constbflag[1] != 0 && (constbflag[0] + constbflag[2] > 0)) ||
      (constbflag[2] != 0 && (constbflag[0] + constbflag[1] > 0))) 
    error->all(FLERR,"Cannot use constantb on more than one boundary simultaneously");
  else if (constbflag[0] + constbflag[1] + constbflag[2] > 0) constbctrl = 1;
  else constbctrl = 0;

  for (int i = 0; i < 6; i++) {
    if ((strflag[i] + defflag[i]) > 1)
      error->all(FLERR,"Cannot repeat dimension specifiers in fix multistress");
      
    //~ Test for negative target stresses
    //~ Note the convention used here is compressive stresses are positive
    if (starget[i] < 0.0)
      error->all(FLERR,"Cannot specify negative target stresses for fix multistress");

    //~ Confirm that the cyclic loading amplitude does not cause negative stresses
    if (cyclicflag[i] == 1 && (cyclicparam[0][i]-cyclicparam[1][i]) < 0.0)
      error->all(FLERR,"Cannot cause negative mean stresses during cycling with fix multistress");

    if (cyclicflag[i] == 2 && lvstressflag == 0)
      error->warning(FLERR,"Using the cyclicdeviator option without linkvolstress is not generally a good idea");
  }
  
  //~ Check values are sensible if constantb is active
  if (constbctrl == 1 && ((constbflag[0] > 0 && (cyclicparam[0][0] < 0.0 || cyclicparam[0][0] > 1.0)) || 
			  (constbflag[1] > 0 && (cyclicparam[0][1] < 0.0 || cyclicparam[0][1] > 1.0)) || 
			  (constbflag[2] > 0 && (cyclicparam[0][2] < 0.0 || cyclicparam[0][2] > 1.0))))
    error->all(FLERR,"b value must be <= 1.0 and >= 0.0 in fix multistress");

  //~ Warn the user if linkvolstress, constantp or constantq are active without maxrateflag
  if (maxrateflag == 0 && (lvstressflag == 1 || constpflag == 1 || constqflag == 1))
    error->warning(FLERR,"It is strongly recommended that a maxrate restriction is set when linkvolstress, constantp or constantq are active");

  //~ Set flags to indicate whether or not the box changes size/shape
  for (int i = 0; i < 3; i++)
    if (strflag[i] == 1 || defflag[i] == 1 || linkvolstress[i] == 1 || constantpq[i] == 1)
      box_change_size = 1;

  for (int i = 3; i < 6; i++)
    if (strflag[i] == 1 || defflag[i] == 1)
      box_change_shape = 1;

  if (dimension == 2 && (strflag[2] != 0 || strflag[4] != 0 || strflag[5] != 0 ||
			 defflag[2] != 0 || defflag[4] != 0 || defflag[5] != 0))
    error->all(FLERR,"No z specifier in fix multistress for 2D simulations");
       
  if (triclinic == 0 && (strflag[3] != 0 || strflag[4] != 0 || strflag[5] != 0 ||
				 defflag[3] != 0 || defflag[4] != 0 || defflag[5] != 0))
    error->all(FLERR,"Shear stress specifications require triclinic boxes");

  if (((strflag[0] == 1 || defflag[0] == 1) && domain->xperiodic == 0) ||
      ((strflag[1] == 1 || defflag[1] == 1) && domain->yperiodic == 0) ||
      ((strflag[2] == 1 || defflag[2] == 1) && domain->zperiodic == 0))
    error->all(FLERR,"Cannot use fix multistress on a non-periodic boundary");

  if (((strflag[0] == 1 || defflag[0] == 1) && (linkvolstress[0] == 1 || constantpq[0] == 1)) ||
      ((strflag[1] == 1 || defflag[1] == 1) && (linkvolstress[1] == 1 || constantpq[1] == 1)) ||
      ((strflag[2] == 1 || defflag[2] == 1) && (linkvolstress[2] == 1 || constantpq[2] == 1)))
    error->all(FLERR,"Cannot use linkvolstress, constantp or constantq and also control the periodic boundary positions by other methods");

  if (dimension == 2 && (lvstressflag == 1 || constpflag == 1 || constqflag == 1 || constbctrl == 1))
    error->all(FLERR,"Cannot use linkvolstress, constantp, constantq or constantb for 2D simulations");

  if (lvstressflag + constpflag + constqflag + constbctrl > 1)
    error->all(FLERR,"Cannot use more than one of linkvolstress, constantp, constantq and constantb at the same time");    
    
  if (cyclicflag[0] == 2 && (cyclicflag[1] == 2 || cyclicflag[2] == 2) ||
      cyclicflag[1] == 2 && (cyclicflag[0] == 2 || cyclicflag[2] == 2))
    error->all(FLERR,"Cannot simultaneously use the cyclicdeviator option on multiple boundaries");

  //~ Create a new compute stress/atom style
  int n = strlen(id) + strlen("_stress") + 1;
  id_stress = new char[n];
  strcpy(id_stress,id);
  strcat(id_stress,"_stress");
  
  char **snewarg = new char*[7];
  snewarg[0] = id_stress;
  snewarg[1] = (char *) "all";
  snewarg[2] = (char *) "stress/atom";
  snewarg[3] = (char *) "NULL"; //~ No temperature [KH - 14 Feb. 2014]
  snewarg[4] = (char *) "pair";
  snewarg[5] = (char *) "fix";
  snewarg[6] = (char *) "bond";

  modify->add_compute(7,snewarg);
  delete [] snewarg;
}

/* ---------------------------------------------------------------------- */

FixMultistress::~FixMultistress()
{
  //~ Save the number of cyclic loading steps in domain
  if (cstressflag == 1) domain->ncyclicsteps = ncyclicsteps;

  //~ Delete the stress compute only if fix_crushing is not present
  int fixcrush = -1;
  for (int q = 0; q < modify->nfix; q++)
    if (strcmp(modify->fix[q]->style,"crushing") == 0) fixcrush = q;

  if (fixcrush < 0) {
    modify->delete_compute(id_stress);
    delete [] id_stress;
  }

  modify->delete_fix(id_multistress);
  delete [] id_multistress;
}

/* ---------------------------------------------------------------------- */

int FixMultistress::setmask()
{
  int mask = 0;
  if (domain->triclinic) mask |= PRE_EXCHANGE;
  mask |= PRE_FORCE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMultistress::init()
{ 
  //~ Ensure no conflict with another active fix deform
  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"deform") == 0) {
      if (strcmp(modify->fix[i]->id,id_multistress) == 0)
	modify->delete_fix(id_multistress);
      else error->all(FLERR,"Cannot use fix multistress and fix deform together");
    }

  //~ Check that all particles have a non-zero radius
  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double minrad = BIG; //~ minrad contains the radius of the smallest atom

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (radius[i] < minrad)
	minrad = radius[i];

  if (minrad == 0.0)
    error->all(FLERR,"Fix multistress requires extended particles");

  //~ Update the affected rates if the calculations included the delt variable
  double delt = (update->endstep - update->beginstep) * update->dt;
  
  for (int i = 0; i < 6; i++)
    if (deltflag[i] == 1) {
      erates[i] /= delt;
      deltflag[i] = 0;
    }

  //~ Set pointers for the stress compute
  int icompute = modify->find_compute(id_stress);
  if (icompute < 0) error->all(FLERR,"Stress ID for fix multistress does not exist");
  tstress = modify->compute[icompute];

  if (constructorflag == 1)
    for (int i = 0; i < 6; i++)
      cumul[i] = 0.0; //~ Initialise the cumulative error for the integral control term

  //~ Initialise the required elements of the instability array
  if (instabcheck == 1 && constructorflag == 1)
    for (int i = 0; i < 6; i++) {
      instability[i][0] = starget[i];
      instability[i][2] = instability[i][0];
    }

  /*~ Set constanteflag to 1 if fix crushing is active and the option to
    maintain a constant void ratio is enabled*/
  for (int q = 0; q < modify->nfix; q++)
    if (strcmp(modify->fix[q]->style,"crushing") == 0)
      constanteflag = ((FixCrushing *) modify->fix[q])->constante;

  /*~ Calculate the volume enclosed by the periodic boundries initially. This is
    done only once even if fix multistress is redefined while linkvolstress
    continues to be active*/
  if (lvstressflag) {
    if (initialvolume == 0.0) initialvolume = domain->initialvolume;
    if (initialvolume == 0.0) //~ If it is still equal to zero...
      initialvolume = (domain->boxhi[0]-domain->boxlo[0])*(domain->boxhi[1]-domain->boxlo[1])*(domain->boxhi[2]-domain->boxlo[2]);
  } else { //~ Reset initialvolume to 0 if linkvolstress is disabled
    initialvolume = 0.0;
  }

  //~ Always keep correspondence between these values
  domain->initialvolume = initialvolume;

  //~ Restore the number of steps of cyclic loading
  if (cstressflag == 1 && ncyclicsteps == 0)
    ncyclicsteps = domain->ncyclicsteps;

  //~ Reset ncyclicsteps to 0 if the cyclic loading is disabled
  if (cstressflag == 0) {
    ncyclicsteps = 0;
    domain->ncyclicsteps = 0;
  }

  //~ Store the initial positions of the periodic cell boundaries
  if (constructorflag == 1) {
    for (int i = 0; i < 2; i++) {
      pboundstart[2*i] = domain->boxlo[i];
      pboundstart[2*i+1] = domain->boxhi[i];
    }

    if (dimension == 3) {
      pboundstart[4] = domain->boxlo[2];
      pboundstart[5] = domain->boxhi[2];
    } else pboundstart[4] = pboundstart[5] = 0.0;
  }

  //~ Set up the fix_deform that is required initially
  int n = strlen(id) + strlen("_multistress") + 1;
  id_multistress = new char[n];
  strcpy(id_multistress,id);
  strcat(id_multistress,"_multistress");

  //~ numchar is the number of strings to store in tnewarg (in create_fix)
  numchar = 3*(dimension+2+triclinic*(2*(dimension-triclinic)-1))+2+dimension;

  constructorflag = 0; //~ Reset the constructorflag to zero
  create_fix(); //~ Otherwise it will be created by the end_of_step function
}

/* ---------------------------------------------------------------------- */

void FixMultistress::pre_exchange()
{
  if (flip == 1)
    deffix->pre_exchange(); //~ TEST THIS - when does it run and does it work correctly.
}

/* ---------------------------------------------------------------------- */

void FixMultistress::pre_force(int vflag)
{
  if (update->integrate->vflag <= 2) update->integrate->vflag += 4;
}

/* ---------------------------------------------------------------------- */

void FixMultistress::create_fix()
{
  //~ Convert erates * to delta * * format so that the lower boundary can remain stationary
  double dhi[3]; //~ To store the differences in position of the upper boundary

  for (int i = 0; i < 3; i++)
    dhi[i] = erates[i]*(pboundstart[2*i+1]-pboundstart[2*i])*update->dt;

  //~ Carry out the necessary string conversions
  char sxxdhi[20] = {0};
  char syydhi[20] = {0};
  char szzdhi[20] = {0};
  char sxyerates[20] = {0};
  char sxzerates[20] = {0};
  char syzerates[20] = {0};

  sprintf(sxxdhi,"%1.16f",dhi[0]);
  sprintf(syydhi,"%1.16f",dhi[1]);
  sprintf(szzdhi,"%1.16f",dhi[2]);
  sprintf(sxyerates,"%1.16f",erates[3]);
  sprintf(sxzerates,"%1.16f",erates[4]);
  sprintf(syzerates,"%1.16f",erates[5]);

  char **tnewarg = new char*[numchar];
  tnewarg[0] = id_multistress;
  tnewarg[1] = (char *) "all";
  tnewarg[2] = (char *) "deform";
  tnewarg[3] = (char *) "1"; //~ 1 here is not the number of timesteps over which to do the deformation
  tnewarg[4] = (char *) "x";
  tnewarg[5] = (char *) "delta";
  tnewarg[6] = (char *) "0.0"; //~ The lower boundary is not moved
  tnewarg[7] = sxxdhi;
  tnewarg[8] = (char *) "y";
  tnewarg[9] = (char *) "delta";
  tnewarg[10] = (char *) "0.0";
  tnewarg[11] = syydhi;
  int iarg = 12;

  if (dimension == 3) {
    tnewarg[12] = (char *) "z";
    tnewarg[13] = (char *) "delta";
    tnewarg[14] = (char *) "0.0";
    tnewarg[15] = szzdhi;
    iarg += 4;
  }

  if (triclinic == 1) {
    tnewarg[iarg] = (char *) "xy";
    tnewarg[iarg+1] = (char *) "erate";
    tnewarg[iarg+2] = sxyerates;

    if (dimension == 3) {
      tnewarg[19] = (char *) "xz";
      tnewarg[20] = (char *) "erate";
      tnewarg[21] = sxzerates;
      tnewarg[22] = (char *) "yz";
      tnewarg[23] = (char *) "erate";
      tnewarg[24] = syzerates;
    }
  }

  tnewarg[numchar-4] = (char *) "units";
  scaleflag == 0 ? tnewarg[numchar-3] = (char *) "box" : tnewarg[numchar-3] = (char *) "lattice";
  tnewarg[numchar-2] = (char *) "remap";
  tnewarg[numchar-1] = (char *) "x";

  modify->add_fix(numchar,tnewarg);
  delete [] tnewarg;

  //~ Set pointers for the deform fix
  int accfix = modify->find_fix(id_multistress);
  if (accfix < 0) error->all(FLERR,"Fix ID for fix multistress does not exist");
  deffix = modify->fix[accfix];
}

/* ---------------------------------------------------------------------- */

void FixMultistress::calc_ctrl_params()
{
  // double Ku,Pu; //~ The ultimate gain and period of the controller
  // Pu = 1000*update->dt;

  for (int i = 0; i < 6; i++) {
    //~ The Ziegler-Nichols settings were used for the PID controller
    // ti[i] = Pu/2; //~ These times become irrelevant if the gain is 0
    // td[i] = Pu/8;

    if (strflag[i] == 1 || (i < 3 && constantpq[i] == 1)) {
      if (maxrateflag == 1) Kp[i] = 2.0*maxrate[i]/starget[i];
      else Kp[i] = 50.0/starget[i];
    } else Kp[i] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void FixMultistress::end_of_step()
{
  bigint remsteps = update->endstep - update->ntimestep;

  //~ Initialise the % differences between the current and target stresses
  double maxpercdiff = 0.0;
  double percdiff[6] = {0.0};

  //~ Assume below that the root processor is representative of all
  if (remsteps == 0 && me == 0 && extrasteps > 0 && (strflag[0] || strflag[1] ||
					  strflag[2] || strflag[3] ||
					  strflag[4] || strflag[5]))
    error->warning(FLERR,"The specified stress tolerance was not attained");

  //~ Necessary to include currstep != 0 as means not defined otherwise
  //~ Also omit boundaries which are loaded cyclically
  for (int i = 0; i < 6; i++)
    if (strflag[i] == 1 && cyclicflag[i] == 0 && currstep != 0) {
      percdiff[i] = 100*fabs(starget[i]- tallymeans[i])/starget[i];
      if (percdiff[i] > maxpercdiff) maxpercdiff = percdiff[i];
    }

  if (maxpercdiff <= tolerance) extrasteps--;

  //~ Run an optional check for lost atoms every 100000 timesteps
  if (currstep % 100000 == 0) lost_atom_check();

  //~ Store the old mean stresses
  if (currstep != 0)
    for (int i = 0; i < 6; i++)
      oldmeans[i] = tallymeans[i];

  if (tstress->invoked_peratom != update->ntimestep)
    tstress->compute_peratom(); //~ Update the atom stresses if necessary

  //~ Import a 6-element array of mean stresses from ComputeStressAtom
  means = tstress->array_export();

  //~ Accumulate the means from all processors in tallymeans and initialise temprates
  for (int i = 0; i < 6; i++) tallymeans[i] = temprates[i] = 0.0;
  MPI_Allreduce(&means[0],&tallymeans[0],6,MPI_DOUBLE,MPI_SUM,world);

  /*~ Calculate the initial values of mean effective and deviator stress. 
    This is done only once even if fix multistress is redefined while
    constantp or constantq continues to be active*/
  if (currstep == 0) {
    //~ Firstly the mean effective stress
    if (constpflag == 1 && meaneffectivestress == 0.0)
      meaneffectivestress = domain->meaneffectivestress;

    //~ If it is still equal to zero...
    if (constpflag == 1 && meaneffectivestress == 0.0)
      meaneffectivestress = (tallymeans[0] + tallymeans[1] + tallymeans[2])/3.0;

    //~ Reset meaneffectivestress to 0 if constantp is disabled
    if (constpflag == 0) {
      meaneffectivestress = 0.0;
      domain->meaneffectivestress = 0.0;
    } else domain->meaneffectivestress = meaneffectivestress;

    //~ Now the deviator stress
    if (constqflag == 1 && deviatorstress == 0.0)
      deviatorstress = domain->deviatorstress;

    //~ If it is still equal to zero...
    if (constqflag == 1 && deviatorstress == 0.0)
      for (int i = 0; i < 3; i++) {
	if (constantpq[i] == 0) deviatorstress += tallymeans[i];
	else deviatorstress -= 0.5*tallymeans[i];
      }

    //~ Reset deviatorstress to 0 if constantq is disabled
    if (constqflag == 0) {
      deviatorstress = 0.0;
      domain->deviatorstress = 0.0;
    } else domain->deviatorstress = deviatorstress;

    //~ Need to specify initial values for starget on the first timestep if constbctrl active
    if (constbctrl) {
      for (int i = 0; i < 6; i++)
	if (constbflag[i] > 0) starget[i] = tallymeans[i];
    }
  }

  calc_ctrl_params(); //~ Calculate the controller parameters required, which may be a function of maxrate (if so desired)

  //~ Update the target stresses if the stresses on the boundary are varied cyclically
  if (cstressflag) {
    ncyclicsteps++; //~ Increment the number of cyclic loading steps

    for (int i = 0; i < 6; i++) {
      if (cyclicflag[i] == 1) starget[i] = cyclicparam[0][i] + cyclicparam[1][i]*sin(8.0*atan(1.0)*update->dt*ncyclicsteps/cyclicparam[2][i]);
      else if (cyclicflag[i] == 2) {
	int firstboundaryid = (i+1)%3;
	int secondboundaryid = (i+2)%3;
	double minorpstress = 0.0;

	if (lvstressflag == 0 || (update->ntimestep - update->beginstep) == 1 || ncyclicsteps == 1)
	  minorpstress = 0.5*(tallymeans[firstboundaryid] + tallymeans[secondboundaryid]);
	else {
	  double updatedvolume,denominator;
	  double predstress[3] = {0.0}; //~ The predicted stresses

	  //~ Firstly guess what the target stress on boundary i will become
	  //~ If the sinudoid is larger than previously, the target stress will generally increase
	  //~ starget still contains the value from the previous timestep
	  predstress[i] = starget[i] + cyclicparam[1][i]*(sin(8.0*atan(1.0)*update->dt*ncyclicsteps/cyclicparam[2][i]) - sin(8.0*atan(1.0)*update->dt*(ncyclicsteps-1)/cyclicparam[2][i]));
	  //~ For this target stress, find the strain rate
	  temprates[i] = -Kp[i]*(predstress[i] - tallymeans[i]);
	  if (fabs(temprates[i]) > maxrate[i] && maxrate[i] > 0.0)
	    temprates[i] < 0.0 ? temprates[i] = -maxrate[i] : temprates[i] = maxrate[i];

	  //~ Next find the approximate strain rates on the other boundaries
	  updatedvolume = (1+temprates[i]*update->dt)*initialvolume;
	  
	  if (currstep%2 == 1) {//~ currstep not yet updated, so 1 differs from the 0 used below
	    denominator = tallymeans[secondboundaryid]-oldmeans[secondboundaryid]+erates[secondboundaryid]*((tallymeans[firstboundaryid]-oldmeans[firstboundaryid])/erates[firstboundaryid]);

	    if (erates[firstboundaryid] != 0.0 && denominator != 0.0)
	      temprates[firstboundaryid] = (((tallymeans[secondboundaryid]-oldmeans[secondboundaryid])*(initialvolume-updatedvolume)/(updatedvolume*update->dt))-erates[secondboundaryid]*(tallymeans[firstboundaryid]-tallymeans[secondboundaryid]))/denominator;
	    else temprates[firstboundaryid] = erates[secondboundaryid]; //~ Use the rate on the other boundary from the previous step

	    temprates[secondboundaryid] = (initialvolume-updatedvolume*(1+temprates[firstboundaryid]*update->dt))/(updatedvolume*(1+temprates[firstboundaryid]*update->dt)*update->dt);
	  } else {
	    denominator = tallymeans[firstboundaryid]-oldmeans[firstboundaryid]+erates[firstboundaryid]*((tallymeans[secondboundaryid]-oldmeans[secondboundaryid])/erates[secondboundaryid]);

	    if (erates[secondboundaryid] != 0.0 && denominator != 0.0)
	      temprates[secondboundaryid] = (((tallymeans[firstboundaryid]-oldmeans[firstboundaryid])*(initialvolume-updatedvolume)/(updatedvolume*update->dt))-erates[firstboundaryid]*(tallymeans[secondboundaryid]-tallymeans[firstboundaryid]))/denominator;
	    else temprates[secondboundaryid] = erates[firstboundaryid];

	    temprates[firstboundaryid] = (initialvolume-updatedvolume*(1+temprates[secondboundaryid]*update->dt))/(updatedvolume*(1+temprates[secondboundaryid]*update->dt)*update->dt);
	  }

	  //~ Make sure they're not ridiculously high
	  if (fabs(temprates[firstboundaryid]) > maxrate[firstboundaryid] && maxrate[firstboundaryid] > 0.0)
	    temprates[firstboundaryid] < 0.0 ? temprates[firstboundaryid] = -maxrate[firstboundaryid] : temprates[firstboundaryid] = maxrate[firstboundaryid];
	  if (fabs(temprates[secondboundaryid]) > maxrate[secondboundaryid] && maxrate[secondboundaryid] > 0.0)
	    temprates[secondboundaryid] < 0.0 ? temprates[secondboundaryid] = -maxrate[secondboundaryid] : temprates[secondboundaryid] = maxrate[secondboundaryid];

	  //~ Now use these to find the predicted stresses in the two linkvolstress directions and thereby find the minor principal stress
	  int tempcounter = 0;
	  if (erates[firstboundaryid] != 0.0) {
	    minorpstress += tallymeans[firstboundaryid] + (tallymeans[firstboundaryid]-oldmeans[firstboundaryid])*temprates[firstboundaryid]/erates[firstboundaryid];
	    tempcounter++;
	  }

	  if (erates[secondboundaryid] != 0.0) {
	    minorpstress += tallymeans[secondboundaryid] + (tallymeans[secondboundaryid]-oldmeans[secondboundaryid])*temprates[secondboundaryid]/erates[secondboundaryid];
	    tempcounter++;
	  }

	  if (tempcounter == 2) minorpstress *= 0.5;
	  else if (tempcounter == 0) minorpstress = 0.5*(tallymeans[firstboundaryid] + tallymeans[secondboundaryid]);
	}

	starget[i] = minorpstress + cyclicparam[0][i] + cyclicparam[1][i]*sin(8.0*atan(1.0)*update->dt*ncyclicsteps/cyclicparam[2][i]);
      }
    }

    //~ Reset temprates to 0
    for (int j = 0; j < 6; j++)
      temprates[j] = 0.0;
  }

  //~ Update the target stresses if the stresses on the boundary are calculated using constantb
  if (constbctrl) {
    for (int i = 0; i < 3; i++) {
      if (constbflag[i] > 0) {
	//~ Predict stresses on the other two boundaries
	double predstress[3] = {0.0};
	int firstboundaryid = (i+1)%3;
	int secondboundaryid = (i+2)%3;
	
	//~ Boundary 1
	if ((update->ntimestep - update->beginstep) > 1 && erates[firstboundaryid] != 0.0) {
	  if (strflag[firstboundaryid] == 1) {//~ If stress control
	    temprates[firstboundaryid] = -Kp[firstboundaryid]*(starget[firstboundaryid] - tallymeans[firstboundaryid]);
	    if (fabs(temprates[firstboundaryid]) > maxrate[firstboundaryid] && maxrate[firstboundaryid] > 0.0)
	      temprates[firstboundaryid] < 0.0 ? temprates[firstboundaryid] = -maxrate[firstboundaryid] : temprates[firstboundaryid] = maxrate[firstboundaryid];
	  } else temprates[firstboundaryid] = erates[firstboundaryid]; //~ Strain control

	  predstress[firstboundaryid] = tallymeans[firstboundaryid] + (tallymeans[firstboundaryid]-oldmeans[firstboundaryid])*temprates[firstboundaryid]/erates[firstboundaryid];
	} else predstress[firstboundaryid] = tallymeans[firstboundaryid];

	//~ Boundary 2
	if ((update->ntimestep - update->beginstep) > 1 && erates[secondboundaryid] != 0.0) {
	  if (strflag[secondboundaryid] == 1) {//~ If stress control
	    temprates[secondboundaryid] = -Kp[secondboundaryid]*(starget[secondboundaryid] - tallymeans[secondboundaryid]);
	    if (fabs(temprates[secondboundaryid]) > maxrate[secondboundaryid] && maxrate[secondboundaryid] > 0.0)
	      temprates[secondboundaryid] < 0.0 ? temprates[secondboundaryid] = -maxrate[secondboundaryid] : temprates[secondboundaryid] = maxrate[secondboundaryid];
	  } else temprates[secondboundaryid] = erates[secondboundaryid]; //~ Strain control

	  predstress[secondboundaryid] = tallymeans[secondboundaryid] + (tallymeans[secondboundaryid]-oldmeans[secondboundaryid])*temprates[secondboundaryid]/erates[secondboundaryid];
	} else predstress[secondboundaryid] = tallymeans[secondboundaryid];

	//~ Now calculate the target stress for boundary i using the user-defined b value
	//~ constbflag[i] == 1: {y,z}, {x,z} or {x,y}; constbflag[i] == 2: {z,y}, {z,x} or {y,x}; 
	if (constbflag[i] == 1) {
	  if (firstboundaryid > secondboundaryid)
	    starget[i] = cyclicparam[0][i]*predstress[secondboundaryid] + (1-cyclicparam[0][i])*predstress[firstboundaryid];
	  else
	    starget[i] = cyclicparam[0][i]*predstress[firstboundaryid] + (1-cyclicparam[0][i])*predstress[secondboundaryid];
	} else {
	  if (firstboundaryid > secondboundaryid)
	    starget[i] = cyclicparam[0][i]*predstress[firstboundaryid] + (1-cyclicparam[0][i])*predstress[secondboundaryid];
	  else
	    starget[i] = cyclicparam[0][i]*predstress[secondboundaryid] + (1-cyclicparam[0][i])*predstress[firstboundaryid];
	}
	
	//~ Reset temprates to 0
	for (int j = 0; j < 3; j++)
	  temprates[j] = 0.0;

	break; //~ Only one boundary can be controlled in a simulation with constantb
      }
    }
  }

  //~ Now increment currstep and calculate the updated parameters for fix_deform
  currstep++;
  if (lvstressflag == 0 && constpflag == 0 && constqflag == 0) eval_fix_deform_params_basic();
  else eval_fix_deform_params_special();
  
  update_fix_deform_params(); //~ Fix_deform must be updated. 
  deffix->init();
  deffix->end_of_step(); //~ Run the fix_deform end_of_step function

  if (extrasteps == 0 && me == 0)
    error->message(FLERR,"The specified stress tolerance was attained");
}

/* ---------------------------------------------------------------------- */

void FixMultistress::eval_fix_deform_params_basic()
{
  /*~ For this simple approach, it is necessary to assume that the stress in
    future timesteps will increase linearly if the rate of boundary motion
    remains constant.

    Note that erates[i] is not updated unless strflag[i] is active.*/

  for (int i = 0; i < 6; i++) {
    if (strflag[i] == 1) {
      //~ The first step cannot use general PID control
      //      if (currstep > 1) {
      //~ For P control only:
      temprates[i] = -Kp[i]*(starget[i] - tallymeans[i]);

      //~ For general PID control:
      // double rates[6] = {0.0}; //~ The rates of change over a timestep
      // if (stabtestflag[i] == 1 || fabs(erates[i]) < 1000*Kp[i]) ictrlflag[i] = 1;
      // cumul[i] += (starget[i] - 0.5*(tallymeans[i]+oldmeans[i]))*update->dt*ictrlflag[i];
      // rates[i] = (oldmeans[i]-tallymeans[i])/update->dt; //~ Careful with signs here	
      // temprates[i] = -Kp[i]*(starget[i] - tallymeans[i] + cumul[i]/ti[i] + td[i]*rates[i]);
      
      //~ Check for the stress component crossing the target value
      //	if ((starget[i]- oldmeans[i])*(starget[i] - tallymeans[i]) < 0) stabtestflag[i] = 1;
      
      //~ Run an optional test for instability
      //	if (instabcheck == 1 && stabtestflag[i] == 1) instability_test(i);
      //      } else temprates[i] = -Kp[i]*(starget[i] - tallymeans[i]);
      
      //~ Impose the strain rate limitation, if specified
      if (fabs(temprates[i]) > maxrate[i] && maxrate[i] > 0.0) {
	if (temprates[i] < 0.0) temprates[i] = -maxrate[i]; //~ maxrate is an absolute value
	else temprates[i] = maxrate[i];
      }
    }
  }

  for (int i = 0; i < 6; i++)
    if (strflag[i] == 1) erates[i] = temprates[i];
}

/* ---------------------------------------------------------------------- */

void FixMultistress::eval_fix_deform_params_special()
{
  /*~ Used to choose between the linkvolstress and constantpq functions which
    simply calculate an appropriate set of three strain rates for the special
    cases of constant volume and constant p' or q, respectively.*/

  iterateflag = 1; //~ May be needed for an iterative conditional code branch

  if (lvstressflag == 1) eval_fix_deform_params_linkvolstress();
  else eval_fix_deform_params_constantpq();

  //~ The three triclinic directions are controlled as normal.
  if (triclinic == 1) {
    for (int i = 3; i < 6; i++) {
      if (strflag[i] == 1) {
	//~ For P control only:
	temprates[i] = -Kp[i]*(starget[i] - tallymeans[i]);

	//~ Impose the strain rate limitation, if specified
	if (fabs(temprates[i]) > maxrate[i] && maxrate[i] > 0.0) {
	  if (temprates[i] < 0.0) temprates[i] = -maxrate[i]; //~ maxrate is an absolute value
	  else temprates[i] = maxrate[i];
	}
      }
    }
  }

  for (int i = 0; i < 6; i++)
    if (strflag[i] == 1 || (i < 3 && (linkvolstress[i] == 1 || constantpq[i] == 1)))
      erates[i] = temprates[i];
}

/* ---------------------------------------------------------------------- */

void FixMultistress::eval_fix_deform_params_linkvolstress()
{
  /*~ This version of eval_fix_deform_params is substantially more complicated
    than eval_fix_deform_params_basic, and is invoked only if linkvolstress
    is active. The logic below was substantially revised by KH in June 2012
    to achieve more precise control of stress and volume. The previous version
    was based on updating stress and volume using different boundaries, and
    alternating every second timestep - now of historical interest only.*/

  double denominator;

  //~ Fix crushing has an option to maintain a constant void ratio
  if (constanteflag && domain->initialvolume < initialvolume)
    initialvolume = domain->initialvolume;

  //~ Note that linkvolstress will be zero only for one index
  for (int i = 0; i < 3; i++) {
    if (linkvolstress[i] == 0) {
      //~ The i direction may be controlled independently or uncontrolled.
      //~ Sort out the rates for this direction firstly.
      if (strflag[i] == 1) {
	temprates[i] = -Kp[i]*(starget[i] - tallymeans[i]);

	//~ Impose the strain rate limitation, if specified.
	if (fabs(temprates[i]) > maxrate[i] && maxrate[i] > 0.0) {
	  if (temprates[i] < 0.0) temprates[i] = -maxrate[i]; //~ maxrate is an absolute value
	  else temprates[i] = maxrate[i];
	}
      } else temprates[i] = erates[i];
  
      int firstboundaryid = (i+1)%3;
      int secondboundaryid = (i+2)%3;

      //~ At this point, the new volume if the other rates are set to zero is:
      double updatedvolume = (1+temprates[i]*update->dt)*(domain->boxhi[0]-domain->boxlo[0])*(domain->boxhi[1]-domain->boxlo[1])*(domain->boxhi[2]-domain->boxlo[2]);

      /*~ There are 2 unknowns: the two strain rates, temprates[firstboundaryid] and 
	temprates[secondboundaryid]. It is desired to maintain a constant volume
	and to keep the stresses on both boundaries equal. Therefore two independent
	equations can be written in the following forms:

	1) tallymeans[firstboundaryid] + stress_change_due_to_temprates[firstboundaryid]
	= tallymeans[secondboundaryid] + stress_change_due_to_temprates[secondboundaryid]

	Reasonable predictions of these stress changes can be made since oldmeans are
	stored, and erates caused oldmeans to change to tallymeans. A linear relationship
	is assumed between strain rate and change of stress, i.e., doubling the strain
	rate causes twice the change in stress.

	2) volume_after_temprates[firstboundaryid]_&_temprates[secondboundaryid]_applied
	= initialvolume.

	Though the equations below look awful, all that is being done is these two simple
	simultaneous equations are being solved for temprates[firstboundaryid] and 
	temprates[secondboundaryid].

	A slight complication is that the equations alternate between firstboundaryid
	and secondboundaryid on successive timesteps (controlled by currstep%2). This is
	done to prevent bias, but of course the equations remain the same.

	There is a denominator in one of the equations which can be zero in some situations
	or contain a term which becomes infinite if a strain rate on the previous timestep
	is zero. The denominator is calculated separately and an if conditional is used to
	check that neither of these scenarios occur.*/
      if (update->ntimestep - update->beginstep > 1) {
	if (currstep%2 == 0) {
	  denominator = tallymeans[secondboundaryid]-oldmeans[secondboundaryid]+erates[secondboundaryid]*((tallymeans[firstboundaryid]-oldmeans[firstboundaryid])/erates[firstboundaryid]);

	  if (erates[firstboundaryid] != 0.0 && denominator != 0.0)
	    temprates[firstboundaryid] = (((tallymeans[secondboundaryid]-oldmeans[secondboundaryid])*(initialvolume-updatedvolume)/(updatedvolume*update->dt))-erates[secondboundaryid]*(tallymeans[firstboundaryid]-tallymeans[secondboundaryid]))/denominator;
	  else temprates[firstboundaryid] = erates[secondboundaryid]; //~ Use the rate on the other boundary from the previous step

	  temprates[secondboundaryid] = (initialvolume-updatedvolume*(1+temprates[firstboundaryid]*update->dt))/(updatedvolume*(1+temprates[firstboundaryid]*update->dt)*update->dt);
	} else {
	  denominator = tallymeans[firstboundaryid]-oldmeans[firstboundaryid]+erates[firstboundaryid]*((tallymeans[secondboundaryid]-oldmeans[secondboundaryid])/erates[secondboundaryid]);

	  if (erates[secondboundaryid] != 0.0 && denominator != 0.0)
	    temprates[secondboundaryid] = (((tallymeans[firstboundaryid]-oldmeans[firstboundaryid])*(initialvolume-updatedvolume)/(updatedvolume*update->dt))-erates[firstboundaryid]*(tallymeans[secondboundaryid]-tallymeans[firstboundaryid]))/denominator;
	  else temprates[secondboundaryid] = erates[firstboundaryid];

	  temprates[firstboundaryid] = (initialvolume-updatedvolume*(1+temprates[secondboundaryid]*update->dt))/(updatedvolume*(1+temprates[secondboundaryid]*update->dt)*update->dt);
	}
      } else {
	/*~ On the first timestep, assume (in order of preference) that...
	  i) temprates[firstboundaryid] = erates[firstboundaryid] if non-zero
	  ii) or temprates[secondboundaryid] = erates[secondboundaryid]
	  iii) or else assume temprates[firstboundaryid] = temprates[secondboundaryid]*/
	if (force->pair_match("gran",0)) {
	  if (force->pair->ierates[firstboundaryid] != 0.0) {
	    temprates[firstboundaryid] = force->pair->ierates[firstboundaryid];
	    temprates[secondboundaryid] = (initialvolume-updatedvolume*(1+temprates[firstboundaryid]*update->dt))/(updatedvolume*(1+temprates[firstboundaryid]*update->dt)*update->dt);
	  } else if (force->pair->ierates[secondboundaryid] != 0.0) {
	    temprates[secondboundaryid] = force->pair->ierates[secondboundaryid];
	    temprates[firstboundaryid] = (initialvolume-updatedvolume*(1+temprates[secondboundaryid]*update->dt))/(updatedvolume*(1+temprates[secondboundaryid]*update->dt)*update->dt);
	  } else {
	    temprates[firstboundaryid] = ((initialvolume/updatedvolume)-1)/(2*update->dt);
	    temprates[secondboundaryid] = temprates[firstboundaryid];
	  }
	} else {
	  temprates[firstboundaryid] = ((initialvolume/updatedvolume)-1)/(2*update->dt);
	  temprates[secondboundaryid] = temprates[firstboundaryid];
	}
      }

      //~ temprates[firstboundaryid] or temprates[secondboundaryid] could exceed maxrate (if defined)
      if ((maxrate[firstboundaryid] > 0.0 && fabs(temprates[firstboundaryid]) > maxrate[firstboundaryid]) || (maxrate[secondboundaryid] > 0.0 && fabs(temprates[secondboundaryid]) > maxrate[secondboundaryid])) {
	/*~ There is a great deal of repetition below, but this is most efficient. It is possible
	  to achieve the same effect using only one while condition, but the nested if statements
	  would have to be executed on each iteration which is less efficient. Therefore keep
	  the while loops innermost.*/
	if (fabs(temprates[firstboundaryid]) > maxrate[firstboundaryid] && fabs(temprates[secondboundaryid]) > maxrate[secondboundaryid]) {
	  /*~ If the signs differ, there is little problem; just fix the dimension which has
	    the opposite sign as temprates[i] at its maxrate and correct the other dimension
	    appropriately to keep the volume constant.*/
	  if (temprates[firstboundaryid]*temprates[secondboundaryid] < 0.0) {
	    if (temprates[firstboundaryid]*temprates[i] < 0.0) linkvolstress_loop(temprates,firstboundaryid,secondboundaryid,updatedvolume);
	    else linkvolstress_loop(temprates,secondboundaryid,firstboundaryid,updatedvolume);
	  } else {
	    /*~ If the signs are the same, favour adjusting the boundary which is liable to
	      bring the stresses closer together.*/
	    if (temprates[firstboundaryid] > 0.0) {
	      //~ If both are positive, temprates[i] is negative (compressive) - the most common scenario
	      //~ Move the boundary with the highest stress by a greater amount to relieve the stresses
	      if (tallymeans[firstboundaryid] > tallymeans[secondboundaryid])
		linkvolstress_loop(temprates,firstboundaryid,secondboundaryid,updatedvolume);
	      else linkvolstress_loop(temprates,secondboundaryid,firstboundaryid,updatedvolume);
	    } else {
	      //~ Both rates are negative - favour the boundary with the lowest stress
	      if (tallymeans[firstboundaryid] < tallymeans[secondboundaryid])
		linkvolstress_loop(temprates,firstboundaryid,secondboundaryid,updatedvolume);
	      else linkvolstress_loop(temprates,secondboundaryid,firstboundaryid,updatedvolume);
	    }
	  }
	} else {
	  //~ Only one of the temprates exceeds its maxrate.
	  if (fabs(temprates[firstboundaryid]) > maxrate[firstboundaryid])
	    linkvolstress_loop(temprates,firstboundaryid,secondboundaryid,updatedvolume);
	  else linkvolstress_loop(temprates,secondboundaryid,firstboundaryid,updatedvolume);
	}
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMultistress::eval_fix_deform_params_constantpq()
{
  /*~ This version of eval_fix_deform_params is invoked only if either 
    constantp or constantq are active.*/

  //~ Note that constantpq will be zero only for one index
  for (int i = 0; i < 3; i++) {
    if (constantpq[i] == 0) {
      //~ The i direction may be controlled independently or uncontrolled.
      //~ Sort out the rates for this direction firstly.
      if (strflag[i] == 1) {
	temprates[i] = -Kp[i]*(starget[i] - tallymeans[i]);

	//~ Impose the strain rate limitation, if specified.
	if (fabs(temprates[i]) > maxrate[i] && maxrate[i] > 0.0) {
	  if (temprates[i] < 0.0) temprates[i] = -maxrate[i]; //~ maxrate is an absolute value
	  else temprates[i] = maxrate[i];
	}
      } else temprates[i] = erates[i];

      int firstboundaryid = (i+1)%3;
      int secondboundaryid = (i+2)%3;

      /*~ There are 2 unknowns: the two strain rates, temprates[firstboundaryid] and 
	temprates[secondboundaryid]. As suggested by Xin Huang, the stress changes on
	one timestep should be small so we can simplify the old approach. Firstly, we
	have the strain rate on boundary i. Estimate the resulting stress on boundary
	i, and then calculate the stresses on the other two boundaries knowing that
	these are constrained to be equal.

	Once all the stresses have been predicted, we can write two decoupled independent
	equations to find the required strain rates for the two boundaries:

	1) predicted[firstboundaryid] - tallymeans[firstboundaryid] = stress_change_due_to_temprates[firstboundaryid]
	2) predicted[secondboundaryid] - tallymeans[secondboundaryid] = stress_change_due_to_temprates[secondboundaryid]

	Reasonable predictions of these stress changes can be made since oldmeans are
	stored, and erates caused oldmeans to change to tallymeans. A linear relationship
	is assumed between strain rate and change of stress, i.e., doubling the strain
	rate causes twice the change in stress.

	There are denominators which can become zero, so if conditionals are used to check.*/
      if (update->ntimestep - update->beginstep > 1) {
	//~ Predict stress on boundary i
	double predstress[3] = {0.0};
	
	if (erates[i] != 0.0)
	  predstress[i] = tallymeans[i] + temprates[i]*(tallymeans[i]-oldmeans[i])/erates[i];
	else predstress[i] = tallymeans[i];

	if (constpflag) {
	  //~ Check whether it will be possible to maintain p' at its initial value
	  if (predstress[i] > 3.0*meaneffectivestress)
	    error->all(FLERR,"p' cannot be maintained at its desired value");

	  //~ Calculate the predicted stresses on the other two boundaries
	  predstress[firstboundaryid] = predstress[secondboundaryid] = 0.5*(3.0*meaneffectivestress-predstress[i]);
	} else {//~ constqflag active
	  if (predstress[i] < deviatorstress)
	    error->all(FLERR,"q cannot be maintained at its desired value");

	  //~ Calculate the predicted stresses on the other two boundaries
	  predstress[firstboundaryid] = predstress[secondboundaryid] = predstress[i]-deviatorstress;
	}

	//~ Calculate the required movements of the lateral boundaries
	if (tallymeans[firstboundaryid] != oldmeans[firstboundaryid] && erates[firstboundaryid] != 0.0)
	  temprates[firstboundaryid] = erates[firstboundaryid]*(predstress[firstboundaryid]-tallymeans[firstboundaryid])/(tallymeans[firstboundaryid]-oldmeans[firstboundaryid]);
	else temprates[firstboundaryid] = -Kp[firstboundaryid]*(predstress[firstboundaryid] - tallymeans[firstboundaryid]);

	if (tallymeans[secondboundaryid] != oldmeans[secondboundaryid] && erates[secondboundaryid] != 0.0)
	  temprates[secondboundaryid] = erates[secondboundaryid]*(predstress[secondboundaryid]-tallymeans[secondboundaryid])/(tallymeans[secondboundaryid]-oldmeans[secondboundaryid]);
	else temprates[secondboundaryid] = -Kp[secondboundaryid]*(predstress[secondboundaryid] - tallymeans[secondboundaryid]);

	/*~ Because of the way in which the equations are structured, it is possible for both of the
	  temprates to fall to zero. In this case, the loop will always return zeros for the temprates
	  on future timesteps. Check here for this condition.*/
	if (temprates[firstboundaryid] == 0.0 && temprates[secondboundaryid] == 0.0) {
	  temprates[firstboundaryid] = -Kp[firstboundaryid]*(predstress[firstboundaryid] - tallymeans[firstboundaryid]);
	  temprates[secondboundaryid] = -Kp[secondboundaryid]*(predstress[secondboundaryid] - tallymeans[secondboundaryid]);
	}
      } else {
	/*~ On the first timestep, assume (in order of preference) that...
	  i) temprates[firstboundaryid] = erates[firstboundaryid] if non-zero
	  and temprates[secondboundaryid] = erates[secondboundaryid]
	  ii) or else assume temprates[firstboundaryid] = temprates[secondboundaryid] = -0.5*temprates[i]*/
	if (force->pair_match("gran",0)) {
	  if (force->pair->ierates[firstboundaryid] != 0.0 && force->pair->ierates[secondboundaryid] != 0.0) {
	    temprates[firstboundaryid] = force->pair->ierates[firstboundaryid];
	    temprates[secondboundaryid] = force->pair->ierates[secondboundaryid];
	  } else temprates[firstboundaryid] = temprates[secondboundaryid] = -0.5*temprates[i];
	} else temprates[firstboundaryid] = temprates[secondboundaryid] = -0.5*temprates[i];
      }

      //~ temprates[firstboundaryid] or temprates[secondboundaryid] could exceed maxrate (if defined)
      if ((maxrate[firstboundaryid] > 0.0 && fabs(temprates[firstboundaryid]) > maxrate[firstboundaryid]) || (maxrate[secondboundaryid] > 0.0 && fabs(temprates[secondboundaryid]) > maxrate[secondboundaryid])) {
	if (fabs(temprates[firstboundaryid]) > maxrate[firstboundaryid] && fabs(temprates[secondboundaryid]) > maxrate[secondboundaryid]) {
	  if (temprates[firstboundaryid]*temprates[secondboundaryid] < 0.0) {
	    if (temprates[firstboundaryid]*temprates[i] < 0.0) constantpq_loop(temprates,firstboundaryid,secondboundaryid);
	    else constantpq_loop(temprates,secondboundaryid,firstboundaryid);
	  } else {
	    if (temprates[firstboundaryid] > 0.0) {
	      if (tallymeans[firstboundaryid] > tallymeans[secondboundaryid])
		constantpq_loop(temprates,firstboundaryid,secondboundaryid);
	      else constantpq_loop(temprates,secondboundaryid,firstboundaryid);
	    } else {
	      if (tallymeans[firstboundaryid] < tallymeans[secondboundaryid])
		constantpq_loop(temprates,firstboundaryid,secondboundaryid);
	      else constantpq_loop(temprates,secondboundaryid,firstboundaryid);
	    }
	  }
	} else {
	  if (fabs(temprates[firstboundaryid]) > maxrate[firstboundaryid])	
	    constantpq_loop(temprates,firstboundaryid,secondboundaryid);
   	  else constantpq_loop(temprates,secondboundaryid,firstboundaryid);
   	}
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMultistress::linkvolstress_loop(double therates[], int firstid, int secondid, double upvol)
{
  //~ therates is equivalent to temprates, which is passed call-by-reference
  while (iterateflag >= 1) {
    therates[firstid] < 0.0 ? therates[firstid] = -maxrate[firstid]/iterateflag : therates[firstid] = maxrate[firstid]/iterateflag;
    therates[secondid] = (initialvolume-upvol*(1+therates[firstid]*update->dt))/(upvol*(1+therates[firstid]*update->dt)*update->dt);
 
    fabs(therates[secondid]) <= maxrate[secondid] ? iterateflag = 0 : iterateflag++;
    if (iterateflag == 20) {
      therates[firstid] = 0.0; 
      therates[secondid] < 0.0 ? therates[secondid] = -maxrate[secondid] : therates[secondid] = maxrate[secondid];
      iterateflag = 0; //~ To guarantee that an infinite loop does not occur
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMultistress::constantpq_loop(double therates[], int firstid, int secondid)
{
  if (fabs(therates[firstid]) > maxrate[firstid])
    therates[firstid] < 0.0 ? therates[firstid] = -maxrate[firstid] : therates[firstid] = maxrate[firstid];
  
  if (fabs(therates[secondid]) > maxrate[secondid])
    therates[secondid] < 0.0 ? therates[secondid] = -maxrate[secondid] : therates[secondid] = maxrate[secondid];
  
  iterateflag = 0; //~ Reset this
}

/* ---------------------------------------------------------------------- */

void FixMultistress::update_fix_deform_params()
{
  //~ Get the setting of the flip flag from fix_deform
  flip = 0;
  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"deform") == 0)
      if (((FixDeform *) modify->fix[i])->flip == 1) flip = 1;
  
  //~ Delete the existing fix_deform
  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->id,id_multistress) == 0)
      modify->delete_fix(id_multistress);
  
  create_fix(); //~ Set up a replacement fix_deform
}

/* ---------------------------------------------------------------------- */

void FixMultistress::instability_test(int i)
{
  /*~ Instability arises because of the development of oscillations in
    the response(s). This can be quantified by storing the maximum and
    minimum stress components and checking where these occur relative
    to each other. If a maximum is detected, and the preceding maximum
    was followed by a minimum, this indicates oscillations of increasing
    amplitude. Vice-versa for minima.

    The structure of the 6x4 2D array is as follows:

       min_xx   step_min_xx   max_xx   step_max_xx
       min_yy   step_min_yy   max_yy   step_max_yy
       min_zz   step_min_zz   max_zz   step_max_zz
       min_xy   step_min_xy   max_xy   step_max_xy
       min_xz   step_min_xz   max_xz   step_max_xz
       min_yz   step_min_yz   max_yz   step_max_yz

    Note that all min_* and max_* elements of the array were initialised
    at the *target values.*/

  int werrflag = 0; //~ A flag used for convenience to indicate instability  
  int delttarget = 5; //~ The percentage disparity that must exist 
  double percdiff = 0.0;

  if (tallymeans[i] > instability[i][2]) {
    //~ Update the elements of the array if necessary
    instability[i][2] = tallymeans[i];
    instability[i][3] = currstep;
    percdiff = fabs(starget[i]-tallymeans[i])*100/starget[i];
    if (instability[i][1] > instability[i][3] && percdiff > delttarget) werrflag = 1;
  } else if (tallymeans[i] < instability[i][0]) {
    instability[i][0] = tallymeans[i];
    instability[i][1] = currstep;
    percdiff = fabs(starget[i]-tallymeans[i])*100/starget[i];
    if (instability[i][1] < instability[i][3] && percdiff > delttarget) werrflag = 1;
  }

  if (werrflag == 1) error->all(FLERR,"Unstable oscillations were detected");
}

/* ---------------------------------------------------------------------- */

double *FixMultistress::param_export()
{
  /*~ This is necessary to allow the strain rates to be exported to the
    pair styles, which allow the relative velocities between particles to
    be updated because of the applied strain rate field.
    
    The static specifier ensures that the data exists for the duration of
    the program [KH - 9 November 2011]

    These engineering strain rates were changed to true strain rates, the
    correct strain rates required, rather than engineering strain rates.
    These true strain rates are used in FixEnergyBoundary and the granular
    pairstyles [KH - 16 April 2014]*/
  static double exprates[6];
  for (int i = 0; i < 6; i++) exprates[i] = erates[i];
  for (int i = 0; i < 3; i++) exprates[i] *= (pboundstart[2*i+1]-pboundstart[2*i])/(domain->boxhi[i] - domain->boxlo[i]);
  return exprates;
}

/* ---------------------------------------------------------------------- */

double FixMultistress::compute_vector(int i)
{
  if (currstep > 0) return tallymeans[i];
  else return 0.0;
}

/* ---------------------------------------------------------------------- */

void FixMultistress::lost_atom_check()
{
  /*~ This was added to identify rapidly whether atoms have been lost, and
    if so, to stop immediately rather than continuing to the end of the
    simulation.*/

  //~ The content is almost identical to Thermo::lost_check().

  bigint ntotal;
  bigint nblocal = atom->nlocal;
  MPI_Allreduce(&nblocal,&ntotal,1,MPI_LMP_BIGINT,MPI_SUM,world);

  if (ntotal != atom->natoms) 
    error->all(FLERR,"Atoms have been lost. Check that the simulation conditions are sensible.");
}

/* ---------------------------------------------------------------------- */

void FixMultistress::write_restart(FILE *fp)
{
  int n = 0;
  double list[4];
  list[n++] = ncyclicsteps;
  list[n++] = initialvolume;
  list[n++] = meaneffectivestress;
  list[n++] = deviatorstress;

  if (me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ---------------------------------------------------------------------- */

void FixMultistress::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;
  ncyclicsteps = static_cast<int> (list[n++]);
  initialvolume = static_cast<double> (list[n++]);
  meaneffectivestress = static_cast<double> (list[n++]);
  deviatorstress = static_cast<double> (list[n++]);
}

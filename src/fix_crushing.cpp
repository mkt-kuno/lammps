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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix_crushing.h"
#include "atom.h"
#include "atom_vec.h"
#include "update.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "random_park.h"
#include "force.h"
#include "pair.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "integrate.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "compute.h"
#include "fix_multistress.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define SMALL 1.0e-20
#define BIG   1.0e20

/* ----------------------------------------------------------------------
The syntax of the input command is as follows:

fix ID group crushing outputflag seed m sigma0 d0 a b chi alpha redtype {reduction} constante {commlimit} {m2} {sigma02} {d02} {a2} {b2} {reallocate}

   outputflag Set this at 1 to display detailed crushing output; 0 to suppress this output
   seed       Seed of the random number generator
   m          Weibull modulus
   sigma0     Characteristic strength at which 37% of particles of diameter d0 survive
   d0         Nominal particle diameter in m
   a          Slope of linear trendline of Ps vs. normalised characteristic stress
   b          y-intercept of linear trendline of Ps vs. normalised characteristic stress
   chi        Parameter in brittle failure criterion of Christensen
   alpha      Multiplicative factor by which to reduce compressive stress
   redtype    Flag = 1 if diameter reduced to just give touching contact; else 0
   reduction  If redtype = 0, the fractional reduction in radius between 0 and 1
   constante  Flag = 1 if void ratio is held constant; else 0
   commlimit  Optional: a comminution limit on radii in m
   m2         Optional: Weibull modulus for > 1 breakage
   sigma02    Optional: sigma0 for particles of diam. d02 after > 1 breakage
   d02        Optional: Nominal particle diameter in m for > 1 breakage
   a2         Optional: a for > 1 breakage
   b2         Optional: b for > 1 breakage
   reallocate Optional: keyword meaning that strengths should be reallocated to particles
 ---------------------------------------------------------------------- */

FixCrushing::FixCrushing(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  cparams(NULL)
{
  restart_global = 1; //~ Global information is saved to the restart file
  restart_peratom = 1; //~ As is per-atom information
  nevery = 1; //~ Set how often to run the end_of_step function
  peratom_flag = 1;
  size_peratom_cols = 3; //~ sigma_c, sigma_t, number of failures
  peratom_freq = 1;
  create_attribute = 1;
  force_reneighbor = 1; //~ This fix can induce neighbour list rebuilds
  next_reneighbor = -1; //~ Initially ensure no rebuild is caused

  if (narg < 14 || narg > 22)
    error->all(FLERR,"Illegal fix crushing command");

  if (strcmp(atom->atom_style,"sphere") != 0)
    error->all(FLERR,"Fix crushing is defined only for the sphere atom style");
  
  if (domain->dimension == 2)
    error->all(FLERR,"The 2D case has not been written for fix_crushing");

  //~ Issue a warning if SI units are not used
  if (strcmp(update->unit_style,"si") != 0)
    error->warning(FLERR,"SI units are best when using fix crushing");

  //~ Read in all user-specified data
  displaymessages = force->inumeric(FLERR,arg[3]); //~ Write information to the screen (1) or not (0)
  seed = force->inumeric(FLERR,arg[4]);

  for (int i = 0; i < 5; i++)
    weibullparams[i][0] = force->numeric(FLERR,arg[5+i]);

  chiplusone = force->numeric(FLERR,arg[10])+1.0;
  alphafactor = force->numeric(FLERR,arg[11]);
  redtype = force->inumeric(FLERR,arg[12]);

  int iarg = 13;
  if (redtype == 0) {
    reduction = force->numeric(FLERR,arg[iarg]);

    if (reduction <= 0 || reduction >= 1)
      error->all(FLERR,"Radius reduction in fix crushing must be between 0 and 1");

    iarg++;
  } else if (redtype != 1) error->all(FLERR,"Illegal redtype specification in fix crushing command");

  constante = force->inumeric(FLERR,arg[iarg]);

  //~ Check if the reallocate keyword is present
  reallocateflag = 0;
  int numarg = narg;

  if (strcmp(arg[narg-1],"reallocate") == 0) {
    reallocateflag = 1;
    numarg--;
  }

  if (numarg == iarg+1 || numarg == iarg+6) commlimit = -1.0;
  else if (numarg == iarg+2 || numarg == iarg+7) {
    commlimit = force->numeric(FLERR,arg[iarg+1]);
    iarg++;
  } else error->all(FLERR,"Illegal number of arguments in fix crushing command");
  
  if (numarg == iarg+6) {
    //~ Read in m, sigma0, d0, a and b for second and subsequent breakages
    for (int i = 0; i < 5; i++)
      weibullparams[i][1] = force->numeric(FLERR,arg[iarg+1+i]);
  } else { //~ Use the same values for the first, second... breakages
    for (int i = 0; i < 5; i++)
      weibullparams[i][1] = weibullparams[i][0];
  }

  //~ Now check that these inputs are sensible
  if (seed <= 0 || weibullparams[0][0] <= 0.0 || weibullparams[1][0] <= 0.0 || 
      weibullparams[2][0] <= 0.0 || chiplusone <= 1.0 || weibullparams[0][1] <= 0.0 ||
      weibullparams[1][1] <= 0.0 || weibullparams[2][1] <= 0.0)
    error->all(FLERR,"Fix crushing parameter must be positive");

  if (displaymessages != 0 && displaymessages != 1)
    error->all(FLERR,"outputflag in fix crushing must be 0 or 1");

  if (constante != 0 && constante != 1)
    error->all(FLERR,"constante flag in fix crushing must be 0 or 1");

  //~ Confirm that a granular pairstyle is used
  if (!force->pair_match("gran",0))
    error->all(FLERR,"A granular pairstyle must be used for fix crushing");

  random = new RanPark(lmp,seed); //~ Establish the random number generator

  // perform initial allocation of atom-based array
  // register with Atom class
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  atom->add_callback(1);

  /*~ The cparams array stores three pieces of data for each
    atom: 1) the uniaxial compressive strength; 2) the uniaxial 
    tensile strength; 3) the number of particle breakage events.
    Initialise everything at zero regardless of masking.*/
  for (int i = 0; i < atom->nlocal; i++)
    cparams[i][0] = cparams[i][1] = cparams[i][2] = 0.0;

  PI = 4.0*atan(1.0); //~ Calculate pi once as it is used often

  /*~ Check that all relevant (masked) particles have non-zero radii
    and if a comminution limit is specified, ensure that the initial
    radii are greater than or equal to this limit*/
  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double minrad = BIG; //~ minrad contains the radius of the smallest atom

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  for (int i = 0; i < nlocal; i++)
    if ((mask[i] & groupbit) && radius[i] < minrad)
      minrad = radius[i];

  if (minrad <= SMALL)
    error->all(FLERR,"Fix crushing requires extended particles");
  else if (commlimit > 0.0 && minrad < commlimit)
    error->all(FLERR,"Particles smaller than the comminution limit are in the system");
  if (commlimit > 0.0) radiusparticletoinsert = commlimit;
  else MPI_Allreduce(&minrad,&radiusparticletoinsert,1,MPI_DOUBLE,MPI_MIN,world);

  /*~ Calculate the volume of the small particles which may be 
    inserted into voids*/
  double fourpioverthree = 4.0*PI/3.0;
  volumeparticletoinsert = fourpioverthree*radiusparticletoinsert*radiusparticletoinsert*radiusparticletoinsert;

  /*~ Calculate the total volume of particles and the domain initially
    if the void ratio is to be held constant*/
  if (constante == 1) {
    double pvolume = 0.0; //~ Volume of particles on a processor
    double totalvolume = 1.0;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
	pvolume += fourpioverthree*radius[i]*radius[i]*radius[i];
  
    //~ Gather pvolume from all processors in totalpvolume
    MPI_Allreduce(&pvolume,&totalpvolume,1,MPI_DOUBLE,MPI_SUM,world);

    //~ Find the current size of the simulation domain
    for (int i = 0; i < domain->dimension; i++)
      totalvolume *= (domain->boxhi[i]-domain->boxlo[i]);
    
    voidratio = (totalvolume - totalpvolume)/totalpvolume;
  }
  
  //~ Initialise the cumulative loss of volume from the system
  cumulredvolume = 0.0;
}

/* ---------------------------------------------------------------------- */

FixCrushing::~FixCrushing()
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id,0);
  atom->delete_callback(id,1);

  // delete locally stored array
  memory->destroy(cparams);

  //~ Delete the stress compute if fix_multistress is not active
  mstressid = -1;
  for (int q = 0; q < modify->nfix; q++)
    if (strcmp(modify->fix[q]->style,"multistress") == 0) mstressid = q;

  if (mstressid < 0) {
    modify->delete_compute(id_stress);
    delete [] id_stress;
  }

  //~ Also delete the connectivity compute
  modify->delete_compute("crush_coord");

  delete random;
}

/* ---------------------------------------------------------------------- */

int FixCrushing::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCrushing::init()
{
  /*~ Allocate initial strengths to all particles. This is relevant only
    if the data are not read in from a restart file or reallocateflag = 1.
    Note that the compressive strengths are negative, if present.*/
  int weibullparamflag = 0;
  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (force->pair_match("gran/hertz",0) || force->pair_match("gran/shm",0)) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
	if (cparams[i][0] > -1.0*SMALL || reallocateflag) {
	  cparams[i][0] = strength_calculation(i,radius[i],weibullparamflag);
	  cparams[i][1] = -1.0*cparams[i][0]/chiplusone;
	}
  } else
    error->all(FLERR,"Currently fix crushing requires a Hertzian pairstyle");

  /*~ If the particle strengths have been reallocated, need to reset
    reallocateflag so that they are not reallocated again later when init
    is rerun.*/
  reallocateflag = 0;

  //~ If redtype == 1, need to set up a mechanism for neighbour list updating
  if (redtype == 1) {
    // need a full neighbor list, built whenever re-neighboring occurs
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->fix = 1;
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
  }
}

/* ---------------------------------------------------------------------- */

void FixCrushing::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixCrushing::set_weibull_parameters(int numbreaks)
{
  m = weibullparams[0][numbreaks];
  sigma0 = weibullparams[1][numbreaks];
  d0 = weibullparams[2][numbreaks];
  slopea = weibullparams[3][numbreaks];
  interceptb = weibullparams[4][numbreaks];
}

/* ---------------------------------------------------------------------- */

double FixCrushing::strength_calculation(int i, double radius, int weibullparamflag)
{
  /*~ Assign a suitable uniaxial tensile and compressive strength to 
    particle i*/

  //~ Import data from pair and calculate G and Poisson's ratio
  double kn,kt,xmu,smod,pr;
  kn = force->pair->kn;
  kt = force->pair->kt;
  xmu = force->pair->xmu;
  smod = 0.75*kn*kt/(3.0*kn-kt);
  pr = (3.0*kn-2.0*kt)/(3.0*kn-kt);

  //~ Pick appropriate values for m, sigma0, d0, a and b from weibullparams
  set_weibull_parameters(weibullparamflag);

  double firstfixedterm = pow(2.0*radius/d0,-3.0/m);
  double secondfixedterm = 3.0*(3.0/32.0 + sqrt(2.0)/24.0 + xmu*(sqrt(2.0)/12.0 - 0.25) + xmu*xmu*(0.5 - sqrt(2.0)/3.0))/((2.0-sqrt(2.0))*(1.0+xmu)); //~ Russell & Muir-Wood, 2009, Eq. 19
  double sigmaf = sigma0*firstfixedterm*(random->uniform()-interceptb)/slopea;
  double forcef = fabs(4.0*sigmaf*radius*radius); //~ abs in case sigmaf < 0

  double estar,contactarea,compressivestrength;

  /*~ Assume contact with a sphere of infinite radius and stiffness. Recall
    that Young's modulus is 2*smod*(1+pr).*/
  estar = 2.0*smod*(1.0+pr)/(1.0-pr*pr);
  contactarea = PI*pow(0.75*forcef*radius/estar,(2.0/3.0));
  
  /*~ The -abs operation below is used to yield negative compressive strengths,
    while the alphafactor multiplication serves to reduce the compressive
    strengths to more reasonable values.*/
  compressivestrength = -alphafactor*fabs(secondfixedterm*forcef/contactarea); //~ Russell & Muir-Wood, 2009, Eq. 19

  return compressivestrength;
}

/* ---------------------------------------------------------------------- */

void FixCrushing::setup(int vflag)
{
  //~ Set up or find an existing stress compute
  
  //~ Determine whether fix_multistress is active
  mstressid = -1;

  for (int q = 0; q < modify->nfix; q++)
    if (strcmp(modify->fix[q]->style,"multistress") == 0) {
      mstressid = q;
 
      if (constante == 1) {
	//~ Set constanteflag in fix multistress to 1 if lvstressflag == 1
	int lvstressflag = ((FixMultistress *) modify->fix[mstressid])->lvstressflag;
	
	if (lvstressflag == 0)
	  error->all(FLERR,"Constant e option of fix crushing requires the linkvolstress option of fix multistress to be active");
      }
      break;
    }

  /*~ If fix multistress is active, find the pointer to the 
    existing stress compute; otherwise set up a new stress compute*/
  if (mstressid >= 0) {
    id_stress = new char[20];
    strcpy(id_stress,modify->fix[mstressid]->id);
  } else {
    int n = strlen(id) + strlen("_stress") + 1;
    id_stress = new char[n];
    strcpy(id_stress,id);
  }

  strcat(id_stress,"_stress");

  //~ Check whether the compute already exists
  int icompute = modify->find_compute(id_stress);

  if (mstressid < 0 && icompute < 0) {
    char **snewarg = new char*[7];
    snewarg[0] = id_stress;
    snewarg[1] = (char *) "all";
    snewarg[2] = (char *) "stress/atom";
    snewarg[3] = (char *) "NULL"; //~ No temperature [KH - 11 March 2014]
    snewarg[4] = (char *) "pair";
    snewarg[5] = (char *) "fix";
    snewarg[6] = (char *) "bond";

    modify->add_compute(7,snewarg);
    delete [] snewarg;
  }

  //~ Confirm that the compute exists now
  icompute = modify->find_compute(id_stress);
  if (icompute < 0) error->all(FLERR,"Stress ID does not exist in fix crushing");
  tstress = modify->compute[icompute];

  //~ Also set up a compute coord/gran to identify rattlers
  if (modify->find_compute("crush_coord") < 0) {;
    char **tnewarg = new char*[3];
    tnewarg[0] = (char *) "crush_coord";
    tnewarg[1] = (char *) "all";
    tnewarg[2] = (char *) "coord/gran";
    
    modify->add_compute(3,tnewarg);
    delete [] tnewarg;
    
    //~ Confirm that the compute exists
    icompute = modify->find_compute("crush_coord");
    if (icompute < 0) error->all(FLERR,"Coordination number ID does not exist in fix crushing");
    tcompute = modify->compute[icompute];
  }
}

/* ---------------------------------------------------------------------- */

void FixCrushing::pre_force(int vflag)
{
  if (update->integrate->vflag <= 2) update->integrate->vflag += 4;
}

/* ---------------------------------------------------------------------- */

void FixCrushing::end_of_step()
{
  if (tstress->invoked_peratom != update->ntimestep)
    tstress->compute_peratom(); //~ Update the atom stresses if necessary
  double **stresses = tstress->array_atom; //~ Fetch the stresses

  /*~ Note that tensile stresses are positive in Christensen (2000) (or
    Russell and Muir-Wood (2009)), but compressive stresses are positive
    in the stresses array. Swap the signs around in the calculations
    below to take this into account. All of the entries in the cstrength
    vector are negative (hence the need for the fabs operation below).*/

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int ncols = 10;
  int nrows = 1;
  double i1,i2,j2,kappa,lhs,rhs;
  double oneover3 = 1.0/3.0; //~ To minimise the number of divisions
  double oneoverroot3 = sqrt(oneover3);
  double chiplusonesqd = chiplusone*chiplusone;
  double oneoverchiplusone = 1.0/chiplusone;
  double tempredvolume = 0.0; //~ Reinitialise on each timestep

  /*~ Set up arrays in which to store information to display to the 
    screen or write to the log file. Ensure there is always at least
    1 row in localdata.*/
  double **localdata, **displaydata;
  memory->create(localdata,nrows,ncols,"fix_crushing:localdata");

  //~ Initialise the newly-created row with zeros
  for (int j = 0; j < ncols; j++) localdata[nrows-1][j] = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      //~ Calculate the required invariants for particle i
      i1 = -1.0*(stresses[i][0] + stresses[i][1] + stresses[i][2]);
      i2 = stresses[i][0]*stresses[i][1] + stresses[i][0]*stresses[i][2]
	+ stresses[i][1]*stresses[i][2] - stresses[i][3]*stresses[i][3]
	- stresses[i][4]*stresses[i][4] - stresses[i][5]*stresses[i][5];
      j2 = i1*i1*oneover3 - i2;

      //~ Calculate kappa
      kappa = chiplusone*fabs(cparams[i][0])*oneoverroot3;

      //~ Now calculate the left and right sides of the failure condition
      lhs = (chiplusone-1.0)*kappa*i1*oneoverroot3 + chiplusonesqd*j2;
      rhs = kappa*kappa*oneoverchiplusone;

      //~ If failure has occurred, reduce the particle diameter
      if (lhs >= rhs) {
	nrows++; //~ The number of failures on this proc
	memory->grow(localdata,nrows,ncols,"fix_crushing:localdata");
	for (int j = 0; j < ncols; j++) localdata[nrows-1][j] = 0.0;
	tempredvolume += failure_occurs(i,localdata,nrows);
      }
    }
  }

  int n = ncols*nrows;
  int *recvcounts, *displs;
  recvcounts = new int[nprocs];
  displs = new int[nprocs];
  MPI_Allgather(&n,1,MPI_INT,recvcounts,1,MPI_INT,world);

  displs[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++)
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];

  //~ Find the total number of particle failures
  int totalnrows = 0;
  MPI_Allreduce(&nrows,&totalnrows,1,MPI_INT,MPI_SUM,world);
  
  //~ Accumulate all the localdata information into displaydata
  memory->create(displaydata,totalnrows,ncols,"fix_crushing:displaydata");

  //~ Accumulate the volume reduction of all particles in redvolume
  double redvolume = 0.0;
  MPI_Allreduce(&tempredvolume,&redvolume,1,MPI_DOUBLE,MPI_SUM,world);

  if (redvolume > SMALL) {
    /*~ Use allgatherv to accumulate localdata arrays into displaydata array.
      Only do this if particle failure has occurred.*/

    double *ptr = NULL;
    ptr = localdata[0];
    MPI_Allgatherv(ptr,n,MPI_DOUBLE,displaydata[0],recvcounts,displs,MPI_DOUBLE,world);

    //~ Display optional messages for the user
    if (displaymessages && me == 0) print_optional_info(displaydata,totalnrows);
  }

  delete [] recvcounts;
  delete [] displs;
  memory->destroy(localdata); //~ Destroy allocated memory as soon as possible
  memory->destroy(displaydata); //~ If null, this does nothing
 
  //~ Next fetch the connectivities
  if (tcompute->invoked_peratom != update->ntimestep)
    tcompute->compute_peratom();
  double *connectivities = tcompute->vector_atom;

  if (redvolume > SMALL) {
    cumulredvolume += redvolume; //~ Update the total cumulative volume loss

    /*~ Insert one or more small particles into unoccupied voids
      if the accumulated volume loss is sufficient to permit this
      without adding mass which was not present initially*/
    int nnew = 0;
    int numinserted;
    if (cumulredvolume > volumeparticletoinsert) {      
      while (cumulredvolume > volumeparticletoinsert) {
	cumulredvolume -= volumeparticletoinsert;
	nnew++;
      }
      
      //~ Now not allowing inter-particle overlaps
      numinserted = insert_particles(nnew);

      //~ Correct cumulredvolume if all requested particles not added
      if (numinserted < nnew)
	cumulredvolume += (nnew - numinserted)*volumeparticletoinsert;

      //~ Compute the volume still to be added on each proc
      double perprocvol = cumulredvolume/static_cast<double>(nprocs);

      int countincreases = 0; //~ The numbers of radii increased
      int overallcountincreases;

      /*~ Identify and increase diameter of rattlers with no contacts.
	Do this only if no new particles have been added to the
	system as neighbour lists have not yet been rebuilt (done near
	end of this function).*/
      if (numinserted < nnew && numinserted == 0) {
	for (int i = 0; i < nlocal; i++) {
	  if (connectivities[i] == 0 && perprocvol > 0.0) {
	    perprocvol -= increase_rattler_diameter(i,perprocvol);
	    countincreases++;
	  }
	}
      
	double oldcumulredvolume = cumulredvolume;

	//~ Gather perprocvol values from all procs and assign to cumulredvolume
	MPI_Allreduce(&perprocvol,&cumulredvolume,1,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(&countincreases,&overallcountincreases,1,MPI_INT,MPI_SUM,world);

	if (me == 0)
	  fprintf(screen,"Volume to add: %1.6e | Volume remaining to add: %1.6e | Number of increased radii: %u.\n",oldcumulredvolume,cumulredvolume,overallcountincreases);
      }
    }
    
    /*~ If radii have been reduced, init entire system since
      comm->borders and neighbor->build is done. comm::init 
      needs neighbor::init needs pair::init needs kspace::init, etc*/

    lmp->init();

    // setup domain, communication and neighboring
    // acquire ghosts
    // build neighbor lists

    atom->setup();
    modify->setup_pre_exchange();
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    if (atom->sortfreq > 0) atom->sort();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build();

    /*~ Shrink the domain so that the void ratio is held constant
      if constante == 1*/
    if (constante == 1) {
      totalpvolume += (numinserted*volumeparticletoinsert - redvolume);
      domain->initialvolume = (voidratio+1)*totalpvolume;
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixCrushing::failure_occurs(int i, double **localdata, int nrows)
{
  //~ Check whether the comminution limit has been reached, if present
  if (atom->radius[i] > commlimit) {
    //~ Increment the number of particle failures
    cparams[i][2] += 1.0;

    //~ Reduce the radius of the particle
    double volred = reduce_radius(i,localdata,nrows);
    return volred;
  } else return 0.0;
}

/* ---------------------------------------------------------------------- */

double FixCrushing::reduce_radius(int i, double **localdata, int nrows)
{
  double *radius = atom->radius;
  double oldradius = radius[i];
  double mindist = -1.0; //~ The amount by which to reduce the radius

  if (redtype == 1) {
    //~ Reduce the diameter to just lose contact with neighbouring particles 
    double **x = atom->x;
    int *mask = atom->mask;

    //~ Check for contacts using the neighbor list
    int i2,ii,j,jj,inum,jnum;
    int *ilist,*jlist,*numneigh,**firstneigh;
    double delx,dely,delz,r,rsq,radsum,overlap;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    for (ii = 0; ii < inum; ii++) {
      i2 = ilist[ii];
      jlist = firstneigh[i2];
      jnum = numneigh[i2];

      //~ Check whether i and i2 are the same particle
      //~ If so, loop through its neighbours and break
      if (radius[i] == radius[i2] && x[i][0] == x[i2][0]
	  && x[i][1] == x[i2][1] && x[i][2] == x[i2][2]) {
	for (jj = 0; jj < jnum; jj++) {
	  j = jlist[jj];
	  j &= NEIGHMASK;

	  //~ Special allowances must be made here for periodic boundaries
	  delx = fabs(x[i][0] - x[j][0]);
	  dely = fabs(x[i][1] - x[j][1]);
	  delz = fabs(x[i][2] - x[j][2]);

	  if (delx > 0.5*(domain->boxhi[0]-domain->boxlo[0]))
	    delx -= (domain->boxhi[0]-domain->boxlo[0]);

	  if (dely > 0.5*(domain->boxhi[1]-domain->boxlo[1]))
	    dely -= (domain->boxhi[1]-domain->boxlo[1]);

	  if (delz > 0.5*(domain->boxhi[2]-domain->boxlo[2]))
	    delz -= (domain->boxhi[2]-domain->boxlo[2]);

	  rsq = delx*delx + dely*dely + delz*delz;
	  r = sqrt(rsq);
	  radsum = radius[i] + radius[j];
	
	  if (r < radsum) {
	    overlap = radsum - r;
	    if (overlap > mindist) mindist = overlap;
	  }
	}
	/*~ Include a small tolerance to ensure that contact is lost
	  and the shear displacements are reset to zero*/
	mindist *= 1.0001;

	break;
      }
    }
  } else {
    //~ Reduce the radius by the fixed fraction 'reduction'
    mindist = reduction*radius[i];
  }
  
  /*~ Reduce the radius of particle i by mindist if the comminution
    limit has not been reached.*/
  int comminutionflag = 0;

  if (radius[i]-mindist >= commlimit) {//~ Comminution limit inactive or not reached
    if (radius[i]-mindist <= 0.0)
      error->all(FLERR,"Check the simulation conditions: particles are inside of other particles");
    
    radius[i] -= mindist;
  } else {
    radius[i] = commlimit; //~ Set equal to the comminution limit
    comminutionflag = 1;
  }

  /*~ Change the stored values of uniaxial tensile and compressive
    strength using the radius which has already been reduced*/
  double oldczero = cparams[i][0];
  double oldcone = cparams[i][1];
  change_strengths(i,radius[i]);

  //~ Store any relevant data in the localdata array
  localdata[nrows-1][0] = 1.0; //~ Position 1 == 1 indicates a particle failure; else == 0
  localdata[nrows-1][1] = comminutionflag; //~ Whether comminution limit reached or not
  localdata[nrows-1][2] = atom->tag[i]; //~ The atom tag
  localdata[nrows-1][3] = cparams[i][2]; //~ The (updated) number of failures
  localdata[nrows-1][4] = oldradius;
  localdata[nrows-1][5] = radius[i];
  localdata[nrows-1][6] = oldczero;
  localdata[nrows-1][7] = oldcone;
  localdata[nrows-1][8] = cparams[i][0];
  localdata[nrows-1][9] = cparams[i][1];

  //~ Return the change in solid volume
  double volchange = (4.0*PI/3.0)*(oldradius*oldradius*oldradius-radius[i]*radius[i]*radius[i]);

  return volchange;
}

/* ---------------------------------------------------------------------- */

double FixCrushing::increase_rattler_diameter(int i, double perprocvol)
{
  //~ Increase the diameter to just avoid contacting neighbouring particles 
  double *radius = atom->radius;
  double oldradius = radius[i];
  double maxdist = BIG; //~ The amount by which to increase the radius
  double **x = atom->x;
  int *mask = atom->mask;

  //~ Check for contacts using the neighbor list
  int i2,ii,j,jj,inum,jnum;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double delx,dely,delz,r,rsq,radsum,rincrease;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i2 = ilist[ii];
    jlist = firstneigh[i2];
    jnum = numneigh[i2];

    //~ Check whether i and i2 are the same particle
    //~ If so, loop through its neighbours and break
    if (radius[i] == radius[i2] && x[i][0] == x[i2][0]
	&& x[i][1] == x[i2][1] && x[i][2] == x[i2][2]) {
      for (jj = 0; jj < jnum; jj++) {
	j = jlist[jj];
	j &= NEIGHMASK;

	//~ Special allowances must be made here for periodic boundaries
	delx = fabs(x[i][0] - x[j][0]);
	dely = fabs(x[i][1] - x[j][1]);
	delz = fabs(x[i][2] - x[j][2]);

	if (delx > 0.5*(domain->boxhi[0]-domain->boxlo[0]))
	  delx -= (domain->boxhi[0]-domain->boxlo[0]);

	if (dely > 0.5*(domain->boxhi[1]-domain->boxlo[1]))
	  dely -= (domain->boxhi[1]-domain->boxlo[1]);

	if (delz > 0.5*(domain->boxhi[2]-domain->boxlo[2]))
	  delz -= (domain->boxhi[2]-domain->boxlo[2]);
	
	rsq = delx*delx + dely*dely + delz*delz;
	r = sqrt(rsq);
	radsum = radius[i] + radius[j];
	
	rincrease = r - radsum;
	if (rincrease < maxdist) maxdist = rincrease;
      }
      //~ Include a small tolerance to ensure no contact
      maxdist *= 0.9999;
      
      break;
    }
  }

  /*~ Increase the radius of particle i by maxdist if the added volume
    is less than perprocvol.*/
  double newradius = oldradius + maxdist;
  double volchange = (4.0*PI/3.0)*(newradius*newradius*newradius-oldradius*oldradius*oldradius);

  if (volchange <= perprocvol)
    radius[i] = newradius;
  else
    radius[i] = pow((0.75*perprocvol/PI + oldradius*oldradius*oldradius),(1.0/3.0));

  //~ Change the stored values of uniaxial tensile and compressive strength
  // change_strengths(i,radius[i],1); //~ Send a '1' as an optional parameter

  //~ Return the change in solid volume
  volchange = (4.0*PI/3.0)*(radius[i]*radius[i]*radius[i]-oldradius*oldradius*oldradius);

  return volchange;
}

/* ---------------------------------------------------------------------- */

void FixCrushing::change_strengths(int i, double radius, int a)
{
  int counter = 0;
  int weibullparamflag = 1;
  int counterlimit = 1000;
  double newcstrength = 0.0;

  if (a == 1) newcstrength = -BIG;

  if (a == 0) {
    //~ Force the new compressive strength to be larger in magnitude than the old value
    while (-1.0*newcstrength <= -1.0*cparams[i][0]) {//~ cparams[i][0] is negative
      newcstrength = strength_calculation(i,radius,weibullparamflag);
      
      counter++;
      if (counter > counterlimit) {
	newcstrength = cparams[i][0];
	fprintf(screen,"Loop exit condition invoked in FixCrushing on timestep "BIGINT_FORMAT" to prevent an infinite loop.\n",update->ntimestep);
      
	break; //~ To prevent an infinite loop
      }
    } 
  } else {
    //~ Force the new compressive strength to be smaller in magnitude than the old value
    while (-1.0*newcstrength >= -1.0*cparams[i][0]) {//~ cparams[i][0] is negative
      newcstrength = strength_calculation(i,radius,weibullparamflag);
  
      counter++;
      if (counter > counterlimit) {
	newcstrength = cparams[i][0];
	fprintf(screen,"Loop exit condition invoked in FixCrushing on timestep "BIGINT_FORMAT" to prevent an infinite loop.\n",update->ntimestep);
	
	break; //~ To prevent an infinite loop
      }
    }
  }

  cparams[i][0] = newcstrength;
  cparams[i][1] = -1.0*cparams[i][0]/chiplusone;
}

/* ---------------------------------------------------------------------- */

double FixCrushing::insert_particles(int nnew)
{
  /*~ Function based loosely on pre_exchange function 
    in fix_pour [KH - 25 July 2013]*/

  //~ xmine is for atoms on this proc and ghosts
  int nall = atom->nlocal + atom->nghost;
  int ntotal = nall + nnew;
  double **xmine;
  double **x = atom->x;
  double *radius = atom->radius;

  memory->create(xmine,ntotal,4,"fix_crushing:xmine");
  
  for (int i = 0; i < nall; i++) {
    xmine[i][0] = x[i][0];
    xmine[i][1] = x[i][1];
    xmine[i][2] = x[i][2];
    xmine[i][3] = radius[i];
  }

  // insert new atoms into xmine list, one by one
  // check against all nearby atoms and previously inserted ones
  // if there is an overlap then retry using a new randomly-selected coord
  // else insert by adding to xmine list
 
  /*~ There are two user-defined options to set here. 'maxattempt' 
    attempts are made to place each individual particle without overlapping
    any pre-existing particles, and 'allowoverlaps' is a flag which takes
    the following two values: '0', meaning that if maxattempt is exceeded,
    the particle is not placed and hence volume is not conserved (at least
    until the next particle failure occurs); and '1', meaning that if
    maxattempt is exceeded, the particle is placed anyway in a position
    which minimises the inter-particle overlap.

    If allowoverlaps == 1 and the randomly-selected coordinates which
    minimise the overlap (out of maxattempt trials) has been stored, these
    randomly-selected coordinates are adjusted slightly to find a local 
    minimum of inter-particle overlap.*/ 

  int maxattempt = 1e4; //~ The maximum number of attempts to place an atom
  int allowoverlaps = 0;

  if (maxattempt < 1)
    error->all(FLERR,"maxattempt in fix crushing must be a positive integer");

  if (allowoverlaps != 0 && allowoverlaps != 1)
    error->all(FLERR,"allowoverlaps flag must be either 0 or 1");
  
  int i,oneprocsuccess,success,successproc,sproc,attempt,numinserted,overlapflag,skipflag;
  double coord[3],bestcoord[3],delx,dely,delz,rsq,radsum;
  double overlapsq,minoverlapsq,tempmaxoverlapsq;
  double tempbiggestoverlap;
  double biggestoverlap = 0.0;

  /*~ This part of the code is run on all procs (mostly), but only the local
    list of particle coordinates is parsed. This can potentially cause problems
    because if > 1 particle is placed, the ghosts of the first particle
    are not created until all particles have been placed. Hence an
    overlap between the first particle and subsequent particles may
    not be detected as ghosts are not present. This is avoided by adding the
    coordinates of newly-added particles to the xmine arrays on all procs.*/

  while (nall < ntotal) {
    oneprocsuccess = success = attempt = 0;
    successproc = sproc = -1; //~ There is a proc '0' so initialise at negative value
    minoverlapsq = BIG;

    while (attempt < maxattempt && !success) {
      attempt++;
      overlapflag = 0; //~ '0' indicates that there are no inter-particle overlaps
      skipflag = 0; //~ '1' indicates that part of the computation must be bypassed after a break command
      tempmaxoverlapsq = 0.0;
      xyz_random(coord);

      for (i = 0; i < nall; i++) {
	//~ Special allowances must be made here for periodic boundaries
	delx = fabs(coord[0] - xmine[i][0]);
	dely = fabs(coord[1] - xmine[i][1]);
	delz = fabs(coord[2] - xmine[i][2]);

	if (delx > 0.5*(domain->boxhi[0]-domain->boxlo[0]))
	  delx -= (domain->boxhi[0]-domain->boxlo[0]);

	if (dely > 0.5*(domain->boxhi[1]-domain->boxlo[1]))
	  dely -= (domain->boxhi[1]-domain->boxlo[1]);

	if (delz > 0.5*(domain->boxhi[2]-domain->boxlo[2]))
	  delz -= (domain->boxhi[2]-domain->boxlo[2]);
	
	rsq = delx*delx + dely*dely + delz*delz;
	radsum = radiusparticletoinsert + xmine[i][3];
	overlapsq = radsum*radsum - rsq;

	if (overlapsq > 0.0) {//~ There is an overlap
	  overlapflag = 1;

	  if (allowoverlaps) {
	    if (overlapsq > minoverlapsq) {
	      skipflag = 1;
	      break;
	    } else if (overlapsq > tempmaxoverlapsq)
	      tempmaxoverlapsq = overlapsq;
	  } else break;
	}
      }

      if (!skipflag && tempmaxoverlapsq <= minoverlapsq) {
	minoverlapsq = tempmaxoverlapsq;
	for (int j = 0; j < 3; j++) bestcoord[j] = coord[j];
      }

      if (!overlapflag) {
	oneprocsuccess = 1; //~ Increment 'oneprocsuccess'
	successproc = me;
      }

      /*~ Now find the maximum value across all procs and store in 'success'.
	If a suitable particle position has been found on any proc, the value
	of success will equal 1 on all procs. Also store the proc in sproc.*/
      MPI_Allreduce(&oneprocsuccess,&success,1,MPI_INT,MPI_MAX,world);
      if (success) MPI_Allreduce(&successproc,&sproc,1,MPI_INT,MPI_MAX,world);
    }

    //~ Transfer the successful insertion point (on proc sproc) to all procs
    if (success) MPI_Bcast(&coord[0],3,MPI_DOUBLE,sproc,world);

    /*~ If overlaps are allowed and present in all cases, success will
      still be zero. Find the minimum value of 'minoverlapsq' and the
      corresponding particle coordinates.*/
    if (allowoverlaps && !success) {
      double overallmin = 0.0;
      double tolerance = 1e-10;
      MPI_Allreduce(&minoverlapsq,&overallmin,1,MPI_DOUBLE,MPI_MIN,world);

      //~ Find the corresponding bestcoord values. successproc == -1.
      if (fabs(overallmin-minoverlapsq) < tolerance) successproc = me;
      MPI_Allreduce(&successproc,&sproc,1,MPI_INT,MPI_MAX,world);

      //~ Can run minimise_overlap function on sproc
      if (me == sproc)
	tempbiggestoverlap = minimise_overlap(bestcoord,xmine,nall,radiusparticletoinsert);

      MPI_Bcast(&bestcoord[0],3,MPI_DOUBLE,sproc,world);
      MPI_Bcast(&tempbiggestoverlap,1,MPI_DOUBLE,sproc,world);
      
      if (tempbiggestoverlap > biggestoverlap) biggestoverlap = tempbiggestoverlap;
      for (int j = 0; j < 3; j++) coord[j] = bestcoord[j];
      success = 1;	
    }
    
    if (success) {
      for (int j = 0; j < 3; j++) xmine[nall][j] = coord[j];
      xmine[nall][3] = radiusparticletoinsert;
      nall++;
    } else break;
  }

  numinserted = nall - (atom->nlocal + atom->nghost);
 
  //~ Write a message for the user
  if (me == 0 && displaymessages) {
    if (numinserted < nnew)
      fprintf(screen,"WARNING: Inserted only %u of %u required particles on timestep "BIGINT_FORMAT".\n",numinserted,nnew,update->ntimestep);
    else fprintf(screen,"Inserted all %u required particles on timestep "BIGINT_FORMAT".\n",numinserted,update->ntimestep);

    if (allowoverlaps)
      fprintf(screen,"The largest percentage overlap was %1.2f%%.\n",biggestoverlap);
  }
  
  AtomVec *avec = atom->avec;
  double *sublo = domain->sublo;
  double *subhi = domain->subhi;

  //~ Fetch the standard atom type and density from another atom
  int ntype,m;
  double denstmp,radtmp;
  if (atom->nlocal > 0) {
    ntype = atom->type[0];
    denstmp = 0.75*atom->rmass[0]/(PI*atom->radius[0]*atom->radius[0]*atom->radius[0]);
  } else {
    ntype = 1;
    denstmp = 1000;
    error->warning(FLERR,"An approximate density was used in fix crushing");
  }

  // check if new atom is in my sub-box or above it if I'm highest proc
  // if so, add to my list via create_atom()
  // initialize info about the atom
  // set npartner for new atom to 0 (assume not touching any others)

  nall = atom->nlocal + atom->nghost; //~ Reset nall
  for (i = nall; i < numinserted+nall; i++) {
    for (int j = 0; j < 3; j++) coord[j] = xmine[i][j];
    radtmp = xmine[i][3];

    /*~ When creating the atom, note that all velocities are 0.0 by
      default (in AtomVecSphere->create_atom) and the mask is 1*/

    if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
        coord[1] >= sublo[1] && coord[1] < subhi[1] &&
        coord[2] >= sublo[2] && coord[2] < subhi[2]) {
      //~ If true, atom will be local to this proc
      avec->create_atom(ntype,coord);
      m = atom->nlocal - 1;
      atom->type[m] = ntype;
      atom->radius[m] = radtmp;
      atom->rmass[m] = 4.0*PI/3.0 * radtmp*radtmp*radtmp * denstmp;
      modify->create_attribute(m);
    }
  }

  memory->destroy(xmine); //~ Destroy locally-allocated memory

  // reset global natoms
  // set tag # of new particles beyond all previous atoms
  // if global map exists, reset it now instead of waiting for comm

  if (numinserted > 0) {
    atom->natoms += numinserted;
    if (atom->tag_enable) {
      atom->tag_extend();
      if (atom->map_style) {
        atom->nghost = 0;
        atom->map_init();
        atom->map_set();
      }
    }
  }

  //~ Return the number of atoms that were actually inserted
  return numinserted;
}

/* ---------------------------------------------------------------------- */

void FixCrushing::xyz_random(double *coord)
{
  for (int i = 0; i < 3; i++)
    coord[i] = domain->sublo[i] + random->uniform()*(domain->subhi[i]-domain->sublo[i]);
}

/* ---------------------------------------------------------------------- */

double FixCrushing::minimise_overlap(double *coord, double **xnear, int nnear, double placedradius)
{
  /*~ The purpose of this function is to adjust the position of a 
    placed particle slightly to minimise its overlaps with neighbouring 
    particles. Identifying the position uses a helper function,
    adjust_position, which is called several times to hone in on an
    optimal position to minimise overlaps with neighbours.*/

  /*~ Firstly identify those particles which are close to the 'coord' 
    position. Store these in a coordneighs array.*/

  int numneighs = 0;
  int neighstatus[nnear];
  double delx,dely,delz,rsq,neighdist;

  for (int i = 0; i < nnear; i++)
    neighstatus[i] = 0;

  for (int i = 0; i < nnear; i++) {
    delx = fabs(coord[0] - xnear[i][0]);
    dely = fabs(coord[1] - xnear[i][1]);
    delz = fabs(coord[2] - xnear[i][2]);

    if (delx > 0.5*(domain->boxhi[0]-domain->boxlo[0]))
      delx -= (domain->boxhi[0]-domain->boxlo[0]);

    if (dely > 0.5*(domain->boxhi[1]-domain->boxlo[1]))
      dely -= (domain->boxhi[1]-domain->boxlo[1]);

    if (delz > 0.5*(domain->boxhi[2]-domain->boxlo[2]))
      delz -= (domain->boxhi[2]-domain->boxlo[2]);
	
    rsq = delx*delx + dely*dely + delz*delz;

    //~ Use sum of radii + placed particle radius to decide neighbour or not
    neighdist = 2.0*radiusparticletoinsert + xnear[i][3];

    if (rsq < neighdist*neighdist) {
      numneighs++;
      neighstatus[i]++;
    }
  }

  int counter = 0;
  double **coordneighs;
  memory->create(coordneighs,numneighs,4,"fix_crushing:coordneighs");

  for (int i = 0; i < nnear; i++)
    if (neighstatus[i] == 1) {
      for (int j = 0; j < 4; j++)
	coordneighs[counter][j] = xnear[i][j];
 
      counter++;
    }

  /*~ Now adjust the position of the particle to be placed so as to
    minimise its overlaps with the members of the coordneighs array.
    Try moving the particle by increments equal to a fixed fraction
    of its radius. Careful when setting resolution: the number of
    operations increases cubically with resolution.*/

  int resolution = 10; //~ Number of new points to test in each dimension
  int immediatebreak = 0; //~ Set to 1 if position identified without overlaps
  double maxoverlap = BIG; //~ A nominal maximum percentage overlap

  if (resolution < 1)
    error->all(FLERR,"resolution must be a positive integer in fix crushing");

  double radfrac[5] = {0.15, 0.05, 0.01, 0.004, 0.001}; //~ Fraction of radius by which to move the point

  /*~ Call the adjust_position function multiple times, and on each occasion
    use a smaller value of radfrac so as to find a local minimum overlap
    in the vicinity of the 'coord' position in an efficient way*/
  for (int i = 0; i < 5; i++)
    if (immediatebreak == 0)
      immediatebreak = adjust_position(coord,coordneighs,numneighs,placedradius,radfrac[i],resolution,maxoverlap);

  memory->destroy(coordneighs); //~ Free allocated memory
 
  if (immediatebreak == 1) return 0.0;
  return maxoverlap;
}

/* ---------------------------------------------------------------------- */

int FixCrushing::adjust_position(double *coord, double **coordneighs, int numneighs, double placedradius, double radfrac, double resolution, double &maxoverlap)
{
  /*~ The purpose of this function is to adjust the position of a 
    placed particle slightly to minimise its overlaps with neighbouring 
    particles. maxoverlap is passed by reference to allow its value in
    minimise_overlap to be changed by this function.*/

  if (radfrac <= 0.0 || radfrac >= 1.0)
    error->all(FLERR,"radfrac in fix crushing must be between 0 and 1");

  int immediatebreak = 0;
  double contactingrad,overallcontactingrad;
  double tempcoord[3],bestcoord[3],radsum;
  double overlapsq,tempmaxoverlapsq,delx,dely,delz,rsq;
  double minoverlapsq = BIG; //~ Initialise at a large value

  for (int i = 0; i < resolution+1; i++) {
    tempcoord[0] = coord[0] + placedradius*radfrac*(static_cast<double>(i)/static_cast<double>(resolution) - 0.5);

    //~ Ensure this is inside periodic cell
    if (tempcoord[0] < domain->boxlo[0]) 
      tempcoord[0] += (domain->boxhi[0]-domain->boxlo[0]);
    else if (tempcoord[0] >= domain->boxhi[0]) 
      tempcoord[0] -= (domain->boxhi[0]-domain->boxlo[0]);

    for (int j = 0; j < resolution+1; j++) {
      tempcoord[1] = coord[1] + placedradius*radfrac*(static_cast<double>(j)/static_cast<double>(resolution) - 0.5);

      if (tempcoord[1] < domain->boxlo[1]) 
	tempcoord[1] += (domain->boxhi[1]-domain->boxlo[1]);
      else if (tempcoord[1] >= domain->boxhi[1]) 
	tempcoord[1] -= (domain->boxhi[1]-domain->boxlo[1]);

      for (int k = 0; k < resolution+1; k++) {
	tempcoord[2] = coord[2] + placedradius*radfrac*(static_cast<double>(k)/static_cast<double>(resolution) - 0.5);

	if (tempcoord[2] < domain->boxlo[2]) 
	  tempcoord[2] += (domain->boxhi[2]-domain->boxlo[2]);
	else if (tempcoord[2] >= domain->boxhi[2]) 
	  tempcoord[2] -= (domain->boxhi[2]-domain->boxlo[2]);

	tempmaxoverlapsq = 0.0;

	for (int m = 0; m < numneighs; m++) {
	  delx = fabs(tempcoord[0] - coordneighs[m][0]);
	  dely = fabs(tempcoord[1] - coordneighs[m][1]);
	  delz = fabs(tempcoord[2] - coordneighs[m][2]);

	  if (delx > 0.5*(domain->boxhi[0]-domain->boxlo[0]))
	    delx -= (domain->boxhi[0]-domain->boxlo[0]);

	  if (dely > 0.5*(domain->boxhi[1]-domain->boxlo[1]))
	    dely -= (domain->boxhi[1]-domain->boxlo[1]);

	  if (delz > 0.5*(domain->boxhi[2]-domain->boxlo[2]))
	    delz -= (domain->boxhi[2]-domain->boxlo[2]);
	
	  rsq = delx*delx + dely*dely + delz*delz;
	  radsum = placedradius + coordneighs[m][3];
	  overlapsq = radsum*radsum - rsq; //~ > 0 if overlap
	  if (overlapsq > tempmaxoverlapsq) {
	    tempmaxoverlapsq = overlapsq;
	    contactingrad = coordneighs[m][3];
	    if (overlapsq >= minoverlapsq) break;
	  }
	}

	if (tempmaxoverlapsq < minoverlapsq) {
	  minoverlapsq = tempmaxoverlapsq;
	  for (int m = 0; m < 3; m++) bestcoord[m] = tempcoord[m];
	  overallcontactingrad = contactingrad;

	  if (tempmaxoverlapsq < SMALL) {
	    immediatebreak = 1;
	    break;
	  }
	}
      }
      if (immediatebreak == 1) break;
    }
    if (immediatebreak == 1) break;
  }
  
  //~ At this stage, the optimum coordinates are stored in bestcoord
  for (int j = 0; j < 3; j++) coord[j] = bestcoord[j];

  //~ Update the representative maximum percentage overlap
  radsum = placedradius + overallcontactingrad;
  double d = sqrt(radsum*radsum - minoverlapsq);
  maxoverlap = 50.0*(radsum - d)/placedradius;
  
  return immediatebreak;
}

/* ---------------------------------------------------------------------- */

void FixCrushing::print_optional_info(double **data, int nrows)
{
  for (int i = 0; i < nrows; i++) {
    if (data[i][0] < 0.5) continue;
    else {//~ Equal to 1.0 if particle failure and comminution limit not previously reached
      if (data[i][1] > 0.5) //~ Comminution limit reached
	fprintf(screen,"\nFailure number %u of particle %u on timestep "BIGINT_FORMAT".\n\tRadius changed from %1.3e to %1.3e (comminution limit).\n\tStrengths changed from \t%1.3e (compressive) and %1.3e (tensile)\n\t\t\tto\t%1.3e (compressive) and %1.3e (tensile).\n",static_cast<int> (data[i][3]),static_cast<int> (data[i][2]),update->ntimestep,data[i][4],data[i][5],data[i][6],data[i][7],data[i][8],data[i][9]);
      else
	fprintf(screen,"\nFailure number %u of particle %u on timestep "BIGINT_FORMAT".\n\tRadius changed from %1.3e to %1.3e.\n\tStrengths changed from \t%1.3e (compressive) and %1.3e (tensile)\n\t\t\tto\t%1.3e (compressive) and %1.3e (tensile).\n",static_cast<int> (data[i][3]),static_cast<int> (data[i][2]),update->ntimestep,data[i][4],data[i][5],data[i][6],data[i][7],data[i][8],data[i][9]);
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixCrushing::memory_usage()
{
  double bytes = atom->nmax*3 * sizeof(double); //~ For cparams array
  bytes += (atom->nlocal+atom->nghost)*4 * sizeof(double); //~ For xmine array in insert_particles

  /*~ This is the most memory which is allocated concurrently. Other arrays
    (e.g., displaydata) are created and destroyed in another function,
    and are smaller in size than xnear.*/
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixCrushing::grow_arrays(int nmax)
{
  memory->grow(cparams,nmax,3,"fix_crushing:cparams");
  array_atom = cparams;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixCrushing::copy_arrays(int i, int j, int delflag)
{
  cparams[j][0] = cparams[i][0];
  cparams[j][1] = cparams[i][1];
  cparams[j][2] = cparams[i][2];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixCrushing::set_arrays(int i)
{
  /*~ This is used for setting the parameters for daughter particles.
    Since the size of these is usually the comminution limit, there is
    no point in using weibullparamflag == 1. Use the first set of
    Weibull parameters, in case a larger particle is added to the system.*/
  int weibullparamflag = 0;
  cparams[i][0] = strength_calculation(i,atom->radius[i],weibullparamflag);
  cparams[i][1] = -1.0*cparams[i][0]/chiplusone;
  cparams[i][2] = 0.0; //~ The number of breakage events
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixCrushing::pack_exchange(int i, double *buf)
{
  buf[0] = cparams[i][0];
  buf[1] = cparams[i][1];
  buf[2] = cparams[i][2];
  return 3;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixCrushing::unpack_exchange(int nlocal, double *buf)
{
  cparams[nlocal][0] = buf[0];
  cparams[nlocal][1] = buf[1];
  cparams[nlocal][2] = buf[2];
  return 3;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixCrushing::pack_restart(int i, double *buf)
{
  buf[0] = 4;
  buf[1] = cparams[i][0];
  buf[2] = cparams[i][1];
  buf[3] = cparams[i][2];
  return 4;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixCrushing::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  cparams[nlocal][0] = extra[nlocal][m++];
  cparams[nlocal][1] = extra[nlocal][m++];
  cparams[nlocal][2] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixCrushing::maxsize_restart()
{
  return 4;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixCrushing::size_restart(int nlocal)
{
  return 4;
}

/* ---------------------------------------------------------------------- */

void FixCrushing::write_restart(FILE *fp)
{
  int n = 0;
  double list[1];
  list[n++] = cumulredvolume;

  if (me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ---------------------------------------------------------------------- */

void FixCrushing::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;
  cumulredvolume = static_cast<double> (list[n++]);
}

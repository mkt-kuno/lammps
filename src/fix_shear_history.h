/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(SHEAR_HISTORY,FixShearHistory)

#else

#ifndef LMP_FIX_SHEAR_HISTORY_H
#define LMP_FIX_SHEAR_HISTORY_H

#include "fix.h"
#include "my_page.h"

namespace LAMMPS_NS {

class FixShearHistory : public Fix {
  friend class Neighbor;
  friend class PairGranHookeHistory;
  friend class PairGranShmHistory; //~ Added this [KH - 23 November 2012]
  friend class PairGranCMHistory; //~ Added this [MO - 05 June 2014]
  friend class PairGranHMDHistory; //~ Added this [MO - 21 July 2014]
  friend class PairGranCMDHistory; //~ Added this [MO - 18 November 2014]
  friend class PairLineGranHookeHistory;
  friend class PairTriGranHookeHistory;

 public:
  FixShearHistory(class LAMMPS *, int, char **);
  ~FixShearHistory();
  int setmask();
  void init();
  virtual void pre_exchange();
  void min_pre_exchange();
  void post_run();

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  void set_arrays(int);

  int pack_reverse_comm_size(int, int);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  int pack_restart(int, double *);
  void unpack_restart(int, int);
  int size_restart(int);
  int maxsize_restart();

 protected:
  int newton_pair;              // same as force setting
  int onesided;                 // 1 for line/tri history, else 0
  int nlocal_neigh;             // nlocal at last time neigh list was built
  int nall_neigh;               // ditto for nlocal+nghost

  int *npartner;                // # of touching partners of each atom
  tagint **partner;             // global atom IDs for the partners

  /*~ The number of shear quantities is not necessarily 3, but can
    be several different values. In here, define arrays with differing
    numbers of shear quantities. There are no space implications as
    memory will be allocated for only one of these later on (using new)
    [KH - 9 January 2014]*/
  /*~~ CM, HMD and CMD models have been added where HMD and CMD use
    the same number of shear quantities. [KH - 18 November 2014] ~~*/
  //~ ----------------- without energy tracing ----------------------
  double (**shearpartner3)[3]; //~ hooke/history or hertz/history
  double (**shearpartner4)[4]; //~ shm/history
  double (**shearpartner5)[5]; //~ CM/history [MO - 04 June 2014]
  double (**shearpartner18)[18]; //~ hooke/history or hertz/history with rolling
  double (**shearpartner19)[19]; //~ shm/history with rolling
  double (**shearpartner26)[26]; //~ HMD/history [MO - 14 November 2014]
  double (**shearpartner24)[24]; //~ shm/history with D_spin [MO - 14 November 2014]
  double (**shearpartner25)[25]; //~ CM/history with D_spin  [MO - 14 November 2014]
  double (**shearpartner46)[46]; //~ HMD/history with D_spin [MO - 14 November 2014]
  //~ ----------------- with energy tracing ----------------------
  double (**shearpartner7)[7]; //~ hooke/history or hertz/history
  double (**shearpartner8)[8]; //~ shm/history 
  double (**shearpartner9)[9]; //~ CM/history [MO 04 June 2014]
  double (**shearpartner22)[22]; //~ hooke/history or hertz/history with rolling
  double (**shearpartner23)[23]; //~ shm/history with rolling 
  double (**shearpartner30)[30]; //~ HMD/history [MO - 14 November 2014]
  double (**shearpartner28)[28]; //~ shm/history with D_spin [MO - 14 November 2014]
  double (**shearpartner29)[29]; //~ CM/history with D_spin  [MO - 14 November 2014]
  double (**shearpartner50)[50]; //~ HMD/history with D_spin [MO - 14 November 2014]	

  int num_quants;               // the number of extra quantities for each partner (i.e. contact) modified GM
  int maxtouch;                 // max # of touching partners for my atoms

  int commflag;                 // mode of reverse comm to get ghost info

  class Pair *pair;

  int pgsize,oneatom;           // copy of settings in Neighbor
  MyPage<tagint> *ipage;        // pages of partner atom IDs

  /*~ As before, the number of shear quantities can vary [KH - 9 January 
    2014]*/
  MyPage<double[3]> *dpage3;     // pages of shear history with partners
  MyPage<double[4]> *dpage4;
  MyPage<double[5]> *dpage5;
  MyPage<double[7]> *dpage7;
  MyPage<double[8]> *dpage8;
  MyPage<double[9]> *dpage9;
  MyPage<double[18]> *dpage18;
  MyPage<double[19]> *dpage19;
  MyPage<double[22]> *dpage22;
  MyPage<double[23]> *dpage23;
  // added [MO - 14 November 2014]
  MyPage<double[24]> *dpage24; 
  MyPage<double[25]> *dpage25;
  MyPage<double[26]> *dpage26;	 
  MyPage<double[28]> *dpage28;
  MyPage<double[29]> *dpage29;
  MyPage<double[30]> *dpage30; 
  MyPage<double[46]> *dpage46;
  MyPage<double[50]> *dpage50;

  void pre_exchange_onesided();
  void pre_exchange_newton();
  void pre_exchange_no_newton();
  void allocate_pages();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Pair style granular with history requires atoms have IDs

Atoms in the simulation do not have IDs, so history effects
cannot be tracked by the granular pair potential.

E: Shear history overflow, boost neigh_modify one

There are too many neighbors of a single atom.  Use the neigh_modify
command to increase the max number of neighbors allowed for one atom.
You may also want to boost the page size.

*/

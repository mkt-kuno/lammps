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

#ifdef PAIR_CLASS

PairStyle(gran/CMD/history,PairGranCMDHistory)

#else

#ifndef LMP_PAIR_GRAN_CMD_HISTORY_H
#define LMP_PAIR_GRAN_CMD_HISTORY_H

#include "pair_gran_hooke_history.h"

namespace LAMMPS_NS {

class PairGranCMDHistory : public PairGranHookeHistory {
 public:
  PairGranCMDHistory(class LAMMPS *);
  void compute(int, int);
  void settings(int, char **);
  double single(int, int, int, int, double, double, double, double &);
  void init_style();
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);

 protected:
  double Geq,Poiseq,RMSf,Hp; // Added [MO - 04 April 2015]
  int Model,THETA1;          // Added [MO - 12 Sep 2015]
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/

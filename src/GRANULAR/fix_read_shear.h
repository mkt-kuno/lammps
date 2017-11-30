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

FixStyle(read_shear,FixReadShear)

#else

#ifndef LMP_FIX_READ_SHEAR_H
#define LMP_FIX_READ_SHEAR_H

#include "fix.h"

namespace LAMMPS_NS {

class FixReadShear : public Fix {
  public:
    FixReadShear(class LAMMPS *, int, char **);
    ~FixReadShear();
    int setmask();
    void setup_pre_force(int);

  private:
    int nrows; //~ Number of rows in the sheardata array
    int numshearquants; //~ The number of shear quantities
    double **sheardata;
    class FixNeighHistory *fix_history;
};

}

#endif
#endif

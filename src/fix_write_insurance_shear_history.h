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

FixStyle(write_insurance_shear_history,FixWriteInsuranceShearHistory)

#else

#ifndef LMP_FIX_WRITE_INSURANCE_SHEAR_HISTORY_H
#define LMP_FIX_WRITE_INSURANCE_SHEAR_HISTORY_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWriteInsuranceShearHistory : public Fix {
 public:
    FixWriteInsuranceShearHistory(class LAMMPS *, int, char **);
    ~FixWriteInsuranceShearHistory();
    int setmask();
    void setup(int);

 private:
    class FixNeighHistory *fix_history;
};

}

#endif
#endif

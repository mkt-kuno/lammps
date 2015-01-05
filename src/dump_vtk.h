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

/*~ Dump_vtk is simply a modification of dump_custom which creates
  .vtk files which are perfectly formatted for use in ParaView.
  [KH - 3 May 2012]*/

#ifdef DUMP_CLASS

DumpStyle(vtk,DumpVTK)

#else

#ifndef LMP_DUMP_VTK_H
#define LMP_DUMP_VTK_H

#include "dump.h"

namespace LAMMPS_NS {

class DumpVTK : public Dump {
 public:
  DumpVTK(class LAMMPS *, int, char **);
  virtual ~DumpVTK();

 protected:
  int alreadywritten;        //~ Counter to control writing the headers
  int nevery;                // dump frequency for output
  int iregion;               // -1 if no region, else which region
  char *idregion;            // region ID
  int nthresh;               // # of defined threshholds
  int *thresh_array;         // array to threshhhold on for each nthresh
  int *thresh_op;            // threshhold operation for each nthresh
  double *thresh_value;      // threshhold value for each nthresh

  int *vtype;                // type of each vector (INT, DOUBLE)
  char **vformat;            // format string for each vector element

  /*~ columns was modified from a vector to an array as it is now
    used for a different application [KH - 4 May 2012]*/
  char **columns;             // column labels

  int nchoose;               // # of selected atoms
  int maxlocal;              // size of atom selection and variable arrays
  int *choose;               // local indices of selected atoms
  double *dchoose;           // value for each atom to threshhold against
  int *clist;                // compressed list of indices of selected atoms

  int nfield;                // # of keywords listed by user
  int ioptional;             // index of start of optional args

  int *field2index;          // which compute,fix,variable calcs this field
  int *argindex;             // index into compute,fix scalar_atom,vector_atom
                             // 0 for scalar_atom, 1-N for vector_atom values

  int ncompute;              // # of Compute objects used by dump
  char **id_compute;         // their IDs
  class Compute **compute;   // list of ptrs to the Compute objects

  int nfix;                  // # of Fix objects used by dump
  char **id_fix;             // their IDs
  class Fix **fix;           // list of ptrs to the Fix objects

  int nvariable;             // # of Variables used by dump
  char **id_variable;        // their names
  int *variable;             // list of indices for the Variables
  double **vbuf;             // local storage for variable evaluation

  int ncustom;               // # of custom atom properties
  char **id_custom;          // their names
  int *flag_custom;          // their data type

  int ntypes;                // # of atom types
  char **typenames;             // array of element names for each type

  // private methods

  virtual void init_style();
  virtual void write_header(bigint);
  int count();
  void pack(tagint *);
  virtual void write_data(int, double *);
  bigint memory_usage();

  int parse_fields(int, char **);
  int add_compute(char *);
  int add_fix(char *);
  int add_variable(char *);
  int add_custom(char *, int);
  virtual int modify_param(int, char **);

  //~ Add three extra functions
  void write_diameter(int, double *);
  void write_extra(int, double *, int);
  void write_footer();

  typedef void (DumpVTK::*FnPtrHeader)(bigint);
  FnPtrHeader header_choice;           // ptr to write header functions
  void header_item(bigint);

  typedef void (DumpVTK::*FnPtrData)(int, double *);
  FnPtrData write_choice;              // ptr to write data functions
  void write_text(int, double *);

  //~ Added 2 extra writing functions and a type definition
  typedef void (DumpVTK::*FnPtrSupp)(int, double *, int);
  FnPtrSupp write_additional;
  void write_diam(int, double *);
  void write_supp(int, double *, int);

  // customize by adding a method prototype

  typedef void (DumpVTK::*FnPtrPack)(int);
  FnPtrPack *pack_choice;              // ptrs to pack functions

  void pack_compute(int);
  void pack_fix(int);
  void pack_variable(int);
  void pack_custom(int);

  void pack_id(int);
  void pack_molecule(int);
  void pack_proc(int);
  void pack_procp1(int);
  void pack_type(int);
  void pack_mass(int);

  void pack_x(int);
  void pack_y(int);
  void pack_z(int);
  void pack_xs(int);
  void pack_ys(int);
  void pack_zs(int);
  void pack_xs_triclinic(int);
  void pack_ys_triclinic(int);
  void pack_zs_triclinic(int);
  void pack_xu(int);
  void pack_yu(int);
  void pack_zu(int);
  void pack_xu_triclinic(int);
  void pack_yu_triclinic(int);
  void pack_zu_triclinic(int);
  void pack_xsu(int);
  void pack_ysu(int);
  void pack_zsu(int);
  void pack_xsu_triclinic(int);
  void pack_ysu_triclinic(int);
  void pack_zsu_triclinic(int);
  void pack_ix(int);
  void pack_iy(int);
  void pack_iz(int);

  void pack_vx(int);
  void pack_vy(int);
  void pack_vz(int);
  void pack_fx(int);
  void pack_fy(int);
  void pack_fz(int);
  void pack_q(int);
  void pack_mux(int);
  void pack_muy(int);
  void pack_muz(int);
  void pack_mu(int);
  void pack_radius(int);
  void pack_diameter(int);

  void pack_omegax(int);
  void pack_omegay(int);
  void pack_omegaz(int);
  void pack_angmomx(int);
  void pack_angmomy(int);
  void pack_angmomz(int);
  void pack_tqx(int);
  void pack_tqy(int);
  void pack_tqz(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Insufficient dump vtk arguments supplied

Self-explantory.

E: Invalid attribute in dump vtk command

Self-explantory.

E: Could not find dump vtk compute ID

The compute ID needed by dump vtk to compute a per-atom quantity
does not exist.

E: Could not find dump vtk fix ID

Self-explanatory.

E: Dump vtk and fix not computed at compatible times

The fix must produce per-atom quantities on timesteps that dump vtk
needs them.

E: Could not find dump vtk variable name

Self-explanatory.

E: Region ID for dump vtk does not exist

Self-explanatory.

E: Threshhold for an atom property that isn't allocated

A dump threshhold has been requested on a quantity that is
not defined by the atom style used in this simulation.

E: Dumping an atom property that isn't allocated

The chosen atom style does not define the per-atom quantity being
dumped.

E: Dumping an atom quantity that isn't allocated

Only per-atom quantities that are defined for the atom style being
used are allowed.

E: Dump vtk compute does not compute per-atom info

Self-explanatory.

E: Dump vtk compute does not calculate per-atom vector

Self-explanatory.

E: Dump vtk compute does not calculate per-atom array

Self-explanatory.

E: Dump vtk compute vector is accessed out-of-range

Self-explanatory.

E: Dump vtk fix does not compute per-atom info

Self-explanatory.

E: Dump vtk fix does not compute per-atom vector

Self-explanatory.

E: Dump vtk fix does not compute per-atom array

Self-explanatory.

E: Dump vtk fix vector is accessed out-of-range

Self-explanatory.

E: Dump vtk variable is not atom-style variable

Only atom-style variables generate per-atom quantities, needed for
dump output.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Dump_modify region ID does not exist

Self-explanatory.

E: Dump modify element names do not match atom types

Number of element names must equal number of atom types.

E: Invalid attribute in dump modify command

Self-explantory.

E: Could not find dump modify compute ID

Self-explanatory.

E: Dump modify compute ID does not compute per-atom info

Self-explanatory.

E: Dump modify compute ID does not compute per-atom vector

Self-explanatory.

E: Dump modify compute ID does not compute per-atom array

Self-explanatory.

E: Dump modify compute ID vector is not large enough

Self-explanatory.

E: Could not find dump modify fix ID

Self-explanatory.

E: Dump modify fix ID does not compute per-atom info

Self-explanatory.

E: Dump modify fix ID does not compute per-atom vector

Self-explanatory.

E: Dump modify fix ID does not compute per-atom array

Self-explanatory.

E: Dump modify fix ID vector is not large enough

Self-explanatory.

E: Could not find dump modify variable name

Self-explanatory.

E: Dump modify variable is not atom-style variable

Self-explanatory.

E: Invalid dump_modify threshhold operator

Operator keyword used for threshold specification in not recognized.

*/

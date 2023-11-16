﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FemModule.cc                                                (C) 2022-2023 */
/*                                                                           */
/* Simple module to test simple FEM mechanism.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/utils/NumArray.h>
#include <arcane/utils/CommandLineArguments.h>
#include <arcane/utils/StringList.h>

#include <arcane/ITimeLoopMng.h>
#include <arcane/IMesh.h>
#include <arcane/IItemFamily.h>
#include <arcane/ItemGroup.h>
#include <arcane/ICaseMng.h>

#if defined(USE_COO) || defined(USE_COO_GPU)
#include "CooFormatMatrix.h"
#endif

#ifdef USE_CSR
#include "CsrFormatMatrix.h"
#endif

#include "IDoFLinearSystemFactory.h"
#include "Fem_axl.h"
#include "FemUtils.h"
#include "DoFLinearSystem.h"
#include "FemDoFsOnNodes.h"

#ifdef REGISTER_TIME
#include <iostream>
#include <fstream>
#include <chrono>
#endif

#ifdef USE_COO
#include "arcane/core/IIndexedIncrementalItemConnectivityMng.h"
#include "arcane/core/IIndexedIncrementalItemConnectivity.h"
#endif

#if defined(USE_CUSPARSE_ADD)
//include for cusparse
#include <cusparse_v2.h>
#endif

//include for GPU use
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/core/RunQueue.h"

//include for connectivity view
#include "arcane/UnstructuredMeshConnectivity.h"

// Fichier à inclure pour avoir RUNCOMMAND_ENUMERATE
#include "arcane/accelerator/RunCommandEnumerate.h"

// Fichier à inclure pour avoir RUNCOMMAND_LOOP
#include "arcane/accelerator/RunCommandLoop.h"
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::FemUtils;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(USE_CUSPARSE_ADD)
/**
 * @brief Macro for use of cusparse
 * 
 */
#define CHECK_CUSPARSE(func) \
  { \
    cusparseStatus_t status = (func); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
             __LINE__, cusparseGetErrorString(status), status); \
      return; \
    } \
  }

#define CHECK_CUDA(func) \
  { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
      printf("CUDA API failed at line %d with error: %s (%d)\n", \
             __LINE__, cudaGetErrorString(status), status); \
      return; \
    } \
  }

/**
 * @brief struct for the csr of cusparse 
 * 
 */
struct cusparseCsr
{
  cusparseMatDescr_t desc;
  Int32 nnz = 0;
  Int32* csrRow;
  Int32* csrCol;
  float* csrVal;
};

#ifdef REGISTER_TIME
struct computeTimer
{
  double add_glob = 0;
  double compute_el = 0;
  double sort_coo = 0;
  double convert_coo = 0;
  double convert_coo_tot = 0;
  double convert_csr_tot = 0;
  double convert_tot = 0;
  double iter_time = 0;
  double compute_tot = 0;
};
#endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module Fem.
 */
class FemModule
: public ArcaneFemObject
{
 public:

  explicit FemModule(const ModuleBuildInfo& mbi)
  : ArcaneFemObject(mbi)
  , m_dofs_on_nodes(mbi.subDomain()->traceMng())
#if defined(USE_COO) || defined(USE_COO_GPU) || defined(USE_CSR_CUSPARSE)
  , m_coo_matrix(mbi.subDomain())
#endif
#ifdef USE_CSR
  , m_csr_matrix(mbi.subDomain())
#endif
  {
    ICaseMng* cm = mbi.subDomain()->caseMng();
    cm->setTreatWarningAsError(true);
    cm->setAllowUnkownRootElelement(false);
  }

 public:

  //! Method called at each iteration
  void compute() override;

  //! Method called at the beginning of the simulation
  void startInit() override;

  VersionInfo versionInfo() const override
  {
    return VersionInfo(1, 0, 0);
  }

 private:

  Real f;
  Real ElementNodes;

  DoFLinearSystem m_linear_system;
  IItemFamily* m_dof_family = nullptr;
  FemDoFsOnNodes m_dofs_on_nodes;

#if defined(USE_COO) || defined(USE_COO_GPU)
  CooFormat m_coo_matrix;
#endif

#ifdef USE_CSR
  CsrFormat m_csr_matrix;
#endif

#if defined(REGISTER_TIME)
  ofstream logger;
  double lhs_time;
  double rhs_time;
  double solver_time;
#endif

 private:

  void _doStationarySolve();
  void _getMaterialParameters();
  void _updateBoundayConditions();
  void _assembleBilinearOperatorTRIA3();
  void _assembleBilinearOperatorQUAD4();
  void _solve();
  void _initBoundaryconditions();
  void _assembleLinearOperator();
  void _applyDirichletBoundaryConditions();
  void _checkResultFile();
  FixedMatrix<3, 3> _computeElementMatrixTRIA3(Cell cell);
  FixedMatrix<4, 4> _computeElementMatrixQUAD4(Cell cell);
  Real _computeAreaTriangle3(Cell cell);
  Real _computeAreaQuad4(Cell cell);
  Real _computeEdgeLength2(Face face);
  Real2 _computeEdgeNormal2(Face face);
#ifdef USE_CUSPARSE_ADD
  void printCsrMatrix(std::string fileName, cusparseCsr csr, bool is_coo);
  void _computeCusparseElementMatrix(cusparseCsr& result, cusparseCsr& global, Cell icell, cusparseHandle_t handle, IndexedNodeDoFConnectivityView node_dof
#ifdef REGISTER_TIME
                                     ,
                                     computeTimer& timer
#endif
  );
  void _assembleCusparseBilinearOperatorTRIA3();
#endif
#ifdef USE_COO
  void _buildMatrix();
  void _assembleCooBilinearOperatorTRIA3();
#endif
#ifdef USE_COO_GPU
  void _buildMatrixGPU();
  NumArray<Real, ExtentsV<3, 3>>
  _computeElementMatrixTRIA3GPU(CellLocalId icell, IndexedCellNodeConnectivityView cnc, VariableNodeReal3InView in_node_coord);
  void _assembleCooGPUBilinearOperatorTRIA3();
#endif
#ifdef USE_CSR
  void _assembleCsrBilinearOperatorTRIA3();
  void _buildMatrixCsr();
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
compute()
{
  info() << "Module Fem COMPUTE";

  // Stop code after computations
  if (m_global_iteration() > 0)
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  m_linear_system.reset();
  m_linear_system.setLinearSystemFactory(options()->linearSystem());

  m_linear_system.initialize(subDomain(), m_dofs_on_nodes.dofFamily(), "Solver");
  // Test for adding parameters for PETSc.
  // This is only used for the first call.
  {
    StringList string_list;
    string_list.add("-trmalloc");
    string_list.add("-log_trace");
    CommandLineArguments args(string_list);
    m_linear_system.setSolverCommandLineArguments(args);
  }
  info() << "NB_CELL=" << allCells().size() << " NB_FACE=" << allFaces().size();

  _doStationarySolve();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
startInit()
{
  info() << "Module Fem INIT";

#ifdef REGISTER_TIME
  logger = ofstream("timer.txt");
#endif

  m_dofs_on_nodes.initialize(mesh(), 1);
  m_dof_family = m_dofs_on_nodes.dofFamily();

  //_buildDoFOnNodes();
  //Int32 nb_node = allNodes().size();
  //m_k_matrix.resize(nb_node, nb_node);
  //m_k_matrix.fill(0.0);

  //m_rhs_vector.resize(nb_node);
  //m_rhs_vector.fill(0.0);

  // # init mesh
  // # init behavior
  // # init behavior on mesh entities
  // # init BCs
  _initBoundaryconditions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_doStationarySolve()
{

#ifdef REGISTER_TIME
  auto fem_start = std::chrono::high_resolution_clock::now();
#endif

  // # get material parameters
  _getMaterialParameters();

  // # update BCs
  _updateBoundayConditions();

  // Assemble the FEM bilinear operator (LHS - matrix A)
  if (options()->meshType == "QUAD4")
    _assembleBilinearOperatorQUAD4();
  else {

#ifdef USE_CUSPARSE_ADD
    _assembleCusparseBilinearOperatorTRIA3();
#endif
#ifdef USE_COO
    _assembleCooBilinearOperatorTRIA3();
#endif
#ifdef USE_LEGACY
    _assembleBilinearOperatorTRIA3();
#endif
#ifdef USE_COO_GPU
    _assembleCooGPUBilinearOperatorTRIA3();
#endif
#ifdef USE_CSR
    _assembleCsrBilinearOperatorTRIA3();
#endif
  }

  // Assemble the FEM linear operator (RHS - vector b)
  //  _assembleLinearOperator();

  // # T=linalg.solve(K,RHS)
  //_solve();

  // Check results
  //_checkResultFile();

#ifdef REGISTER_TIME
  auto fem_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fem_duration = fem_stop - fem_start;
  double total_duration = fem_duration.count();
  logger << "FEM total duration : " << fem_duration.count() << "\n"
         << "LHS time in total duration : " << lhs_time / total_duration * 100 << "%\n"
         << "RHS time in total duration : " << rhs_time / total_duration * 100 << "%\n"
         << "Solver time in total duration : " << solver_time / total_duration * 100 << "%\n";

  logger.close();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_getMaterialParameters()
{
  info() << "Get material parameters...";
  f = options()->f();
  ElementNodes = 3.;

  if (options()->meshType == "QUAD4")
    ElementNodes = 4.;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_initBoundaryconditions()
{
  info() << "Init boundary conditions...";

  info() << "Apply boundary conditions";
  _applyDirichletBoundaryConditions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_applyDirichletBoundaryConditions()
{
  // Handle all the Dirichlet boundary conditions.
  // In the 'arc' file, there are in the following format:
  //   <dirichlet-boundary-condition>
  //   <surface>Haut</surface>
  //   <value>21.0</value>
  // </dirichlet-boundary-condition>

  for (const auto& bs : options()->dirichletBoundaryCondition()) {
    FaceGroup group = bs->surface();
    Real value = bs->value();
    info() << "Apply Dirichlet boundary condition surface=" << group.name() << " v=" << value;
    ENUMERATE_ (Face, iface, group) {
      for (Node node : iface->nodes()) {
        m_u[node] = value;
        m_u_dirichlet[node] = true;
      }
    }
  }

  for (const auto& bs : options()->dirichletPointCondition()) {
    NodeGroup group = bs->node();
    Real value = bs->value();
    info() << "Apply Dirichlet point condition node=" << group.name() << " v=" << value;
    ENUMERATE_ (Node, inode, group) {
      Node node = *inode;
      m_u[node] = value;
      m_u_dirichlet[node] = true;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_updateBoundayConditions()
{
  info() << "TODO " << A_FUNCINFO;
}

/*---------------------------------------------------------------------------*/
// Assemble the FEM linear operator
//  - This function enforces a Dirichlet boundary condition in a weak sense
//    via the penalty method
//  - The method also adds source term
//  - TODO: external fluxes
/*---------------------------------------------------------------------------*/

void FemModule::
_assembleLinearOperator()
{
  info() << "Assembly of FEM linear operator ";
  info() << "Applying Dirichlet boundary condition via  penalty method ";

#ifdef REGISTER_TIME
  auto rhs_start = std::chrono::high_resolution_clock::now();
  double penalty_time;
  double wpenalty_time;
  double sassembly_time;
  double fassembly_time;
#endif

  // Temporary variable to keep values for the RHS part of the linear system
  VariableDoFReal& rhs_values(m_linear_system.rhsVariable());
  rhs_values.fill(0.0);

  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  if (options()->enforceDirichletMethod() == "Penalty") {

    //----------------------------------------------
    // penalty method to enforce Dirichlet BC
    //----------------------------------------------
    //  Let 'P' be the penalty term and let 'i' be the set of DOF for which
    //  Dirichlet condition needs to be applied
    //
    //  - For LHS matrix A the diag term corresponding to the Dirichlet DOF
    //           a_{i,i} = 1. * P
    //
    //  - For RHS vector b the term that corresponds to the Dirichlet DOF
    //           b_{i} = b_{i} * P
    //----------------------------------------------

    info() << "Applying Dirichlet boundary condition via "
           << options()->enforceDirichletMethod() << " method ";

    Real Penalty = options()->penalty(); // 1.0e30 is the default

#ifdef REGISTER_TIME
    auto penalty_start = std::chrono::high_resolution_clock::now();
#endif

    ENUMERATE_ (Node, inode, ownNodes()) {
      NodeLocalId node_id = *inode;
      if (m_u_dirichlet[node_id]) {
        DoFLocalId dof_id = node_dof.dofId(*inode, 0);
        m_linear_system.matrixSetValue(dof_id, dof_id, Penalty);
        Real u_g = Penalty * m_u[node_id];
        rhs_values[dof_id] = u_g;
      }
    }

#ifdef REGISTER_TIME
    auto penalty_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> penalty_duration = penalty_stop - penalty_start;
    penalty_time = penalty_duration.count();
    logger << "Penalty duration : " << penalty_time << "\n";
#endif
  }
  else if (options()->enforceDirichletMethod() == "WeakPenalty") {

    //----------------------------------------------
    // weak penalty method to enforce Dirichlet BC
    //----------------------------------------------
    //  Let 'P' be the penalty term and let 'i' be the set of DOF for which
    //  Dirichlet condition needs to be applied
    //
    //  - For LHS matrix A the diag term corresponding to the Dirichlet DOF
    //           a_{i,i} = a_{i,i} + P
    //
    //  - For RHS vector b the term that corresponds to the Dirichlet DOF
    //           b_{i} = b_{i} * P
    //----------------------------------------------

    info() << "Applying Dirichlet boundary condition via "
           << options()->enforceDirichletMethod() << " method ";

    Real Penalty = options()->penalty(); // 1.0e30 is the default

#ifdef REGISTER_TIME
    auto wpenalty_start = std::chrono::high_resolution_clock::now();
#endif

    ENUMERATE_ (Node, inode, ownNodes()) {
      NodeLocalId node_id = *inode;
      if (m_u_dirichlet[node_id]) {
        DoFLocalId dof_id = node_dof.dofId(*inode, 0);
        m_linear_system.matrixAddValue(dof_id, dof_id, Penalty);
        Real u_g = Penalty * m_u[node_id];
        rhs_values[dof_id] = u_g;
      }
    }
#ifdef REGISTER_TIME
    auto wpenalty_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wpenalty_duration = wpenalty_stop - wpenalty_start;
    wpenalty_time = wpenalty_duration.count();
    logger << "Weak Penalty duration : " << wpenalty_time << "\n";
#endif
  }
  else if (options()->enforceDirichletMethod() == "RowElimination") {

    //----------------------------------------------
    // Row elimination method to enforce Dirichlet BC
    //----------------------------------------------
    //  Let 'I' be the set of DOF for which  Dirichlet condition needs to be applied
    //
    //  to apply the Dirichlet on 'i'th DOF
    //  - For LHS matrix A the row terms corresponding to the Dirichlet DOF
    //           a_{i,j} = 0.  : i!=j
    //           a_{i,j} = 1.  : i==j
    //----------------------------------------------

    info() << "Applying Dirichlet boundary condition via "
           << options()->enforceDirichletMethod() << " method ";

    // TODO
  }
  else if (options()->enforceDirichletMethod() == "RowColumnElimination") {

    //----------------------------------------------
    // Row elimination method to enforce Dirichlet BC
    //----------------------------------------------
    //  Let 'I' be the set of DOF for which  Dirichlet condition needs to be applied
    //
    //  to apply the Dirichlet on 'i'th DOF
    //  - For LHS matrix A the row terms corresponding to the Dirichlet DOF
    //           a_{i,j} = 0.  : i!=j  for all j
    //           a_{i,j} = 1.  : i==j
    //    also the column terms corresponding to the Dirichlet DOF
    //           a_{i,j} = 0.  : i!=j  for all i
    //----------------------------------------------

    info() << "Applying Dirichlet boundary condition via "
           << options()->enforceDirichletMethod() << " method ";

    // TODO
  }
  else {

    info() << "Applying Dirichlet boundary condition via "
           << options()->enforceDirichletMethod() << " is not supported \n"
           << "enforce-Dirichlet-method only supports:\n"
           << "  - Penalty\n"
           << "  - WeakPenalty\n"
           << "  - RowElimination\n"
           << "  - RowColumnElimination\n";
  }

#ifdef REGISTER_TIME
  auto sassemby_start = std::chrono::high_resolution_clock::now();
#endif

  //----------------------------------------------
  // Constant source term assembly
  //----------------------------------------------
  //
  //  $int_{Omega}(f*v^h)$
  //  only for noded that are non-Dirichlet
  //----------------------------------------------
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Real area = _computeAreaTriangle3(cell);
    for (Node node : cell.nodes()) {
      if (!(m_u_dirichlet[node]) && node.isOwn())
        rhs_values[node_dof.dofId(node, 0)] += f * area / ElementNodes;
    }
  }
#ifdef REGISTER_TIME
  auto sassemby_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sassembly_duration = sassemby_stop - sassemby_start;
  sassembly_time = sassembly_duration.count();
  logger << "Constant source term assembly duration : " << sassembly_time << "\n";
  auto fassemby_start = std::chrono::high_resolution_clock::now();
#endif

  //----------------------------------------------
  // Constant flux term assembly
  //----------------------------------------------
  //
  //  only for noded that are non-Dirichlet
  //  $int_{dOmega_N}((q.n)*v^h)$
  // or
  //  $int_{dOmega_N}((n_x*q_x + n_y*q_y)*v^h)$
  //----------------------------------------------
  for (const auto& bs : options()->neumannBoundaryCondition()) {
    FaceGroup group = bs->surface();

    if (bs->value.isPresent()) {
      Real value = bs->value();
      ENUMERATE_ (Face, iface, group) {
        Face face = *iface;
        Real length = _computeEdgeLength2(face);
        for (Node node : iface->nodes()) {
          if (!(m_u_dirichlet[node]) && node.isOwn())
            rhs_values[node_dof.dofId(node, 0)] += value * length / 2.;
        }
      }
      continue;
    }

    if (bs->valueX.isPresent() && bs->valueY.isPresent()) {
      Real valueX = bs->valueX();
      Real valueY = bs->valueY();
      ENUMERATE_ (Face, iface, group) {
        Face face = *iface;
        Real length = _computeEdgeLength2(face);
        Real2 Normal = _computeEdgeNormal2(face);
        for (Node node : iface->nodes()) {
          if (!(m_u_dirichlet[node]) && node.isOwn())
            rhs_values[node_dof.dofId(node, 0)] += (Normal.x * valueX + Normal.y * valueY) * length / 2.;
        }
      }
      continue;
    }

    if (bs->valueX.isPresent()) {
      Real valueX = bs->valueX();
      ENUMERATE_ (Face, iface, group) {
        Face face = *iface;
        Real length = _computeEdgeLength2(face);
        Real2 Normal = _computeEdgeNormal2(face);
        for (Node node : iface->nodes()) {
          if (!(m_u_dirichlet[node]) && node.isOwn())
            rhs_values[node_dof.dofId(node, 0)] += (Normal.x * valueX) * length / 2.;
        }
      }
      continue;
    }

    if (bs->valueY.isPresent()) {
      Real valueY = bs->valueY();
      ENUMERATE_ (Face, iface, group) {
        Face face = *iface;
        Real length = _computeEdgeLength2(face);
        Real2 Normal = _computeEdgeNormal2(face);
        for (Node node : iface->nodes()) {
          if (!(m_u_dirichlet[node]) && node.isOwn())
            rhs_values[node_dof.dofId(node, 0)] += (Normal.y * valueY) * length / 2.;
        }
      }
      continue;
    }
  }
#ifdef REGISTER_TIME
  auto fassemby_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fassembly_duration = fassemby_stop - fassemby_start;
  fassembly_time = fassembly_duration.count();
  logger << "Constant flux term assembly duration : " << fassembly_time << "\n";
  auto rhs_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = rhs_end - rhs_start;
  rhs_time = duration.count();
  logger << "RHS total duration : " << duration.count() << "\n";
  if (penalty_time != 0)
    logger << "Penalty time in rhs : " << penalty_time / rhs_time * 100 << "%\n";
  else
    logger << "Weak Penalty time in rhs : " << wpenalty_time / rhs_time * 100 << "%\n";
  logger << "Constant source term assembly time in rhs : " << sassembly_time / rhs_time * 100 << "%\n"
         << "Constant flux term assembly time in rhs : " << fassembly_time / rhs_time * 100 << "%\n\n"
         << "-------------------------------------------------------------------------------------\n\n";
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real FemModule::
_computeAreaQuad4(Cell cell)
{
  Real3 m0 = m_node_coord[cell.nodeId(0)];
  Real3 m1 = m_node_coord[cell.nodeId(1)];
  Real3 m2 = m_node_coord[cell.nodeId(2)];
  Real3 m3 = m_node_coord[cell.nodeId(3)];
  return 0.5 * ((m1.x * m2.y + m2.x * m3.y + m3.x * m0.y + m0.x * m1.y) - (m2.x * m1.y + m3.x * m2.y + m0.x * m3.y + m1.x * m0.y));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real FemModule::
_computeAreaTriangle3(Cell cell)
{
  Real3 m0 = m_node_coord[cell.nodeId(0)];
  Real3 m1 = m_node_coord[cell.nodeId(1)];
  Real3 m2 = m_node_coord[cell.nodeId(2)];
  return 0.5 * ((m1.x - m0.x) * (m2.y - m0.y) - (m2.x - m0.x) * (m1.y - m0.y));
}

/*---------------------------------------------------------------------------*/
/*----------------------------#endif-----------------------------------------------*/

Real FemModule::
_computeEdgeLength2(Face face)
{
  Real3 m0 = m_node_coord[face.nodeId(0)];
  Real3 m1 = m_node_coord[face.nodeId(1)];
  return math::sqrt((m1.x - m0.x) * (m1.x - m0.x) + (m1.y - m0.y) * (m1.y - m0.y));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real2 FemModule::
_computeEdgeNormal2(Face face)
{
  Real3 m0 = m_node_coord[face.nodeId(0)];
  Real3 m1 = m_node_coord[face.nodeId(1)];
  if (!face.isSubDomainBoundaryOutside())
    std::swap(m0, m1);
  Real2 N;
  Real norm_N = math::sqrt((m1.y - m0.y) * (m1.y - m0.y) + (m1.x - m0.x) * (m1.x - m0.x)); // for normalizing
  N.x = (m1.y - m0.y) / norm_N;
  N.y = (m0.x - m1.x) / norm_N;
  return N;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FixedMatrix<3, 3> FemModule::
_computeElementMatrixTRIA3(Cell cell)
{
  // Get coordiantes of the triangle element  TRI3
  //------------------------------------------------
  //                  0 o
  //                   . .
  //                  .   .
  //                 .     .
  //              1 o . . . o 2
  //------------------------------------------------
  Real3 m0 = m_node_coord[cell.nodeId(0)];
  Real3 m1 = m_node_coord[cell.nodeId(1)];
  Real3 m2 = m_node_coord[cell.nodeId(2)];

  Real area = _computeAreaTriangle3(cell); // calculate area

  Real2 dPhi0(m1.y - m2.y, m2.x - m1.x);
  Real2 dPhi1(m2.y - m0.y, m0.x - m2.x);
  Real2 dPhi2(m0.y - m1.y, m1.x - m0.x);

  FixedMatrix<2, 3> b_matrix;
  b_matrix(0, 0) = dPhi0.x;
  b_matrix(0, 1) = dPhi1.x;
  b_matrix(0, 2) = dPhi2.x;

  b_matrix(1, 0) = dPhi0.y;
  b_matrix(1, 1) = dPhi1.y;
  b_matrix(1, 2) = dPhi2.y;

  b_matrix.multInPlace(1.0 / (2.0 * area));

  FixedMatrix<3, 3> int_cdPi_dPj = matrixMultiplication(matrixTranspose(b_matrix), b_matrix);
  int_cdPi_dPj.multInPlace(area);

  //info() << "Cell=" << cell.localId();
  //std::cout << " int_cdPi_dPj=";
  //int_cdPi_dPj.dump(std::cout);
  //std::cout << "\n";

  return int_cdPi_dPj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FixedMatrix<4, 4> FemModule::
_computeElementMatrixQUAD4(Cell cell)
{
  // Get coordiantes of the quadrangular element  QUAD4
  //------------------------------------------------
  //             1 o . . . . o 0
  //               .         .
  //               .         .
  //               .         .
  //             2 o . . . . o 3
  //------------------------------------------------
  Real3 m0 = m_node_coord[cell.nodeId(0)];
  Real3 m1 = m_node_coord[cell.nodeId(1)];
  Real3 m2 = m_node_coord[cell.nodeId(2)];
  Real3 m3 = m_node_coord[cell.nodeId(3)];

  Real area = _computeAreaQuad4(cell); // calculate area

  Real2 dPhi0(m2.y - m3.y, m3.x - m2.x);
  Real2 dPhi1(m3.y - m0.y, m0.x - m3.x);
  Real2 dPhi2(m0.y - m1.y, m1.x - m0.x);
  Real2 dPhi3(m1.y - m2.y, m2.x - m1.x);

  FixedMatrix<2, 4> b_matrix;
  b_matrix(0, 0) = dPhi0.x;
  b_matrix(0, 1) = dPhi1.x;
  b_matrix(0, 2) = dPhi2.x;
  b_matrix(0, 3) = dPhi3.x;

  b_matrix(1, 0) = dPhi0.y;
  b_matrix(1, 1) = dPhi1.y;
  b_matrix(1, 2) = dPhi2.y;
  b_matrix(1, 3) = dPhi3.y;

  b_matrix.multInPlace(1.0 / (2.0 * area));

  FixedMatrix<4, 4> int_cdPi_dPj = matrixMultiplication(matrixTranspose(b_matrix), b_matrix);
  int_cdPi_dPj.multInPlace(area);

  //info() << "Cell=" << cell.localId();
  //std::cout << " int_cdPi_dPj=";
  //int_cdPi_dPj.dump(std::cout);
  //std::cout << "\n";

  return int_cdPi_dPj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_assembleBilinearOperatorQUAD4()
{
  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    if (cell.type() != IT_Quad4)
      ARCANE_FATAL("Only Quad4 cell type is supported");

    auto K_e = _computeElementMatrixQUAD4(cell); // element stifness matrix
    //             # assemble elementary matrix into the global one
    //             # elementary terms are positionned into K according
    //             # to the rank of associated node in the mesh.nodes list
    //             for node1 in elem.nodes:
    //                 inode1=elem.nodes.index(node1) # get position of node1 in nodes list
    //                 for node2 in elem.nodes:
    //                     inode2=elem.nodes.index(node2)
    //                     K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
    Int32 n1_index = 0;
    for (Node node1 : cell.nodes()) {
      Int32 n2_index = 0;
      for (Node node2 : cell.nodes()) {
        // K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
        Real v = K_e(n1_index, n2_index);
        // m_k_matrix(node1.localId(), node2.localId()) += v;
        if (node1.isOwn()) {
          m_linear_system.matrixAddValue(node_dof.dofId(node1, 0), node_dof.dofId(node2, 0), v);
        }
        ++n2_index;
      }
      ++n1_index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifdef USE_COO_GPU
/**
 * @brief Initialization of the coo matrix. It only works for p=1 since there is
 * one node per Edge.
 * 
 * 
 */
void FemModule::
_buildMatrixGPU()
{
  //Initialization of the coo matrix;
  //This formula only works in p=1

  /*
  //Create a connection between nodes through the faces
  //Useless here because we only need this information once
  IItemFamily* node_family = mesh()->nodeFamily();
  NodeGroup nodes = node_family->allItems();
  auto idx_cn = mesh()->indexedConnectivityMng()->findOrCreateConnectivity(node_family, node_family, "NodeToNeighbourFaceNodes");
  auto* cn = idx_cn->connectivity();
  ENUMERATE_NODE (node, allNodes()) {
  }
  */

  Int32 nnz = nbFace() * 2 + nbNode();
  m_coo_matrix.initialize(m_dof_family, nnz);
  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  //We iterate through the node, and we do not sort anymore : we assume the nodes ID are sorted, and we will iterate throught the column to avoid making < and > comparison
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;

    m_coo_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(node, 0));

    for (Face face : node.faces()) {
      if (face.nodeId(0) == node.localId())
        m_coo_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(face.nodeId(1), 0));
      else
        m_coo_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(face.nodeId(0), 0));
    }
  }
  //m_coo_matrix.printMatrix("test.txt", false);

  //In this one, we begin by filling the diagonal before filling what's left by iterating through the nodes
  /*
  //Fill the diagonal
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;
    m_coo_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(node, 0));
  }

  //Fill what is left
  ENUMERATE_FACE (iface, allFaces()) {
    Face face = *iface;
    auto nodes = face.nodes();
    for (Int32 i = 0; i < nodes.size() - i - 1; i++) {
      m_coo_matrix.setCoordinates(node_dof.dofId(nodes[i], 0), node_dof.dofId(nodes[nodes.size() - i - 1], 0));
      m_coo_matrix.setCoordinates(node_dof.dofId(nodes[nodes.size() - i - 1], 0), node_dof.dofId(nodes[i], 0));
    }
  }

  //Sort the matrix
  m_coo_matrix.sort();
  */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
ARCCORE_DEVICE
NumArray<Real, ExtentsV<3, 3>> FemModule::
_computeElementMatrixTRIA3GPU(CellLocalId icell, IndexedCellNodeConnectivityView cnc, VariableNodeReal3InView in_node_coord)
{
  // Get coordiantes of the triangle element  TRI3
  //------------------------------------------------
  //                  0 o
  //                   . .
  //                  .   .
  //                 .     .
  //              1 o . . . o 2
  //------------------------------------------------
  Real3 m0 = in_node_coord[cnc.nodeId(icell, 0)];
  Real3 m1 = in_node_coord[cnc.nodeId(icell, 1)];
  Real3 m2 = in_node_coord[cnc.nodeId(icell, 2)];

  Real area = 0.5 * ((m1.x - m0.x) * (m2.y - m0.y) - (m2.x - m0.x) * (m1.y - m0.y)); // calculate area

  Real2 dPhi0(m1.y - m2.y, m2.x - m1.x);
  Real2 dPhi1(m2.y - m0.y, m0.x - m2.x);
  Real2 dPhi2(m0.y - m1.y, m1.x - m0.x);

  //We will want to replace fixed matrix by some numarray ?
  NumArray<Real, ExtentsV<2, 3>> b_matrix(eMemoryRessource::Device);
  b_matrix(0, 0) = dPhi0.x * (1.0 / (2.0 * area));
  b_matrix(0, 1) = dPhi1.x * (1.0 / (2.0 * area));
  b_matrix(0, 2) = dPhi2.x * (1.0 / (2.0 * area));

  b_matrix(1, 0) = dPhi0.y * (1.0 / (2.0 * area));
  b_matrix(1, 1) = dPhi1.y * (1.0 / (2.0 * area));
  b_matrix(1, 2) = dPhi2.y * (1.0 / (2.0 * area));
  NumArray<Real, ExtentsV<3, 3>> int_cdPi_dPj;

  //Multiplying b_matrix by its transpose, and doing the mult in place in the same loop
  for (Int32 i = 0; i < 3; i++) {
    for (Int32 j = 0; j < 3; j++) {
      Real x = 0.0;
      for (Int32 k = 0; k < 2; k++) {
        x += b_matrix(k, i) * b_matrix(k, j);
      }
      int_cdPi_dPj(i, j) = x * area;
    }
  }

  //info() << "Cell=" << cell.localId();
  //std::cout << " int_cdPi_dPj=";
  //int_cdPi_dPj.dump(std::cout);
  //std::cout << "\n";

  return int_cdPi_dPj;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_assembleCooGPUBilinearOperatorTRIA3()
{
#ifdef REGISTER_TIME
  logger << "-------------------------------------------------------------------------------------\n"
         << "Using CPU coo with NumArray format\n";
  auto lhs_start = std::chrono::high_resolution_clock::now();
  double compute_average = 0;
  double global_build_average = 0;
  double build_time = 0;
#endif
  // Build the coo matrix
  _buildMatrixGPU();
#ifdef REGISTER_TIME
  auto build_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> build_duration = build_stop - lhs_start;
  build_time = build_duration.count();
#endif

  RunQueue* queue = acceleratorMng()->defaultQueue();
  // Boucle sur les mailles déportée sur accélérateur
  auto command = makeCommand(queue);

  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
  UnstructuredMeshConnectivityView m_connectivity_view;
  VariableNodeReal3InView in_node_coord = ax::viewIn(command, m_node_coord);
  m_connectivity_view.setMesh(this->mesh());
  auto cnc = m_connectivity_view.cellNode();

  command << RUNCOMMAND_ENUMERATE(Cell, icell, allCells())
  {

    auto K_e = _computeElementMatrixTRIA3GPU(icell, cnc, in_node_coord); // element stifness matrix

    //             # assemble elementary matrix into the global one
    //             # elementary terms are positionned into K according
    //             # to the rank of associated node in the mesh.nodes list
    //             for node1 in elem.nodes:
    //                 inode1=elem.nodes.index(node1) # get position of node1 in nodes list
    //                 for node2 in elem.nodes:
    //                     inode2=elem.nodes.index(node2)
    //                     K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
    Int32 n1_index = 0;
    for (NodeLocalId node1 : cnc.nodes(icell)) {
      Int32 n2_index = 0;
      for (NodeLocalId node2 : cnc.nodes(icell)) {
        // K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
        Real v = K_e(n1_index, n2_index);
        // m_k_matrix(node1.localId(), node2.localId()) += v;
        //replacing the isOwn (probably with a nice view)
        if (node1.isOwn()) {
          m_coo_matrix.matrixAddValue(node_dof.dofId(node1, 0), node_dof.dofId(node2, 0), v);
        }
        ++n2_index;
      }
      ++n1_index;
    }
  };

#ifdef REGISTER_TIME
  auto lhs_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = lhs_end - lhs_start;
  double lhs_loc_time = duration.count();
  logger << "Building time of the coo matrix :" << build_time << "\n"
         << "Compute Elements average time : " << compute_average / nbCell() << "\n"
         << "Compute Elements total time : " << compute_average << "\n"
         << "Add in global matrix average time : " << global_build_average / nbCell() << "\n"
         << "Add in global matrix total time : " << global_build_average << "\n"
         << "LHS Total time : " << lhs_loc_time << "\n"
         << "Build matrix time in lhs :" << build_time / lhs_loc_time * 100 << "%\n"
         << "Compute element time in lhs : " << compute_average / lhs_loc_time * 100 << "%\n"
         << "Add in global matrix time in lhs : " << global_build_average / lhs_loc_time * 100 << "%\n\n"
         << "-------------------------------------------------------------------------------------\n\n";
  lhs_time += lhs_loc_time;
#endif
  //m_coo_matrix.printMatrix("ref.txt", false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

#ifdef USE_CSR

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
 * @brief Initialization of the coo matrix. It only works for p=1 since there is
 * one node per Edge.
 * 
 * 
 */
void FemModule::
_buildMatrixCsr()
{
  //Initialization of the coo matrix;
  //This formula only works in p=1

  /*
  //Create a connection between nodes through the faces
  //Useless here because we only need this information once
  IItemFamily* node_family = mesh()->nodeFamily();
  NodeGroup nodes = node_family->allItems();
  auto idx_cn = mesh()->indexedConnectivityMng()->findOrCreateConnectivity(node_family, node_family, "NodeToNeighbourFaceNodes");
  auto* cn = idx_cn->connectivity();
  ENUMERATE_NODE (node, allNodes()) {
  }
  */

  Int32 nnz = nbFace() * 2 + nbNode();
  m_csr_matrix.initialize(m_dof_family, nnz, nbNode());
  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
  //We iterate through the node, and we do not sort anymore : we assume the nodes ID are sorted, and we will iterate throught the column to avoid making < and > comparison
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;

    m_csr_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(node, 0));

    for (Face face : node.faces()) {
      if (face.nodeId(0) == node.localId())
        m_csr_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(face.nodeId(1), 0));
      else
        m_csr_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(face.nodeId(0), 0));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_assembleCsrBilinearOperatorTRIA3()
{
#ifdef REGISTER_TIME
  logger << "-------------------------------------------------------------------------------------\n"
         << "Using CPU CSR with NumArray format\n";
  auto lhs_start = std::chrono::high_resolution_clock::now();
  double compute_average = 0;
  double global_build_average = 0;
  double build_time = 0;
#endif
  // Build the coo matrix
  _buildMatrixCsr();
#ifdef REGISTER_TIME
  auto build_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> build_duration = build_stop - lhs_start;
  build_time = build_duration.count();
#endif

  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    if (cell.type() != IT_Triangle3)
      ARCANE_FATAL("Only Triangle3 cell type is supported");

#ifdef REGISTER_TIME
    auto compute_El_start = std::chrono::high_resolution_clock::now();
#endif

    auto K_e = _computeElementMatrixTRIA3(cell); // element stifness matrix

#ifdef REGISTER_TIME
    auto compute_El_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compute_duration = compute_El_stop - compute_El_start;
    compute_average += compute_duration.count();
#endif

//             # assemble elementary matrix into the global one
//             # elementary terms are positionned into K according
//             # to the rank of associated node in the mesh.nodes list
//             for node1 in elem.nodes:
//                 inode1=elem.nodes.index(node1) # get position of node1 in nodes list
//                 for node2 in elem.nodes:
//                     inode2=elem.nodes.index(node2)
//                     K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
#ifdef REGISTER_TIME
    auto global_build_start = std::chrono::high_resolution_clock::now();
#endif
    Int32 n1_index = 0;
    for (Node node1 : cell.nodes()) {
      Int32 n2_index = 0;
      for (Node node2 : cell.nodes()) {
        // K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
        Real v = K_e(n1_index, n2_index);
        // m_k_matrix(node1.localId(), node2.localId()) += v;
        if (node1.isOwn()) {
          m_csr_matrix.matrixAddValue(node_dof.dofId(node1, 0), node_dof.dofId(node2, 0), v);
        }
        ++n2_index;
      }
      ++n1_index;
    }

#ifdef REGISTER_TIME
    auto global_build_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> global_build_duration = global_build_stop - global_build_start;
    global_build_average += global_build_duration.count();
#endif
  }

#ifdef REGISTER_TIME
  auto lhs_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = lhs_end - lhs_start;
  double lhs_loc_time = duration.count();
  logger << "Building time of the csr matrix :" << build_time << "\n"
         << "Compute Elements average time : " << compute_average / nbCell() << "\n"
         << "Compute Elements total time : " << compute_average << "\n"
         << "Add in global matrix average time : " << global_build_average / nbCell() << "\n"
         << "Add in global matrix total time : " << global_build_average << "\n"
         << "LHS Total time : " << lhs_loc_time << "\n"
         << "Build matrix time in lhs :" << build_time / lhs_loc_time * 100 << "%\n"
         << "Compute element time in lhs : " << compute_average / lhs_loc_time * 100 << "%\n"
         << "Add in global matrix time in lhs : " << global_build_average / lhs_loc_time * 100 << "%\n\n"
         << "-------------------------------------------------------------------------------------\n\n";
  lhs_time += lhs_loc_time;
#endif
}

#endif

#ifdef USE_COO
/**
 * @brief Initialization of the coo matrix. It only works for p=1 since there is
 * one node per Edge.
 * 
 * 
 */
void FemModule::
_buildMatrix()
{
  //Initialization of the coo matrix;
  //This formula only works in p=1

  /*
  //Create a connection between nodes through the faces
  //Useless here because we only need this information once
  IItemFamily* node_family = mesh()->nodeFamily();
  NodeGroup nodes = node_family->allItems();
  auto idx_cn = mesh()->indexedConnectivityMng()->findOrCreateConnectivity(node_family, node_family, "NodeToNeighbourFaceNodes");
  auto* cn = idx_cn->connectivity();
  ENUMERATE_NODE (node, allNodes()) {
  }
  */

  Int32 nnz = nbFace() * 2 + nbNode();
  m_coo_matrix.initialize(m_dof_family, nnz);
  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  //We iterate through the node, and we do not sort anymore : we assume the nodes ID are sorted, and we will iterate throught the column to avoid making < and > comparison
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;

    m_coo_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(node, 0));

    for (Face face : node.faces()) {
      if (face.nodeId(0) == node.localId())
        m_coo_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(face.nodeId(1), 0));
      else
        m_coo_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(face.nodeId(0), 0));
    }
  }

  /*
  //In this one, we begin by filling the diagonal before filling what's left by iterating through the nodes

  //Fill the diagonal
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;
    m_coo_matrix.setCoordinates(node_dof.dofId(node, 0), node_dof.dofId(node, 0));
  }

  //Fill what is left
  ENUMERATE_FACE (iface, allFaces()) {
    Face face = *iface;
    auto nodes = face.nodes();
    for (Int32 i = 0; i < nodes.size() - i - 1; i++) {
      m_coo_matrix.setCoordinates(node_dof.dofId(nodes[i], 0), node_dof.dofId(nodes[nodes.size() - i - 1], 0));
      m_coo_matrix.setCoordinates(node_dof.dofId(nodes[nodes.size() - i - 1], 0), node_dof.dofId(nodes[i], 0));
    }
  }

  //Sort the matrix
  m_coo_matrix.sort();
  */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_assembleCooBilinearOperatorTRIA3()
{
#ifdef REGISTER_TIME
  logger << "-------------------------------------------------------------------------------------\n"
         << "Using CPU coo with NumArray format\n";
  auto lhs_start = std::chrono::high_resolution_clock::now();
  double compute_average = 0;
  double global_build_average = 0;
  double build_time = 0;
#endif
  // Build the coo matrix
  _buildMatrix();
#ifdef REGISTER_TIME
  auto build_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> build_duration = build_stop - lhs_start;
  build_time = build_duration.count();
#endif

  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    if (cell.type() != IT_Triangle3)
      ARCANE_FATAL("Only Triangle3 cell type is supported");

#ifdef REGISTER_TIME
    auto compute_El_start = std::chrono::high_resolution_clock::now();
#endif

    auto K_e = _computeElementMatrixTRIA3(cell); // element stifness matrix

#ifdef REGISTER_TIME
    auto compute_El_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compute_duration = compute_El_stop - compute_El_start;
    compute_average += compute_duration.count();
#endif

//             # assemble elementary matrix into the global one
//             # elementary terms are positionned into K according
//             # to the rank of associated node in the mesh.nodes list
//             for node1 in elem.nodes:
//                 inode1=elem.nodes.index(node1) # get position of node1 in nodes list
//                 for node2 in elem.nodes:
//                     inode2=elem.nodes.index(node2)
//                     K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
#ifdef REGISTER_TIME
    auto global_build_start = std::chrono::high_resolution_clock::now();
#endif
    Int32 n1_index = 0;
    for (Node node1 : cell.nodes()) {
      Int32 n2_index = 0;
      for (Node node2 : cell.nodes()) {
        // K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
        Real v = K_e(n1_index, n2_index);
        // m_k_matrix(node1.localId(), node2.localId()) += v;
        if (node1.isOwn()) {
          m_coo_matrix.matrixAddValue(node_dof.dofId(node1, 0), node_dof.dofId(node2, 0), v);
        }
        ++n2_index;
      }
      ++n1_index;
    }
#ifdef REGISTER_TIME
    auto global_build_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> global_build_duration = global_build_stop - global_build_start;
    global_build_average += global_build_duration.count();
#endif
  }
  m_coo_matrix.printMatrix("COO.txt", false);

#ifdef REGISTER_TIME
  auto lhs_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = lhs_end - lhs_start;
  double lhs_loc_time = duration.count();
  logger << "Building time of the coo matrix :" << build_time << "\n"
         << "Compute Elements average time : " << compute_average / nbCell() << "\n"
         << "Compute Elements total time : " << compute_average << "\n"
         << "Add in global matrix average time : " << global_build_average / nbCell() << "\n"
         << "Add in global matrix total time : " << global_build_average << "\n"
         << "LHS Total time : " << lhs_loc_time << "\n"
         << "Build matrix time in lhs :" << build_time / lhs_loc_time * 100 << "%\n"
         << "Compute element time in lhs : " << compute_average / lhs_loc_time * 100 << "%\n"
         << "Add in global matrix time in lhs : " << global_build_average / lhs_loc_time * 100 << "%\n\n"
         << "-------------------------------------------------------------------------------------\n\n";
  lhs_time += lhs_loc_time;
#endif
  //m_coo_matrix.printMatrix("ref.txt", false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

#ifdef USE_CUSPARSE_ADD
void FemModule::
printCsrMatrix(std::string fileName, cusparseCsr csr, bool is_coo)
{
  ofstream file(fileName);
  file << "size :" << csr.nnz << "\n";
  for (auto i = 0; i < (is_coo ? csr.nnz : nbNode()); i++) {
    file << csr.csrRow[i] << " ";
  }
  file << "\n";
  for (auto i = 0; i < csr.nnz; i++) {
    file << csr.csrCol[i] << " ";
  }
  file << "\n";
  for (auto i = 0; i < csr.nnz; i++) {
    file << csr.csrVal[i] << " ";
  }
  file.close();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_computeCusparseElementMatrix(cusparseCsr& result, cusparseCsr& global, Cell cell, cusparseHandle_t handle, IndexedNodeDoFConnectivityView node_dof
#ifdef REGISTER_TIME
                              ,
                              computeTimer& timer
#endif
)
{

#ifdef REGISTER_TIME
  auto compute_start = std::chrono::high_resolution_clock::now();
#endif
  /*-------------------------------------------------------------------------------------------------------------------------------*/
  //First part : compute element matrix
  // Get coordiantes of the triangle element  TRI3
  //------------------------------------------------
  //                  0 o
  //                   . .
  //                  .   .
  //                 .     .
  //              1 o . . . o 2
  //------------------------------------------------
  Real3 m0 = m_node_coord[cell.nodeId(0)];
  Real3 m1 = m_node_coord[cell.nodeId(1)];
  Real3 m2 = m_node_coord[cell.nodeId(2)];
  // Dead code to remember  : Real3 m2 = in_node_coord[cnc.nodeId(icell, 2)];

  Real area = _computeAreaTriangle3(cell); // calculate area

  Real2 dPhi0(m1.y - m2.y, m2.x - m1.x);
  Real2 dPhi1(m2.y - m0.y, m0.x - m2.x);
  Real2 dPhi2(m0.y - m1.y, m1.x - m0.x);

  FixedMatrix<2, 3> b_matrix;
  b_matrix(0, 0) = dPhi0.x;
  b_matrix(0, 1) = dPhi1.x;
  b_matrix(0, 2) = dPhi2.x;

  b_matrix(1, 0) = dPhi0.y;
  b_matrix(1, 1) = dPhi1.y;
  b_matrix(1, 2) = dPhi2.y;

  b_matrix.multInPlace(1.0 / (2.0 * area));

  FixedMatrix<3, 3> int_cdPi_dPj = matrixMultiplication(matrixTranspose(b_matrix), b_matrix);
  int_cdPi_dPj.multInPlace(area);

#ifdef REGISTER_TIME
  auto compute_el_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> compute_el_duration = compute_el_stop - compute_start;
  timer.compute_el += compute_el_duration.count();
  auto convert_coo = std::chrono::high_resolution_clock::now();
#endif
  /*-------------------------------------------------------------------------------------------------------------------------------*/
  //Second part : putting the matrix in COO format (might want to optimsie that part by doing it earlier) before converting it to csr

  //Must change int_cdPi_dPj in a COO matrix (before converting it to csr);
  Int32* row_indexes;
  CHECK_CUDA(cudaMallocManaged(&row_indexes, 9 * sizeof(Int32)));
  Int32* col_indexes;
  CHECK_CUDA(cudaMallocManaged(&col_indexes, 9 * sizeof(Int32)));
  float* vals;
  CHECK_CUDA(cudaMallocManaged(&vals, 9 * sizeof(float)));

  cusparseMatDescr_t local_mat;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&local_mat));
  cusparseCsr local;
  local.desc = local_mat;
  local.csrRow = row_indexes;
  local.csrCol = col_indexes;
  local.csrVal = vals;
  local.nnz = 9;

  int i = 0;
  int j = 0;
  for (NodeLocalId node1 : cell.nodes()) {
    j = 0;
    for (NodeLocalId node2 : cell.nodes()) {
      vals[i * 3 + j] = int_cdPi_dPj(i, j);
      row_indexes[i * 3 + j] = node_dof.dofId(node1, 0);
      col_indexes[i * 3 + j] = node_dof.dofId(node2, 0);
      j++;
    }
    i++;
  }

#ifdef REGISTER_TIME
  auto convert_coo_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> convert_coo_time = convert_coo_stop - convert_coo;
  timer.convert_coo += convert_coo_time.count();
  auto sort_coo = std::chrono::high_resolution_clock::now();
#endif
  //Sorting of the COO values with an insertion sort
  Int32 rj = 0;
  Int32 cj = 0;
  float vj = 0;
  for (i = 1; i < 9; i++) {
    rj = row_indexes[i];
    cj = col_indexes[i];
    vj = vals[i];
    j = i - 1;
    while (j >= 0 && row_indexes[j] > rj) {
      row_indexes[j + 1] = row_indexes[j];
      col_indexes[j + 1] = col_indexes[j];
      vals[j + 1] = vals[j];
      j--;
    }
    row_indexes[j + 1] = rj;
    col_indexes[j + 1] = cj;
    vals[j + 1] = vj;
    Int32 k = j - 1;
    Int32 rk, ck;
    float vk;
    if (j > 0) {
      rk = row_indexes[j];
      ck = col_indexes[j];
      vk = vals[j];
      while (k >= 0 && row_indexes[k] == rk && col_indexes[k] > ck) {
        col_indexes[k + 1] = col_indexes[k];
        vals[k + 1] = vals[k];
        k--;
      }
      col_indexes[k + 1] = ck;
      vals[k + 1] = vk;
    }
  }
#ifdef REGISTER_TIME
  auto sort_coo_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sort_coo_time = sort_coo_stop - sort_coo;
  std::chrono::duration<double> convert_coo_tot = sort_coo_stop - convert_coo;
  timer.sort_coo += sort_coo_time.count();
  timer.convert_coo_tot += convert_coo_tot.count();
  auto convert_csr = std::chrono::high_resolution_clock::now();
#endif

  //conversion from COO to CSR
  Int32* csrRowPtr;
  CHECK_CUDA(cudaMallocManaged(&csrRowPtr, nbNode() * sizeof(Int32)));
  CHECK_CUSPARSE(cusparseXcoo2csr(handle, row_indexes, 9, nbNode(), csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
  local.csrRow = csrRowPtr;
  CHECK_CUDA(cudaFree(row_indexes));

#ifdef REGISTER_TIME
  auto convert_csr_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> convert_csr_time = convert_csr_stop - convert_csr;
  std::chrono::duration<double> convert_tot = convert_csr_stop - convert_coo;
  timer.convert_csr_tot += convert_coo_time.count();
  timer.convert_tot += convert_tot.count();
  auto adding_global = std::chrono::high_resolution_clock::now();
#endif
  /*-------------------------------------------------------------------------------------------------------------------------------*/
  // Third part : adding the local and global, storing result in the res

  //Adding the CSR local matrix to the global one using cusparsecsrgeam2
  //see https://docs.nvidia.com/cuda/cusparse/index.html?highlight=cusparseScsrgeam#cusparse-t-csrgeam2 for the example code
  Int32 baseC,
  nnzC;
  size_t bufferSizeInBytes;
  char* buffer = NULL;
  Int32* nnzTotalDevHostPtr = &nnzC;
  CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  //Int32* csrRowPtrC;
  //CHECK_CUDA(cudaMallocManaged(&csrRowPtrC, nbNode() + 1 * sizeof(Int32)));
  float alpha = 1.0;
  float beta = 1.0;
  Int32 m = nbNode();
  CHECK_CUSPARSE(cusparseScsrgeam2_bufferSizeExt(handle, m, m,
                                                 &alpha,
                                                 global.desc, global.nnz,
                                                 global.csrVal, global.csrRow, global.csrCol,
                                                 &beta,
                                                 local.desc, local.nnz,
                                                 local.csrVal, local.csrRow, local.csrCol,
                                                 result.desc,
                                                 result.csrVal, result.csrRow, result.csrCol, &bufferSizeInBytes));
  CHECK_CUDA(cudaMallocManaged(&buffer, bufferSizeInBytes * sizeof(char)));
  /*
  std::cerr << "buffer : " << buffer << "\n";
  std::cerr << "nnzTotalDevHst : " << nnzTotalDevHostPtr << "\n";
  std::cerr << "local desc : " << local.desc << "\n";
  std::cerr << "local nnz : " << local.nnz << "\n";
  std::cerr << "local csrRow : " << local.csrRow << "\n";
  std::cerr << "local csrCol: " << local.csrCol << "\n";
  std::cerr << "global desc : " << global.desc << "\n";
  std::cerr << "global nnz : " << global.nnz << "\n";
  std::cerr << "global csrRow : " << global.csrRow << "\n";
  std::cerr << "global csrCol : " << global.csrCol << "\n";
  std::cerr << "result desc : " << result.desc << "\n";
  std::cerr << "result csrRow: " << result.csrRow << "\n";
*/
  CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle, m, m,
                                      local.desc, local.nnz, local.csrRow, local.csrCol,
                                      global.desc, global.nnz, global.csrRow, global.csrCol,
                                      result.desc, result.csrRow, nnzTotalDevHostPtr,
                                      buffer));
  if (NULL != nnzTotalDevHostPtr)
    nnzC = *nnzTotalDevHostPtr;
  else {
    CHECK_CUDA(cudaMemcpy(&nnzC, result.csrRow + m, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&baseC, result.csrRow, sizeof(int), cudaMemcpyDeviceToHost));
    nnzC -= baseC;
  }
  result.nnz = nnzC;
  CHECK_CUDA(cudaMallocManaged(&result.csrCol, sizeof(Int32) * nnzC));
  CHECK_CUDA(cudaMallocManaged(&result.csrVal, sizeof(float) * nnzC));
  CHECK_CUSPARSE(cusparseScsrgeam2(handle, m, m,
                                   &alpha,
                                   local.desc, local.nnz,
                                   local.csrVal, local.csrRow, local.csrCol,
                                   &beta,
                                   global.desc, global.nnz,
                                   global.csrVal, global.csrRow, global.csrCol,
                                   result.desc,
                                   result.csrVal, result.csrRow, result.csrCol,
                                   buffer));

  CHECK_CUDA(cudaFree(local.csrVal));
  CHECK_CUDA(cudaFree(local.csrCol));
  CHECK_CUDA(cudaFree(local.csrRow));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(local.desc));

#ifdef REGISTER_TIME
  auto adding_global_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> adding_tot = adding_global_stop - adding_global;
  std::chrono::duration<double> compute_tot = adding_global_stop - compute_start;
  timer.add_glob += adding_tot.count();
  timer.compute_tot += compute_tot.count();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Assemble Bilinear TRIA3 with cusparse help. It only works for p=1 since there is
 * one node per Edge.
 * 
 * 
 */
void FemModule::
_assembleCusparseBilinearOperatorTRIA3()
{

#ifdef REGISTER_TIME
  logger << "-------------------------------------------------------------------------------------\n"
         << "Using Cusparse for Bilinear assembly\n";
  auto lhs_s = std::chrono::high_resolution_clock::now();
  computeTimer timer = {};
#endif
  //Initialization of the CSR matrix;
  //This formula only works in p=1
#ifdef REGISTER_TIME
  auto cuda_init_start = std::chrono::high_resolution_clock::now();
#endif
  CHECK_CUDA(cudaFree(0));

#ifdef REGISTER_TIME
  auto cuda_init_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cuda_init_time = cuda_init_stop - cuda_init_start;
  double cuda_init = cuda_init_time.count();
#endif

  Int32 nnz = nbFace() * 2 + nbNode();
  //Initialize the global matrix. Everything is in the unified memory
  Int32* res1_row;
  CHECK_CUDA(cudaMallocManaged(&res1_row, sizeof(Int32) * nbNode()));
  Int32* res2_row;
  CHECK_CUDA(cudaMallocManaged(&res2_row, sizeof(Int32) * nbNode()));

  cusparseHandle_t handle;
  CHECK_CUSPARSE(cusparseCreate(&handle));
  //The number of Node must be changed when p != 1

  //init result matrix
  cusparseCsr res1;
  cusparseCsr res2;

  cusparseMatDescr_t res1_desc;
  cusparseMatDescr_t res2_desc;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&res1_desc));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&res2_desc));

  res1.desc = res1_desc;
  res2.desc = res2_desc;
  res1.csrRow = res1_row;
  res2.csrRow = res2_row;
  res1.csrCol = NULL;
  res2.csrCol = NULL;

  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  Int32 i = 0;

  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;

    if (i % 2 == 0)
      //computation of the local matrix and adding it in the global one
      _computeCusparseElementMatrix(res1, res2, cell, handle, node_dof
#ifdef REGISTER_TIME
                                    ,
                                    timer
#endif
      );
    else
      _computeCusparseElementMatrix(res2, res1, cell, handle, node_dof
#ifdef REGISTER_TIME
                                    ,
                                    timer
#endif
      );
    i++;
  }
  /*
  if (nbNode() % 2 == 0)
    printCsrMatrix("csrTest.txt", res1, false);
  else
    printCsrMatrix("csrTest.txt", res2, false);

*/
  CHECK_CUSPARSE(cusparseDestroyMatDescr(res1.desc));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(res2.desc));

  //Must free the resulting vectors, be careful
#ifdef REGISTER_TIME
  auto lhs_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = lhs_end - cuda_init_stop;
  double lhs_loc_time = duration.count();
  logger << "Average Time to compute element matrix : " << timer.compute_el / nbCell() << "\n"
         << "Total Time to compute element matrix : " << timer.compute_el << "\n"
         << "Percentage time to compute element matrix : " << timer.compute_el / lhs_loc_time * 100 << "%\n"
         << "Average Time to convert to coo : " << timer.convert_coo / nbCell() << "\n"
         << "Total Time to convert to coo : " << timer.convert_coo << "\n"
         << "Percentage Time to convert to coo : " << timer.convert_coo / lhs_loc_time * 100 << "%\n"
         << "Average Time to sort the coo : " << timer.sort_coo / nbCell() << "\n"
         << "Total Time to sort the coo : " << timer.sort_coo << "\n"
         << "Percentage Time to sort the coo : " << timer.sort_coo / lhs_loc_time * 100 << "%\n"
         << "Average Time to convert and sort to coo : " << timer.convert_coo_tot / nbCell() << "\n"
         << "Total Time to convert and sort to coo : " << timer.convert_coo_tot << "\n"
         << "Percentage Time to convert and sort to coo : " << timer.convert_coo_tot / lhs_loc_time * 100 << "%\n"
         << "Average Time to convert to csr : " << timer.convert_csr_tot / nbCell() << "\n"
         << "Total Time to convert to csr : " << timer.convert_csr_tot << "\n"
         << "Percentage Time to convert to csr : " << timer.convert_csr_tot / lhs_loc_time * 100 << "%\n"
         << "Average Time to convert the computed matrix : " << timer.convert_tot / nbCell() << "\n"
         << "Total Time to convert the computed matrix : " << timer.convert_tot << "\n"
         << "Percentage Time to convert the computed matrix : " << timer.convert_tot / lhs_loc_time * 100 << "%\n"
         << "Average Time to add to the global matrix : " << timer.add_glob / nbCell() << "\n"
         << "Total Time to add to the global matrix : " << timer.add_glob << "\n"
         << "Percentage Time to add to the global matrix : " << timer.add_glob / lhs_loc_time * 100 << "%\n"
         << "Average Time to make the computation operation : " << timer.compute_tot / nbCell() << "\n"
         << "Total Time to make the computation operation : " << timer.compute_tot << "\n"
         << "Percentage Time to make the computation operation : " << timer.compute_tot / lhs_loc_time * 100 << "%\n"
         << "Total time for the lhs computation : " << lhs_loc_time << "\n"
         << "Total time of the cuda init : " << cuda_init << "\n"
         << "Total time of lhs with the init : " << cuda_init + lhs_loc_time << "\n"
         << "-------------------------------------------------------------------------------------\n\n";
  lhs_time += lhs_loc_time;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif

void FemModule::
_assembleBilinearOperatorTRIA3()
{
  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

#ifdef REGISTER_TIME
  logger << "-------------------------------------------------------------------------------------\n"
         << "Using hashmap legacy format\n";
  auto lhs_start = std::chrono::high_resolution_clock::now();
  double compute_average = 0;
  double global_build_average = 0;
#endif

  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    if (cell.type() != IT_Triangle3)
      ARCANE_FATAL("Only Triangle3 cell type is supported");

#ifdef REGISTER_TIME
    auto compute_El_start = std::chrono::high_resolution_clock::now();
#endif

    auto K_e = _computeElementMatrixTRIA3(cell); // element stifness matrix

#ifdef REGISTER_TIME
    auto compute_El_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compute_duration = compute_El_stop - compute_El_start;
    compute_average += compute_duration.count();
#endif

//             # assemble elementary matrix into the global one
//             # elementary terms are positionned into K according
//             # to the rank of associated node in the mesh.nodes list
//             for node1 in elem.nodes:
//                 inode1=elem.nodes.index(node1) # get position of node1 in nodes list
//                 for node2 in elem.nodes:
//                     inode2=elem.nodes.index(node2)
//                     K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
#ifdef REGISTER_TIME
    auto global_build_start = std::chrono::high_resolution_clock::now();
#endif
    Int32 n1_index = 0;
    for (Node node1 : cell.nodes()) {
      Int32 n2_index = 0;
      for (Node node2 : cell.nodes()) {
        // K[node1.rank,node2.rank]=K[node1.rank,node2.rank]+K_e[inode1,inode2]
        Real v = K_e(n1_index, n2_index);
        // m_k_matrix(node1.localId(), node2.localId()) += v;
        if (node1.isOwn()) {
          m_linear_system.matrixAddValue(node_dof.dofId(node1, 0), node_dof.dofId(node2, 0), v);
        }
        ++n2_index;
      }
      ++n1_index;
    }
#ifdef REGISTER_TIME
    auto global_build_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> global_build_duration = global_build_stop - global_build_start;
    global_build_average += global_build_duration.count();
#endif
  }

#ifdef REGISTER_TIME
  auto lhs_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = lhs_end - lhs_start;
  double lhs_loc_time = duration.count();
  logger << "compute elements average time : " << compute_average / nbCell() << "\n"
         << "compute elements total time : " << compute_average << "\n"
         << "add in global matrix average time : " << global_build_average / nbCell() << "\n"
         << "add in global matrix total time : " << global_build_average << "\n"
         << "lhs total time : " << lhs_loc_time << "\n"
         << "compute element time in lhs : " << compute_average / lhs_loc_time * 100 << "%\n"
         << "add in global matrix time in lhs : " << global_build_average / lhs_loc_time * 100 << "%\n\n"
         << "-------------------------------------------------------------------------------------\n\n";
  lhs_time += lhs_loc_time;
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_solve()
{

#ifdef REGISTER_TIME
  auto solve_start = std::chrono::high_resolution_clock::now();
#endif

  m_linear_system.solve();

  // Re-Apply boundary conditions because the solver has modified the value
  // of u on all nodes
  _applyDirichletBoundaryConditions();

  {
    VariableDoFReal& dof_u(m_linear_system.solutionVariable());
    // Copy RHS DoF to Node u
    auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
    ENUMERATE_ (Node, inode, ownNodes()) {
      Node node = *inode;
      Real v = dof_u[node_dof.dofId(node, 0)];
      m_u[node] = v;
    }
  }

  //test
  m_u.synchronize();
  // def update_T(self,T):
  //     """Update u value on nodes after the FE resolution"""
  //     for i in range(0,len(self.mesh.nodes)):
  //         node=self.mesh.nodes[i]
  //         # don't update T imposed by Dirichlet BC
  //         if not node.is_T_fixed:
  //             self.mesh.nodes[i].T=T[i]

  const bool do_print = (allNodes().size() < 200);
  if (do_print) {
    ENUMERATE_ (Node, inode, allNodes()) {
      Node node = *inode;
      info() << "T[" << node.localId() << "][" << node.uniqueId() << "] = "
             << m_u[node];
      //info() << "T[]" << node.uniqueId() << " "
      //       << m_u[node];
    }
  }

#ifdef REGISTER_TIME
  auto solve_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_duration = solve_end - solve_start;
  solver_time = solve_duration.count();
  logger << "Solver duration : " << solver_time << "\n";
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_checkResultFile()
{
  String filename = options()->resultFile();
  info() << "CheckResultFile filename=" << filename;
  if (filename.empty())
    return;
  const double epsilon = 1.0e-4;
  checkNodeResultFile(traceMng(), filename, m_u, epsilon);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_FEM(FemModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

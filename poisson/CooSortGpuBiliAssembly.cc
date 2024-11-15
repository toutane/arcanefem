// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CooSortGpuiBiliAssembly.cc                                  (C) 2022-2024 */
/*                                                                           */
/* Methods of the bilinear assembly phase using the S-COO data structure     */
/* which handle the parallelization on GPU using Arcane accelerator API and  */
/* an atomic operation for adding the value into the global matrix           */
/*                                                                           */
/* Those algorithms take advantage of the sort implied by Arcane on the mesh */
/* if any, making the ROW array sorted by default.                           */
/*                                                                           */
/* The building of the sparsity is implemented for 2D mesh, 3D mesh with and */
/* without edge.                                                             */
/*                                                                           */
/* Both ROW and COLUMN arrays of the matrix are then sorted, in all cases.   */
/* Sort is currently made on CPU.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "FemModule.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::_buildMatrixCooSortGPU()
{
  Int8 mesh_dim = mesh()->dimension();

  if (mesh_dim != 2 && mesh_dim != 3)
    ARCANE_THROW(NotSupportedException, "Only mesh of dimension 2 or 3 are supported");

  Int64 nb_edge = (mesh_dim == 2) ? nbFace() : m_nb_edge;
  Int32 nb_non_zero = nb_edge * 2 + nbNode();
  m_coo_matrix.initialize(m_dof_family, nb_non_zero);

  RunQueue* queue = acceleratorMng()->defaultQueue();
  auto command = makeCommand(queue);

  UnstructuredMeshConnectivityView m_connectivity_view;
  m_connectivity_view.setMesh(mesh());

  auto inout_m_matrix_row = viewInOut(command, m_coo_matrix.m_matrix_row);
  auto inout_m_matrix_column = viewInOut(command, m_coo_matrix.m_matrix_column);

  if (mesh_dim == 2) {

    info()
    << "_buildMatrixCooSortGPU for 2D mesh with face-node connectivity";
    NumArray<uint, MDDim1> neighbors(nbNode());
    SmallSpan<uint> in_data = neighbors.to1DSmallSpan();

    UnstructuredMeshConnectivityView connectivity_view(mesh());
    auto node_face_connectivity_view = connectivity_view.nodeFace();

    {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_ENUMERATE(Node, node_id, allNodes())
      {
        in_data[node_id] = node_face_connectivity_view.nbFace(node_id) + 1;
      };
    }
    queue->barrier();

    NumArray<uint, MDDim1> copy_output_data(nbNode());
    SmallSpan<uint> out_data = copy_output_data.to1DSmallSpan();
    Accelerator::Scanner<uint> scanner;
    scanner.exclusiveSum(queue, in_data, out_data);

    auto face_node_connectivity_view = connectivity_view.faceNode();

    {
      auto command = makeCommand(queue);
      // Fill the neighbors relation (including node with itself) into the
      // matrix
      command << RUNCOMMAND_ENUMERATE(Node, node_id, allNodes())
      {
        Int32 offset = out_data[node_id];

        for (auto edge_id : node_face_connectivity_view.faceIds(node_id)) {
          auto nodes = face_node_connectivity_view.nodes(edge_id);
          inout_m_matrix_row[offset] = node_id;
          inout_m_matrix_column[offset] =
          nodes[0] == node_id ? nodes[1] : nodes[0];
          ++offset;
        }

        inout_m_matrix_row[offset] = node_id;
        inout_m_matrix_column[offset] = node_id;
      };
    }
  }
  else if (options()
           ->createEdges()) { // 3D mesh without node-node connectivity

    info()
    << "_buildMatrixCooSortGPU for 3D mesh with edge-node connectivity";
    NumArray<uint, MDDim1> neighbors(nbNode());
    SmallSpan<uint> in_data = neighbors.to1DSmallSpan();

    UnstructuredMeshConnectivityView connectivity_view(mesh());
    auto node_edge_connectivity_view = connectivity_view.nodeEdge();

    {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_ENUMERATE(Node, node_id, allNodes())
      {
        in_data[node_id] = node_edge_connectivity_view.nbEdge(node_id) + 1;
      };
    }
    queue->barrier();

    NumArray<uint, MDDim1> copy_output_data(nbNode());
    SmallSpan<uint> out_data = copy_output_data.to1DSmallSpan();
    Accelerator::Scanner<uint> scanner;
    scanner.exclusiveSum(queue, in_data, out_data);

    auto edge_node_connectivity_view = connectivity_view.edgeNode();

    {
      auto command = makeCommand(queue);
      // Fill the neighbors relation (including node with itself) into the
      // matrix
      command << RUNCOMMAND_ENUMERATE(Node, node_id, allNodes())
      {
        Int32 offset = out_data[node_id];

        for (auto edge_id : node_edge_connectivity_view.edgeIds(node_id)) {
          auto nodes = edge_node_connectivity_view.nodes(edge_id);
          inout_m_matrix_row[offset] = node_id;
          inout_m_matrix_column[offset] =
          nodes[0] == node_id ? nodes[1] : nodes[0];
          ++offset;
        }

        inout_m_matrix_row[offset] = node_id;
        inout_m_matrix_column[offset] = node_id;
      };
    }
  }
  else { // 3D mesh with node-node connectivity

    auto* connectivity_ptr = m_node_node_via_edge_connectivity.get();
    ARCANE_CHECK_POINTER(connectivity_ptr);
    IndexedNodeNodeConnectivityView nn_cv = connectivity_ptr->view();

    // We allow the use of the accelerated algorithm only if the user provides
    // an accelerator runtime. On Cpu, the not-accelerated version is faster.

    if (queue->isAcceleratorPolicy()) {
      info() << "Using accelerated version of _buildMatrixCooSortGPU for 3D mesh "
                "with node-node connectivity";

      // This array will contain the number of neighbors of each node
      // (type uint is enough for counting neighbors).
      NumArray<uint, MDDim1> nb_neighbor_arr;
      nb_neighbor_arr.resize(nbNode());
      auto inout_nb_neighbor_arr = viewInOut(command, nb_neighbor_arr);

      command << RUNCOMMAND_ENUMERATE(Node, node_idx, allNodes())
      {
        inout_nb_neighbor_arr[node_idx] = nn_cv.nbNode(node_idx) + 1;
        // We add 1 to count the node's relation with itself.
      };

      // Do the exclusive cumulative sum of the neighbors array
      SmallSpan<uint> input = nb_neighbor_arr.to1DSmallSpan();
      NumArray<uint, MDDim1> tmp_output;
      tmp_output.resize(nbNode());
      SmallSpan<uint> output = tmp_output.to1DSmallSpan();
      Accelerator::Scanner<uint> scanner;
      scanner.exclusiveSum(queue, input, output);

      // Fill the neighbors relation (including node with itself) into the matrix
      command << RUNCOMMAND_ENUMERATE(Node, node_idx, allNodes())
      {
        Int32 offset = output[node_idx];

        for (NodeLocalId other_node_idx : nn_cv.nodeIds(node_idx)) {
          inout_m_matrix_row[offset] = node_idx;
          inout_m_matrix_column[offset] = other_node_idx;
          ++offset;
        }

        inout_m_matrix_row[offset] = node_idx;
        inout_m_matrix_column[offset] = node_idx;
      };
    }
    else {
      info() << "Using unaccelerated version of _buildMatrixCooSortGPU for 3D mesh "
                "with node-node connectivity";

      auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

      ENUMERATE_NODE (inode, allNodes()) {
        Node node = *inode;
        DoFLocalId dof = node_dof.dofId(node, 0);

        m_coo_matrix.setCoordinates(dof, node_dof.dofId(node, 0));

        for (NodeLocalId other_node : nn_cv.nodeIds(node))
          m_coo_matrix.setCoordinates(dof, node_dof.dofId(other_node, 0));
      }
    }
  }

  // Sort both row and column arrays of the matrix
  m_coo_matrix.sort();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_assembleCooSortGPUBilinearOperatorTRIA3()
{
  info() << "Assembling S-COO GPU Bilinear Operator for TRIA3 elements";

  Timer::Action timer_bili(m_time_stats, "AssembleBilinearOperator_CooSort_Gpu");

  {
    Timer::Action timer_build(m_time_stats, "BuildMatrix");
    _buildMatrixCooSortGPU();
  }

  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
  Int32 row_length = m_coo_matrix.m_matrix_row.totalNbElement();

  RunQueue* queue = acceleratorMng()->defaultQueue();
  auto command = makeCommand(queue);

  auto in_row_coo = ax::viewIn(command, m_coo_matrix.m_matrix_row);
  auto in_col_coo = ax::viewIn(command, m_coo_matrix.m_matrix_column);
  auto in_out_val_coo = ax::viewInOut(command, m_coo_matrix.m_matrix_value);
  UnstructuredMeshConnectivityView m_connectivity_view;
  auto in_node_coord = ax::viewIn(command, m_node_coord);
  m_connectivity_view.setMesh(this->mesh());
  auto cnc = m_connectivity_view.cellNode();
  Arcane::ItemGenericInfoListView nodes_infos(this->mesh()->nodeFamily());
  Arcane::ItemGenericInfoListView cells_infos(this->mesh()->cellFamily());

  Timer::Action timer_add_compute(m_time_stats, "AddAndCompute");

  command << RUNCOMMAND_ENUMERATE(Cell, icell, allCells())
  {
    Real K_e[9] = { 0 };
    {
      _computeElementMatrixTRIA3GPU(icell, cnc, in_node_coord, K_e);
    }

    Int32 n1_index = 0;
    for (NodeLocalId node1 : cnc.nodes(icell)) {
      Int32 n2_index = 0;
      for (NodeLocalId node2 : cnc.nodes(icell)) {
        if (nodes_infos.isOwn(node1)) {
          Real v = K_e[n1_index * 3 + n2_index];
          Int32 row_index = node_dof.dofId(node1, 0);
          Int32 col_index = node_dof.dofId(node2, 0);

          // Find the index of the value using binary search on the ROW array (which is always sorted in S-COO)
          auto value_index = findIndexBinarySearch(row_index, col_index, in_row_coo, in_col_coo, row_length);
          ax::doAtomic<ax::eAtomicOperation::Add>(in_out_val_coo(value_index), v);
        }
        ++n2_index;
      }
      ++n1_index;
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModule::
_assembleCooSortGPUBilinearOperatorTETRA4()
{
  info() << "Assembling S-COO GPU Bilinear Operator for TETRA4 elements";

  Timer::Action timer_bili(m_time_stats, "AssembleBilinearOperator_CooSort_Gpu");

  {
    Timer::Action timer_build(m_time_stats, "BuildMatrix");
    _buildMatrixCooGPU();
  }

  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
  Int32 row_length = m_coo_matrix.m_matrix_row.totalNbElement();

  RunQueue* queue = acceleratorMng()->defaultQueue();
  auto command = makeCommand(queue);

  auto in_row_coo = ax::viewIn(command, m_coo_matrix.m_matrix_row);
  auto in_col_coo = ax::viewIn(command, m_coo_matrix.m_matrix_column);
  auto in_out_val_coo = ax::viewInOut(command, m_coo_matrix.m_matrix_value);
  UnstructuredMeshConnectivityView m_connectivity_view;
  auto in_node_coord = ax::viewIn(command, m_node_coord);
  m_connectivity_view.setMesh(this->mesh());
  auto cnc = m_connectivity_view.cellNode();
  Arcane::ItemGenericInfoListView nodes_infos(this->mesh()->nodeFamily());
  Arcane::ItemGenericInfoListView cells_infos(this->mesh()->cellFamily());

  Timer::Action timer_add_compute(m_time_stats, "AddAndCompute");

  command << RUNCOMMAND_ENUMERATE(Cell, icell, allCells())
  {
    Real K_e[16] = { 0 };
    {
      _computeElementMatrixTETRA4GPU(icell, cnc, in_node_coord, K_e);
    }

    Int32 n1_index = 0;
    for (NodeLocalId node1 : cnc.nodes(icell)) {
      Int32 n2_index = 0;
      for (NodeLocalId node2 : cnc.nodes(icell)) {
        if (nodes_infos.isOwn(node1)) {
          Real v = K_e[n1_index * 4 + n2_index];
          Int32 row_index = node_dof.dofId(node1, 0);
          Int32 col_index = node_dof.dofId(node2, 0);

          // Find the index of the value using binary search on the ROW array (which is always sorted in S-COO)
          auto value_index = findIndexBinarySearch(row_index, col_index, in_row_coo, in_col_coo, row_length);
          ax::doAtomic<ax::eAtomicOperation::Add>(in_out_val_coo(value_index), v);
        }
        ++n2_index;
      }
      ++n1_index;
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
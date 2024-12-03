#include "CsrFormatVec.h"
#include <arcane/accelerator/RunCommandEnumerate.h>
#include <arcane/accelerator/VariableViews.h>
#include <arcane/accelerator/core/AcceleratorCoreGlobal.h>
#include <arcane/accelerator/core/RunQueue.h>
#include <arcane/accelerator/core/Runner.h>
#include <arcane/core/IIndexedIncrementalItemConnectivity.h>
#include <arcane/core/ItemEnumerator.h>
#include <arcane/core/UnstructuredMeshConnectivity.h>
#include <arcane/core/VariableTypedef.h>
#include <arcane/utils/UtilsTypes.h>
#include <arccore/base/ArccoreGlobal.h>
#include <arccore/base/ArgumentException.h>
#include <arccore/base/NotImplementedException.h>
#include <arcane/utils/NumArray.h>
#include "arcane/accelerator/NumArrayViews.h"

namespace Arcane::FemUtils
{

void CsrFormatVec::initialize(Int32 nbRow, Int32 nbNz)
{
  m_matrix_row.resize(nbRow);
  m_matrix_column.resize(nbNz);
  m_matrix_value.resize(nbNz);
  m_matrix_value.fill(0);
}

// This implements a GPU compatible version that works with Arcane scanner API.
// Use node-node connectivity (3D compatible only).
// NOTE: CPU version is faster on Topaze (18M elements 3D mesh), this one should
// be faster for bigger meshes.
void CsrFormatVec::computeRow(Runner* runner, IMesh* mesh, FemDoFsOnNodes* dofs_on_nodes, Ref<IIndexedIncrementalItemConnectivity> m_node_node_via_edge_connectivity)
{
  info() << "CsrFormatVec: Start computing row array...";

  if (mesh->dimension() != 3)
    ARCANE_THROW(Arccore::NotImplementedException, "CsrFormatVec: Not implemented for 2D meshes.");

  ARCANE_CHECK_POINTER(runner);
  auto queue = makeQueue(runner);
  auto command = makeCommand(queue);

  NumArray<Int32, MDDim1> out_data;
  out_data.resize(m_matrix_row.extent0());
  auto copy_out_data = ax::viewInOut(command, out_data);

  {
    auto command = makeCommand(queue);

    ARCANE_CHECK_POINTER(dofs_on_nodes);
    auto node_dof_cv = dofs_on_nodes->nodeDoFConnectivityView();

    ARCANE_CHECK_POINTER(mesh);
    auto* connectivity_ptr = m_node_node_via_edge_connectivity.get();
    IndexedNodeNodeConnectivityView nn_cv = connectivity_ptr->view();
    command << RUNCOMMAND_ENUMERATE(Node, inode, mesh->allNodes())
    {
      Int64 index = node_dof_cv.dofId(inode, 0).localId();
      for (auto dofLid : node_dof_cv.dofs(inode)) {
        Int32 index = dofLid.localId();
        copy_out_data[index] = nn_cv.nbNode(inode) + 1;
      };
    };
  }
  queue.barrier();

  ax::Scanner<Int32> scanner;
  scanner.exclusiveSum(&queue, out_data, m_matrix_row);

  info() << "CsrFormatVec: Done computing row array";
}

} // namespace Arcane::FemUtils

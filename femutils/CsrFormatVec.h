#include "FemUtils.h"
#include "FemDoFsOnNodes.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ISubDomain.h"
#include <arcane/utils/MDDim.h>
#include <arcane/utils/NumArray.h>
#include <arccore/trace/TraceAccessor.h>
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/Scan.h"
#include "arcane/core/MeshVariableScalarRef.h"
#include "arcane/core/UnstructuredMeshConnectivity.h"
#include <arcane/utils/Real3x3.h>
#include <arcane/accelerator/RunCommandEnumerate.h>
#include <arcane/accelerator/core/AcceleratorCoreGlobal.h>
#include <arcane/accelerator/core/RunQueue.h>
#include <arcane/accelerator/core/Runner.h>
#include <arcane/core/ItemEnumerator.h>
#include <arcane/core/UnstructuredMeshConnectivity.h>
#include <arcane/core/VariableTypedef.h>
#include <arcane/utils/UtilsTypes.h>
#include <arccore/base/ArccoreGlobal.h>
#include <arccore/base/ArgumentException.h>
#include <arccore/base/NotImplementedException.h>
#include <arcane/utils/NumArray.h>
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/VariableViews.h"
#include "ArcaneFemFunctions.h"

namespace ax = Arcane::Accelerator;

namespace Arcane::FemUtils
{

struct ComputeMatrixFunctor
{
  ARCCORE_HOST_DEVICE
  FixedMatrix<4, 4> operator()(CellLocalId icell, const IndexedCellNodeConnectivityView& cnc, const ax::VariableNodeReal3InView& in_node_coord) const
  {
    Real volume = ArcaneFemFunctions::MeshOperation::computeVolumeTetra4HostDevice(icell, cnc, in_node_coord);
    return {};
  }
};

class CsrFormatVec : public Arccore::TraceAccessor
{
 public:

  //CsrFormatVec(ISubDomain* sd)
  CsrFormatVec(ISubDomain* sd, IMesh* mesh, Runner* runner, FemDoFsOnNodes* dofs_on_nodes)
  : Arccore::TraceAccessor(sd->traceMng())
  , m_mesh(mesh)
  , m_runner(runner)
  , m_dofs_on_nodes(dofs_on_nodes) {};
  CsrFormatVec(const CsrFormatVec&) = delete;
  CsrFormatVec(CsrFormatVec&&) = delete;
  CsrFormatVec& operator=(const CsrFormatVec&) = delete;
  CsrFormatVec& operator=(CsrFormatVec&&) = delete;
  // Also delete destructor ?

  void initialize(Int32 nbRow, Int32 nbNz);
  void computeRow(Runner* runner, IMesh* mesh, FemDoFsOnNodes* dofs_on_nodes, Ref<IIndexedIncrementalItemConnectivity> m_node_node_via_edge_connectivity);
  // template <int N>
  // void assemble(const std::function<FixedMatrix<N, N>(CellLocalId icell, const IndexedCellNodeConnectivityView& cnc, const ax::VariableNodeReal3InView& in_node_coord)>& compute_element_matrix);
  // void assemble(MeshVariableScalarRefT<Node, Real3> node_coord, const std::function<FixedMatrix<N, N>(CellLocalId icell, const IndexedCellNodeConnectivityView& cnc, const ax::VariableNodeReal3InView& in_node_coord)>& compute_element_matrix)
  template <int N>
  void assemble(IMesh* mesh, MeshVariableScalarRefT<Node, Real3> node_coord, ComputeMatrixFunctor compute_element_matrix)
  {
    ARCANE_CHECK_POINTER(mesh);
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);

    Int32 row_csr_size = m_matrix_row.extent0();
    Int32 col_csr_size = m_matrix_column.extent0();

    auto in_row = ax::viewIn(command, m_matrix_row);
    auto in_out_column = ax::viewInOut(command, m_matrix_column);
    auto in_out_values = ax::viewInOut(command, m_matrix_value);

    UnstructuredMeshConnectivityView connectivity_view;
    connectivity_view.setMesh(mesh);
    auto ncc = connectivity_view.nodeCell();
    auto cnc = connectivity_view.cellNode();

    auto node_dof_cv = m_dofs_on_nodes->nodeDoFConnectivityView();
    ItemGenericInfoListView nodes_infos(mesh->nodeFamily());

    auto in_node_coord = ax::viewIn(command, node_coord);

    {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_ENUMERATE(Node, inode, mesh->allNodes())
      {
        Int32 inode_index = 0;
        for (auto cell : ncc.cells(inode)) {
          if (inode == cnc.nodeId(cell, 1))
            inode_index = 1;
          else if (inode == cnc.nodeId(cell, 2))
            inode_index = 2;
          else if (inode == cnc.nodeId(cell, 3))
            inode_index = 3;
          else
            inode_index = 0;

          auto K_e = compute_element_matrix(cell, cnc, in_node_coord);

          Int32 i = 0;
          Int32 row = node_dof_cv.dofId(inode, 0).localId();
          Int32 begin = in_row[row];
          Int32 end = (row == row_csr_size - 1) ? col_csr_size : in_row[row + 1];
          for (NodeLocalId node2 : cnc.nodes(cell)) {

            if (nodes_infos.isOwn(inode)) {
              Real v = K_e(inode_index, i);
              //Real x = b_matrix[inode_index * 3] * b_matrix[i * 3] + b_matrix[inode_index * 3 + 1] * b_matrix[i * 3 + 1] + b_matrix[inode_index * 3 + 2] * b_matrix[i * 3 + 2];
              //x = x * volume;

              Int32 col = node_dof_cv.dofId(node2, 0).localId();
              for (; begin < end; ++begin) {
                const Int32 currentCol = in_out_column[begin];

                if (currentCol == -1 || currentCol == col) {
                  in_out_column[begin] = col;
                  in_out_values[begin] += v;
                  break;
                }
              }
            }
            i++;
          }
        }
      };
    }
    queue.barrier();
  }

  NumArray<Int32, MDDim1> m_matrix_row;
  NumArray<Int32, MDDim1> m_matrix_column;
  NumArray<Real, MDDim1> m_matrix_value;

 private:

  IMesh* m_mesh;
  Runner* m_runner;
  FemDoFsOnNodes* m_dofs_on_nodes;
};

}; // namespace Arcane::FemUtils

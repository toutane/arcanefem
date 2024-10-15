<?xml version="1.0"?>
<case codename="Poisson" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>L-shape 3D with CSR sparse matrix format Gpu compatible and node wise technique. The result of this test is compared with poisson_test_ref_L-shape_3D.txt</title>
    <timeloop>PoissonLoop</timeloop>
  </arcane>

  <arcane-post-processing>
   <output-period>1</output-period>
   <format name="VtkHdfV2PostProcessor" />
   <output>
     <variable>U</variable>
   </output>
  </arcane-post-processing>

  <meshes>
    <mesh>
      <filename>L-shape-3D.msh</filename>
    </mesh>
  </meshes>

  <fem>
    <nwcsr>true</nwcsr>
    <f>5.5</f>
    <result-file>poisson_test_ref_L-shape_3D.txt</result-file>
    <mesh-type>TETRA4</mesh-type>
    <dirichlet-boundary-condition>
      <surface>bot</surface>
      <value>50.0</value>
    </dirichlet-boundary-condition>
    <dirichlet-boundary-condition>
      <surface>bc</surface>
      <value>10.0</value>
    </dirichlet-boundary-condition>
    <linear-system>
      <solver-backend>petsc</solver-backend>
    </linear-system>
  </fem>
</case>
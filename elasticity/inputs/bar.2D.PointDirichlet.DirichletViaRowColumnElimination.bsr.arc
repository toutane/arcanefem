<?xml version="1.0"?>
<case codename="Elasticity" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Sample</title>
    <timeloop>ElasticityLoop</timeloop>
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
      <filename>meshes/bar.msh</filename>
    </mesh>
  </meshes>

  <fem>
    <E>21.0e5</E>
    <nu>0.28</nu>
    <f>NULL -1.0</f>
    <enforce-Dirichlet-method>RowColumnElimination</enforce-Dirichlet-method>
    <dirichlet-boundary-condition>
      <surface>left</surface>
      <u>0.0 NULL</u>
    </dirichlet-boundary-condition>
    <dirichlet-boundary-condition>
      <surface>right</surface>
      <u>1.0 NULL</u>
    </dirichlet-boundary-condition>
    <dirichlet-point-condition>
      <node>botLeft</node>
      <u>0.0 0.0</u>
    </dirichlet-point-condition>
    <dirichlet-point-condition>
      <node>botRight</node>
      <u>NULL 0.0</u>
    </dirichlet-point-condition>
    <result-file>check/elasticity_point-dirichlet_bar_test_ref.txt</result-file>
    <bsr>true</bsr>
    <linear-system name="HypreLinearSystem">
      <rtol>0.</rtol>
      <atol>1e-15</atol>
      <amg-threshold>0.25</amg-threshold>
    </linear-system>
  </fem>
</case>

<?xml version="1.0"?>
<case codename="Heat" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Sample</title>
    <timeloop>HeatLoop</timeloop>
  </arcane>

  <arcane-post-processing>
   <output-period>1</output-period>
   <output>
     <variable>NodeTemperature</variable>
   </output>
  </arcane-post-processing>

  <meshes>
    <mesh>
      <filename>plate.msh</filename>
    </mesh>
  </meshes>

  <fem>
    <lambda>1.75</lambda>
    <qdot>1e5</qdot>
    <tmax>5.</tmax>
    <dt>0.1</dt>
    <h>0.25</h>
    <Text>25.0</Text>
    <Tinit>0.0</Tinit>
    <dirichlet-boundary-condition>
      <surface>left</surface>
      <value>50.0</value>
    </dirichlet-boundary-condition>
    <dirichlet-boundary-condition>
      <surface>right</surface>
      <value>5.0</value>
    </dirichlet-boundary-condition>
    <neumann-boundary-condition>
      <surface>bottom</surface>
      <value>15.0</value>
    </neumann-boundary-condition>
  </fem>
</case>

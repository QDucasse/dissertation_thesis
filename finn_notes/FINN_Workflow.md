# FINN Workflow and Resources



## Brevitas Network Creation and Training



## ONNX Intermediate Representation

### ONNX Format

ONNX **[1]** is the intermediate representation format used to present Neural Network in a standardized way.  ONNX is composed of:

- A definition of an extensible computation graph model
- Definition of standard data types
- Definitions of built-in operators

The top-level ONNX construct is a `Model`, its purpose is to associate metadata with a graph. Each model must explicitly name the operator sets it relies on for its functionality. A graph is used to describe a side-effect-free computation. It consists of a topologically sorted list of nodes. Each node represents a call to an operator and has zero or more inputs and one or more outputs. A node consists of a name, a list of inputs, a list of outputs, a list of attributes and the actual operation it will perform.



### ONNX extensions for FINN

FINN adds a lot of extensions to the ONNX IR to perform the needed *quantized* operations. This extension of of ONNX is named FINN-ONNX. Among them can be found:

- *Custom quantization annotations*: ONNX does not support datatypes smaller than 8-bit integers, whereas QNNs use the integers down to their ternary and bipolar versions. FINN uses the `quantization_annotation` field in ONNX to annotate the tensors with their FINN `Datatype` information. However, ONNX expects all tensors to use single-precision FP storage. Even a 1-bit value has to be stored as floating point. The FINN compiler will then be responsible to produce a packed representation for the target hardware.
- *Custom operations/nodes*: FINN uses custom operations (`op_type` in ONNX `NodeProto`). These custom nodes are marked with `domain="finn"` in the protobuf to identify as such. These operations can consist of specific operations for low-bit operations or targetting specific hardware backend.
- *Custom ONNX execution flow*: To verify the operation of FINN-ONNX graphs, FINN provide its own ONNX execution flow. This flow is only present to provide a way to check the correctness of models after transformations, it will not perform well for high performance inference.
- *ModelWrapper*: FINN provides a higher top-level class around the top-level `Model` element in ONNX. This wrapper makes it easier to analyze and manipulate ONNX graphs. This is done through the use of helper functions and the granting of full access to the ONNX protobuf representation



Resources can be found in the 0_getting_started notebook:

- Python API **[2]**
- ONNX protobuf description **[1] [3] [4]**



Netron **[5]** is very useful to visualize ONNX models (even FINN-ONNX models).

### ONNX Example

Creating a node using the Python API of ONNX boils down to the selection of a operation type, inputs, outputs and the name:

```python
import onnx

Add1_node = onnx.helper.make_node(
    'Add',
    inputs=['in1', 'in2'],
    outputs=['sum1'],
    name='Add1'
)
```

Once the nodes are created, you can define the actual graph by specifying the nodes in the correct order, the inputs, outputs and information on the size of the different inputs/outputs:

```python
 graph = onnx.helper.make_graph(
        nodes=[
            Add1_node,
            Add2_node,
            Abs_node,
            Add3_node,
            Round_node,
        ],
        name="simple_graph",
        inputs=[in1, in2, in3],
        outputs=[out1],
        value_info=[
            onnx.helper.make_tensor_value_info("sum1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("sum2", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("abs1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("sum3", onnx.TensorProto.FLOAT, [4, 4]),
        ],
    )

```

You can then make and save your model using `make_model()` and `save()`:

```python
onnx_model = onnx.helper.make_model(graph, producer_name="simple-model")
onnx.save(onnx_model, 'simple_model.onnx')
```

Netron can be used to display the final graph:

```python
import netron
netron.start('simple_model.onnx', port=8081, host="0.0.0.0")
```

In order to actually test the output of the network, `onnxruntime` can be used. `InferenceSession()` creates a session for the model and `run()` executes it:

```python
import onnxruntime as rt

sess = rt.InferenceSession(onnx_model.SerializeToString())
output = sess.run(None, input_dict)
```

FINN adds several helper function in its `ModelWrapper`. 

```python
from finn.core.modelwrapper import ModelWrapper
model = ModelWrapper(onnx_model)
```

These functions can help prune your graph (several adding layers put one after the other). These functions also provide interesting insight on the underlying protobuf:

```python
# access the ONNX ModelProto, graph and node list
modelproto = model.model
graphproto = model.graph
nodes = model.graph.node
```

All the helper functions can be found in the source code **[6]** directly.

### Brevitas to ONNX

Once the model is defined and trained in Brevitas, it can be imported into FINN by going through the ONNX representation. This translation is done by using FINN-ONNX that is similar to the *PyTorch* basic export capabilities to ONNX. The differences are:

1. The weight quantization is not exported as part of the graph but separately
2. Special quantization annotations are used to preserve the low-bit quantization
3. Low-bit quantized activation functions are exported as `MultiThreshold` operators

The export can be performed as follows:

```python
import brevitas.onnx as bo
export_onnx_path = "/tmp/model.onnx"
input_shape = (4, 4)
bo.export_finn_onnx(model, input_shape, export_onnx_path)
```

FINN's `ModelWrapper` can now handle the ONNX output:

```python
from finn.core.modelwrapper import ModelWrapper
model = ModelWrapper(export_onnx_path)
```

## FINN Compiler

### FINN Workflow

The main principle of FINN are analysis and transformation passes. FINN performs a *Network preparation* step after obtaining a previously trained *ONNX* representation of a *Brevitas* network. This preparation consists of a succession of transformation passes to "clean up" the network.

FINN can perform transformations such as:

- **Tidy-up**: These transformations are applied in many step of the FINN flow and ensure the information is set and displayed the proper way. Such transformations can consist of: `GiveReadableTensorNames`, `GiveUniqueNodeNames`, `InferDataTypes`, `InferShapes` or `FoldConstants` for example.

- **Streamline**: Streamlining operations aims at the elimination of floating point operation by moving them around collapsing them into one and transforming them into multithresholding nodes. Several transformations are involved here and can be looked upon in this module **[7]** and paper **[8]**. After this streamlining pass, functional verification can be used to simulate the model **[11]**.

- **Convert to HLS Layers**: Pairs of `XNORPopcountMatMul` layers are converted to `StreamingFCLayers` and following `Multithreshold` layers are absorbed in the Matrix-Vector-Activate-Unit (MVAU). The result consists of a mixture of *HLS* and non-*HLS* layers. More information can be found in the module **[9]** and chapter **[10]**.

  

- **Dataflow partition**: The graph is split between *HLS* and non-*HLS* layers. The part consisting of *HLS* layers is processed in the FINN flow while the other one remains. PE and SIMD are set to 1 by default, so the result is a network of only *HLS* layers with maximum folding. The model can now be verified using *cppsim* **[12]**.

The final result consists of HLS layers with the desired folding. These layers can be passed to *Vivado*.

###  FINN Cleanups example

Using the example from earlier,the FINN compiler can now perform cleanups and transformations on the resulting ONNX model imported under the `ModelWrapper`.

```python
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
export_onnx_path_transformed = "/tmp/model-clean.onnx"
model.save(export_onnx_path_transformed)
```



### FINN Step Simulation

Three simulation tools can be used in the FINN process. 

1. Right after the *Brevitas* export, Python can be used to interact with the network and simulate a pass into it.
2. For a model containing several HLS custom operations, *cppsim* can help simulate them. They are based on `finn-hlslib`and C++ code can be generated from the single nodes. They can be compiled and run.
3. When the IP blocks are generated, *PyVerilator* can be used, either node by node or for the whole design. *PyVerilator* gets the generated *verilog* files.



## HLS, Synthesis and Deployment

### Vivado HLS

Once the previous transformations have been performed on the network, FINN will convert each layer of the network to a corresponding *IP Block*. The single conversion of each layer allows a good transparency in this step. The functionality of each *IP Block* can be checked against the behavior of the corresponding *ONNX* node.

The conversion to *Vivado HLS* is split into two parts:

1. `PrepareIP()` transformation which generates *HLS C++* code for the node and a *tcl script* which starts the *HLS synthesis* and exports the design as IP. This transformation has to be given an `fpga_part` (`xc7z020clg484-1` for the Zedboard 7000 for example) as a the target clock frequency.
2. `HLSSynthIP()` transformation which passes the *tcl script* to *Vivado HLS* and performs the actual *IP generation*.

Performing those two different transformations result in the creation of a directory for each layer. This directory contains the project created by *Vivado HLS* into which the *IP Block* is exported. Thresholds and weights are stored in their respective files. A *tcl script* is provided to synthesize this particular *IP Block*. A shell script `ipgen.sh` is generated, it consists of a simple directory change then call to `vivado_hls`. The *tcl script* is similar to the following:

```tcl
set config_proj_name project_StreamingFCLayer_Batch_0
puts "HLS project: $config_proj_name"
set config_hwsrcdir "/tmp/finn_dev_qducasse/code_gen_ipgen_StreamingFCLayer_Batch_0_0vf62b__"
puts "HW source dir: $config_hwsrcdir"
set config_proj_part "xc7z020clg484-1"

set config_bnnlibdir "/workspace/finn-hlslib"

set config_toplevelfxn "StreamingFCLayer_Batch_0"
set config_clkperiod 10

open_project $config_proj_name
add_files $config_hwsrcdir/top_StreamingFCLayer_Batch_0.cpp -cflags "-std=c++0x -I$config_bnnlibdir"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

config_interface -m_axi_addr64
config_rtl -auto_prefix
```

It configures the project (FPGA part, clock, name, etc.). The top level function is set and the design is synthesized with `csynth` before getting exported as an *IP Block*.

##### 

Once all the layers have their *IP Block* counterpart, FINN will stitch them together in a larger *IP* that implements the whole network. The transformation used is `CreateStitchedIP` and has to be used only on HLS nodes (i.e. that went through the previous `HLSSynthIP` transformation). A temporary directory is created and named `vivado_stitch_proj_xxx` where `xxx` is a random sequence. The `.xpr` Vivado project file can be inspected here.



### Hardware generation and Deployment

##### For PYNQ

The standard development flow involves PYNQ, a Python wrapper around the Xilinx ZYNQ-7000. 

1. `MakePYNQProject` does the following actions:
   - collects the list of all the IP directories
   - extracts `HLSCustomOp` instances to get i/o stream widths and ensures i/o is padded to bytes
   - creates a temporary folder for the project
   - writes an `ip_config` tcl script (based on a template)
   - creates a shell script for project creation and synthesis (based on templates)
   - runs the project creation script
2. `SynthPYNQProject` runs the synthesis script
3. `MakePYNQDriver` fills the driver template that will be passed to the board
4. `DeployToPYNQ` does the following actions:
   - creates a directory for deployment files
   - adds the necessary `.bit` and `.hwh` files
   - adds the `driver.py` generated from a template
   - creates the target directory on the board using `sshpass` and `scp`

Now that the driver is installed and the files are passed to the board, we can try to pass test data to the deployed bitfile. To perform a test pass, you need to use the parent graph that contains the non-synthesizable nodes along with a child graph that contains the bulk of the network (later turned into a bitfile). The parent graph has to be loaded and then modify the `StreamingDataflowPartition` to point on the deployed ONNX graph. Finally, `execute_onnx` is called on the parent graph which will then remote execute the bitfile once the `StreamingDataflowPartition` node is reached, grab the results and continue the execution of the last portion of the network.



[1]: https://github.com/onnx/onnx/blob/master/docs/IR.md	"ONNX Documentation"
[2]: https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md	"ONNX Python API"
[3]: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto	"Protobuf Descriptors"
[4]: https://github.com/onnx/onnx/blob/master/docs/Operators.md	"Operators Schemas"
[5]: https://lutzroeder.github.io/netron/	"Netron"
[6]: https://github.com/Xilinx/finn/blob/master/src/finn/core/modelwrapper.py	"ModelWrapper source code"
[7]: https://finn.readthedocs.io/en/latest/source_code/finn.transformation.streamline.html#module-finn.transformation.streamline	"Streamline transformations"
[8]: https://arxiv.org/pdf/1709.04060.pdf	"Streamlined Deployment for QNNs"
[9]: https://finn.readthedocs.io/en/latest/source_code/finn.transformation.fpgadataflow.html#module-finn.transformation.fpgadataflow.convert_to_hls_layers	"Conversion to HLS layers"
[10]: https://finn.readthedocs.io/en/latest/internals.html#mem-mode	"Mem_mode"




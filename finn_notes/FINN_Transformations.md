# FINN Transformations

This file aims at having a close look at the different FINN transformations. Different types of transformations exist from simple operations on node attributes to complicated FPGA processes.

#### Global Mechanism

Every transformation defined by *FINN* is a subclass of the `Transformation` class. This abstract class implements a simple abstract method `apply()` that needs to be overridden by the subclasses. This method is called as soon as the transformation is applied on a `ModelWrapper`.

#### Tidy-up Transformations

The first type of transformations consists of simple operations to tidy up the network as it is:

`finn.transformation.general.py`

- `GiveUniqueNodeNames`: Grants unique names to each node in the graph using enumeration on the different operation types.
- `GiveRandomTensorNames`: Grants random names to tensors in the graph.
- `GiveReadableTensorNames`: Grants human-readable names to all internal tensors (performs the above transformation as its first step). Recommended to apply `GiveUniqueNodeNames` beforehand.
- `ConvertSubToAdd`: Changes all "subtract-a-constant" nodes to "add-a-constant" nodes.
- `ConvertDivToMul`: Changes all "divide-by-constant" nodes to "multiply-by-constant" nodes.



- `InferShapes`: Hides the specific *FINN* operators, runs the *ONNX* shape inference then restores the specific operators.

- `InferDataTypes`: Infers the *FINN* `Datatype` info for all intermediate/output tensors based on inputs and node type. For example, `Sign` nodes will  produce `Datatype.BIPOLAR` outputs.

- `FoldConstants`: Replaces the output of a node with constant-only inputs with a precomputed result. The computation of the result is done through the `onnx_exec` function `execute_node`.

- `DoubleToSingleFloat`: Replaces any `float64` initializer to `float32`.

- `InsertTopK`: Adds a `TopK` node at the network output and replaces the graph output with the index.
- `MoveReshape`: Removes a node that implements a (1, -1) reshape if it is between two `fpgadataflow` nodes.
- `LowerConvsToMatMul`: Replaces `Conv` layers with pairs of `Im2Col-Matmul` layers along with `Transpose` layers to keep the original data layout.
- `BatchNormToAffine`: Replaces any test-time `BatchNorm` layers with `Mul-Add` layers.
- `ConvertBipolarMatMulToXnorPopcount`: Converts `MatMul` nodes with all-bipolar inputs to `XnorPopcountMatMul` and associated correction. This transformation is looking for a succession of `MultiThreshold` and `MatMul` layers.

#### Streamlining Transformations

The `Streamline` transformation consists of a succession of transformations to transform any Quantized Neural Network layer into integer-only operations. The *streamlining* process consists of three steps:

- *Quantization as successive thresholding* : Given a set of threshold values, the successive thresholding function maps any real number `x` to an integer corresponding to the number of thresholds `x` is greater or equal to. This way, any uniform quantifier can be expressed as successive thresholding followed by a linear transformation:
  $$
  Q(x) = a*T(x) + b
  $$


- *Moving and collapsing linear transformations* : All *floating point* linear operations are positioned between the quantized matrix operation and the activation quantization. Any sequence of linear transformation can be collapsed in a single linear transformation.

- *Absorbing linear operations into thresholds* : Updating the threshold with the changes in the linear transformations removes any linear transformation from the graph.

The final operations remaining are **successive thresholding** and **bipolar matrix multiplication**.

*Finn* implements the `Streamline` transformation as follows:

```python
class Streamline(Transformation):
    """Apply the streamlining transform, see arXiv:1709.04060."""

    def apply(self, model):
        streamline_transformations = [
            ConvertSubToAdd(),
            ConvertDivToMul(),
            BatchNormToAffine(),
            ConvertSignToThres(),
            MoveAddPastMul(),
            MoveScalarAddPastMatMul(),
            MoveScalarAddPastConv(),
            MoveScalarMulPastMatMul(),
            MoveScalarMulPastConv(),
            MoveAddPastMul(),
            CollapseRepeatedAdd(),
            CollapseRepeatedMul(),
            AbsorbAddIntoMultiThreshold(),
            FactorOutMulSignMagnitude(),
            AbsorbMulIntoMultiThreshold(),
            Absorb1BitMulIntoMatMul(),
            Absorb1BitMulIntoConv(),
            RoundAndClipThresholds(),
        ]
        for trn in streamline_transformations:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(InferDataTypes())
        return (model, False)
```     

The `Streamline` operations are divided in:

- Basic conversions (`ConvertSubToAdd`, `ConvertDivToMul`, `BatchNormToAffine`) explained earlier.
- Conversion of the `Sign` nodes to an addition of a zero threshold in a `MultiThreshold` layer.
- Reordering of the nodes with `MoveScalarAddPastMatMul`, `MoveScalarAddPastConv`, `MoveScalarMulPastMatMul`, etc. These transformations unites the different linear transformations.
- Collapse of successive linear transformations with `CollapseRepeatedAdd` and `CollapseRepeatedMul`.
- Absorption of linear transformations into other layers (`MatMul`, `Conv` or `Thresholds`). This is done through `AbsorbAddIntoMultiThreshold`, `FactorOutMulSignMagnitude`, `FactorOutMulSignMagnitude`.
- Threshold values are rounded to the nearest integer to perform the layer pass on integers only. This is done with `RoundAndClipThresholds`.

Note that between each of these transformation, a succession of `GiveUniqueNodeNames`, `GiveReadableTensorNames` and `InferDataTypes` is used.

#### FPGA Flow Transformations

The final transformations are used in the `FPGADataFlow`. They correspond to the preparation, annotation, HLS conversion, synthesis, IP creation and even integration and deployment. Most of these transformation use the `CustomOp` defined by *FINN* to translate them directly in *HLS* using  the `finn-hls` library.



#### Custom Operations IP Generation

The *IP* Generation of the different custom operators is defined using the `HLSCustomOp` class.

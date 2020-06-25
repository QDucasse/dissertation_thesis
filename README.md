# Dissertation Thesis
## Author:
- Quentin DUCASSE (qd14@hw.ac.uk)

## Subject

Many embedded systems are based on the use of generic or dedicated processors. These processors have hardware calculation units of variable precision (examples: ALU or FPU 8, 16, 32, 64 bits). Some processing algorithms are designed to perform calculations based on a given accuracy. However, the use of calculations with lower accuracy allows, in some cases, an acceleration of these same calculations while maintaining sufficient accuracy for the desired functionality. This acceleration can have several benefits:
- A reduction in the cost of the calculating component: a less powerful component is generally cheaper,
- Better energy efficiency: a less powerful component or with a lower frequency/ voltage consumes less energy, and dissipates less heat,
- Better performance: reduced computing time allows you to work on larger datasets or perform other calculations at constant cost.

The objective of the internship is to analyse a reference application, determine the parts of code where accuracy can be reduced, and then implement it on one or more hardware architectures to verify that the accuracy is sufficient and the performance gain makes sense. The trainee will have to appropriate the existing digital precision analysis tools, then implement these tools on an algorithm, to analyse on each code portion, the loss of precision and its level of acceptability. This step will be performed on a sequential code in C or C++, in order to allow the use of tools that do not support the parallelised code.  This first stage will rely on LLVM or Clang.

In a second step, the application will be based on an architecture with computing unit(s) of the desired precision (GPU or MPPA). Depending on the profiling and the results of the analysis, the application code will be modified to take advantage of the targeted computing units (meaning, exploiting an extended ISA, with extra instructions that invoke hardware primitives). Finally, the global architecture (processor + hardware primitives) will be built automatically, and the impact of a malicious alteration of synthesis scripts will be illustrated to motivate the need for cyber-protection when designing such a soc.

## Structure of the literature review
Please look inside the research_report folder

## Articles read and annotated

- [X] 1967-Moler: Iterative Refinement in Floating Point
- [X] 1989-Imel: Mixed-precision Operations Floating Point Operations from a Single Instruction Opcode
- [X] 2000-Tong: Reducing Power by Optimizing the Necessary Precision/Range of Floating-Point Arithmetic
- [X] 2006-Moore: Cramming More Components onto Integrated Circuits
- [X] 2006-Strzodka: Pipelined mixed-precision Algorithms on FPGAs for Fast and Accurate PDE Solvers from Low-precision Components
- [X] 2007-Goddeke: Performance and Accuracy of hardware-oriented native-, emulated- and mixed-precision solvers in FEM simulations
- [X] 2007-Yates: Fixed-point arithmetic: an introduction
- [X] 2008-Sun: High Performance Mixed-precision Linear Solver for FPGAs
- [X] 2009-Baboulin: Accelerating Scientific Computations with Mixed-Precision Algorithms
- [X] 2010-Clark: Solving lattice QCD systems of equations using mixed-precision solvers on GPUs
- [X] 2012-Chow: A Mixed-precision Monte-Carlo methodology for Reconfigurable Accelerators Systems
- [X] 2013-Darulova: Synthesis of fixed-point programs
- [X] 2013-LeGrand: SPFP: Speed without compromise - A Mixed-precision Model for GPU accelerated Molecular Dynamic Simulations
- [X] 2013-Rubio: Precimonius, tuning assistant for Floating Point programs
- [X] 2014-Horrowitz: Computing's Energy Problem (and what we can do about it)
- [X] 2014-XuanSang: From Smalltalk to Silicon: a methodology to turn Smalltalk code into FPGA
- [X] 2015-Nips: High-Performance Hardware for Machine Learning
- [X] 2016-Courbariaux: Binary-net: Training deep neural networks with weights and activations constrained to +1 or -1
- [X] 2016-Hubara: s neural networks: Training neural networks with low precision weights and activations
- [X] 2016-Park: FPGA based implementation of deep neural networks using on-chip memory only
- [X] 2016-Qiu: - [ ] Going Deeper with Embedded FPGA Platform for Convolutional Neural Network
- [X] 2016-Zhao: F-CNN: An FPGA-based Framework for Training Convolutional Neural Networks
- [ ] 2017-Liang: FP-BNN: Binarised neural network on FPGA
- [X] 2017-Micikevicius: Mixed-Precision Training
- [X] 2017-Umuroglu: FINN: A framework for fast, scalable binarised neural network inference
- [X] 2017-Xilinx: Reduce Power and Cost by Converting from Floating Point to Fixed Point
- [X] 2018-Abdelouahab: Accelerating CNN inference on FPGAs: A Survey
- [X] 2018-Blott: FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of quantised Neural Networks
- [X] 2018-Colangelo: Exploration of Low Numeric Precision Deep Learning Inference Using Intel FPGAs
- [X] 2018-Darulova: Sound mixed-precision with rewriting
- [X] 2018-Haidar: Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed Up Mixed-Precision Iterative Refinement Solvers
- [X] 2018-Jia: Highly Scalable Deep Learning Training System With Mixed-Precision: Training ImageNet in Four Minutes
- [ ] 2018-Joubert: Attacking the opioid epidemic: Determining the Epistatic and Pleiotropic Genetic Architectures for Chronic Pain and Opioid Addiction
- [X] 2018-Kurth: Exascale deep learning for climate analysis
- [X] 2018-LeGallo: Mixed-precision in-memory computing
- [X] 2018-Narang: Mixed-precision training
- [X] 2018-Rybalkin: FINN-L: Library Extensions and Design Trade-off Analysis for Variable Precision LSTM Networks on FPGAs
- [ ] 2019-Ding: REQ-YOLO: A Resource-Aware, Efficient Quantisation Framework for Object Detection on FPGAs
- [X] 2019-Jahanshahi: TinyCNN: A Tiny Modular CNN Accelerator for Embedded FPGA
- [ ] 2019-Wang: Deep neural network approximation for custom hardware: Where we’ve been, where we’re going
- [X] 2019-Zhao: Automatic generation of multi-precision multi-arithmetic CNN accelerators for FPGAs
- [X] 2020-Bacchus: Accuracy, Training Time and Hardware Efficiency Trade-Offs for Quantized Neural Networks on FPGAs
- [X] 2020-Radu: Performance Aware Convolutional Neural Network Channel Pruning for Embedded GPUs

## Important sites
Precision analysis:
- Precision analysis tools: https://fpbench.org/community.html

Floating-point and fixed-point arithmetic:
- Single and double precision summary: https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/
- Floating-point standards: https://www.doc.ic.ac.uk/~eedwards/compsys/float/
- Computerphile, Floating point representations:
  - Part1: https://www.youtube.com/watch?v=PZRI1IfStY0
  - Part2: https://www.youtube.com/watch?v=f4ekifyijIg
  - Part3: https://www.youtube.com/watch?v=782QWNOD_Z0

Mixed-precision applications:
- Climate analytics: https://dl.acm.org/doi/10.1109/SC.2018.00054
- Opioid addiction: https://dl.acm.org/doi/10.5555/3291656.3291732
- Nuclear fusion simulation: https://www.ornl.gov/news/david-green-teaming-solve-questions-fusion

Deep-learning:
- Convolutional Neural Networks: http://cs231n.github.io/convolutional-networks/
- CNN vocabulary: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
- Machine Learning on FPGAs: Neural Networks: https://www.youtube.com/watch?v=3iCifD8gZ0Q
- Machine Learning For Embedded Applications on FPGAs: https://www.youtube.com/watch?v=t520cNlT7bU



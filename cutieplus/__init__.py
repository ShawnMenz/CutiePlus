"""
CutiePlus annotation service: the inference engine shared by the local mode and the GPU worker
(engine), GPU resource management (gpu), the backends the labeling window talks to (backends),
and the coordinator and GPU worker processes of the shared service (coordinator, worker).

Nothing here imports Qt. Importing cutieplus never initializes CUDA; only the code that runs
models on a CUDA device does.
"""

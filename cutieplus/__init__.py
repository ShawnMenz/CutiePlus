"""
CutiePlus: the inference engine of the labeling window (engine) and the backend the window talks
to (backends).

Nothing here imports Qt. Importing cutieplus never initializes CUDA; only the code that runs
models on a CUDA device does.
"""

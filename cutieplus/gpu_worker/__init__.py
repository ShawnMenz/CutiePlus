"""
The GPU worker: the one process of the cooperation module that loads models onto the GPU and runs
them, for every window (python -m cutieplus.gpu_worker, the cutieplus-gpu-worker service). It takes
the GPU lock before it touches CUDA and keeps it while it runs, then connects to the cooperation
service and carries out the commands it sends, one at a time. It has no say over users, roles or
SSH access: it only infers.
"""

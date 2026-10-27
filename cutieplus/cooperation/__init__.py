"""
The cooperation module: application users and roles, managed SSH and VNC access, and the one GPU
queue every labeling window shares (docs/COOPERATION.md).

    service     the cooperation service (cutieplus.cooperation.service), run as the protected
                cutieplus account: sessions, permissions, SQLite, users, keys, desktops, jobs
    helper      the root helper (cutieplus.cooperation.helper): a few fixed system operations
    GPU worker  cutieplus.gpu_worker, the only process that loads models on the GPU
    client      what a window uses to talk to the service (cutieplus.cooperation.client)

Nothing here imports torch or Qt, so the service and the helper run without them.
"""

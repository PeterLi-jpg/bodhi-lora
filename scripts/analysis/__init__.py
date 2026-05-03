"""Post-hoc analyses on the Stage-4/5 eval JSONs produced by tpu/launch_5seeds_tunix.sh.

These run on the local dashboard machine (or any box with the response JSONs)
*after* v37 finishes — no additional TPU compute needed for calibration and
theme-stratified analyses; the cross-grader and OOD probe scripts do require
TPU/GPU and are designed to run on one of the v37 VMs before its EXIT trap
deletes it.

Each module is independent and can be invoked as ``python -m scripts.analysis.<name>``.
"""

# Runtime scripts

These scripts implement the portable Enroot launch path. Start from the release
root with `bash run.sh`; see the root `README.md` for configuration, monitoring,
and output locations.

`training.sh` is the canonical experiment specification. It uses eight-turn
medium boards, 9,500 training seeds, 500 held-out validation seeds, 16 samples
per prompt, and an eight-GPU disaggregated trainer/rollout layout.

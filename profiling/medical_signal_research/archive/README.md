# Historical experiments

`results/` contains earlier synthetic, normalized-EDF, processed-cache/training,
reproducibility and exploratory native-format runs, including profiler traces.
`docs/` contains their original documentation. These are intermediate findings;
the current native-format conclusion is linked from the parent research index.

No measurement or trace was deleted or rewritten. `inventory.json` records each
moved file's original path, current path relative to the repository root, and
SHA256 digest at relocation. The five current research documents were later
updated for the new layout; their original digests remain in the inventory.
Paths embedded in historical JSON and historical commands refer
to the original layout; consult the inventory to locate those artifacts now.
Final native measurements live in the sibling `../results/` directory.

`source_before_cleanup.tar.gz` preserves the fourteen experiment Python modules
immediately before helper extraction, including the source corresponding to the
final native measurements. Extract it into a separate directory to inspect or
reproduce the measured implementation. It contains no source datasets or environment.

The old Python module paths remain available for existing callers and commands.
Their shared helpers are now re-exported from `profiling.signal_utils`.

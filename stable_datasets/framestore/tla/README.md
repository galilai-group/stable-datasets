# TLA+ models of the store converter

`./check.sh` model-checks every config with TLC (Java 11+; `TLA2TOOLS=/path/to/tla2tools.jar`, default
`~/.cache/tla/tla2tools.jar`). Configs named `_old`, `_oldwriter` or `_lostjob` model earlier code and are expected to
fail with the counterexample shown.

| Model | What it covers | Checked |
|---|---|---|
| `CensusGate` | census: main thread scans, pool probes, byte gate, collect-oldest | gate bound; never pops an empty deque; always finishes |
| `ConvertPipeline` | reader thread, writer/lance, pool, GPU verifier and CPU helpers, byte gate | gate bound; lance gets batches in order and only verified ones; fragments committed in order and whole; always finishes (done, bad frame, or decoder error) |
| `ConvertResume` | in-place conversion crashing after any step and run again | no fragment lost; never stuck; ends with every fragment converted once and nothing left over |

Configs that fail (the bugs each one reproduces):

- `CensusGate_old`: a blocking acquire on the only thread that releases. It deadlocked on a pod.
- `ConvertPipeline_oldwriter`: the writer waited on the reader's queue while holding the bytes the reader needed. It deadlocked on a pod.
- `ConvertPipeline_lostjob`: a decoder that raised left its job claimed and never finished, so the writer waited forever.
- `ConvertResume_old` reproduces three failures:
  - A crash during the first write left a store directory with nothing committed, which the next run could not open.
  - A crash during the source's removal left a directory the next run could not open.
  - A crash after the swap, before the progress file was removed, caused data loss. The next run dropped the converted store's fragments by id, because those ids are the same as the source's.

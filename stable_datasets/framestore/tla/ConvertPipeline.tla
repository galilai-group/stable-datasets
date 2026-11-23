--------------------------------- MODULE ConvertPipeline ---------------------------------
(* framestore/convert.py, Converter.stream: the conversion pipeline.

   - The reader thread scans each fragment's batches. It acquires each batch's cost from a ByteGate (blocking),
     then queues the batch to the writer. After a fragment it queues an end marker, and after the last fragment a
     sentinel ("none").
   - The writer is the main thread, and also lance's consumer. It takes items from the reader's queue (with a
     timeout while batches are pending: "idle") and submits each batch to the pool for conversion. It then pops
     the oldest pending batch once it is converted, or once Depth batches are pending, or when idle or finished.
     It waits for that batch's conversion and its verification, and hands the batch to lance. When lance asks
     for the next batch, the writer releases the batch's cost. An end marker makes lance commit its fragment.
   - Each converted batch is verified. On the GPU thread, chunks are taken from a queue, and CPU helpers on the
     pool join when the writer stalls on verification. Without a GPU, verification is a pool task. A
     verification finds the batch identical ("ok"), finds a frame that differs ("bad"), or its decoder raises.

   ByteGate.acquire(n) blocks while used > 0 and used + n > limit: an item larger than the limit passes alone.

   Checked: the gate's bound, lance gets every fragment's batches in order and only verified ones, fragments
   are committed in order, and the run always ends (finished, or stopped on a bad frame or decoder error). *)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    NF, NB,        \* fragments, batches per fragment
    Costs,         \* gate costs a batch may have (include one above Limit)
    Limit,         \* the gate's limit
    Depth,         \* pending batches before the writer waits on the oldest
    Threads,       \* CPU helpers the writer may ask for at once
    GPU,           \* verification on the GPU thread (else on the pool)
    CpuVerify,     \* CPU threads help the GPU when the writer stalls on verification
    BadFrames,     \* verification may find a frame that decodes differently
    Errors,        \* a decoder may raise
    BlockingGet,   \* the old writer: waits on the reader's queue even with batches pending
    ErrorsFinish   \* a decoder error finishes its job as failed (else the job stays claimed, unfinished)

Batches == (1..NF) \X (1..NB)
NoId == <<0, 0>>
None == [k |-> "none", f |-> 0, id |-> NoId, c |-> 0]
Final == {"ok", "bad", "failed"}  \* job states the writer stops waiting at
Outcomes == {"ok"} \cup (IF BadFrames THEN {"bad"} ELSE {}) \cup (IF Errors THEN {"raise"} ELSE {})
Result(o) == IF o = "raise" THEN (IF ErrorsFinish THEN "failed" ELSE "lost") ELSE o

RECURSIVE SumC(_)
SumC(S) == IF S = {} THEN 0 ELSE LET x == CHOOSE x \in S : TRUE IN x[2] + SumC(S \ {x})

VARIABLES
    rpc, rf, ri, rc,         \* reader: pc, fragment and batch it is at, cost it is acquiring
    used, held,              \* gate: bytes acquired, and <<batch, cost>> of each holder
    src,                     \* the reader's queue to the writer
    wpc, witem, pend, cur,   \* writer: pc, the last item it got, pending items, the item it is on
    sub, conv, cpuq,         \* pool: conversions submitted and finished, verification tasks (no GPU)
    job, vq, gpu, helpers,   \* verification: per batch, GPU queue, GPU thread, helpers submitted
    written, committed       \* lance: batch numbers received per fragment, fragments committed

rvars == <<rpc, rf, ri, rc>>
gvars == <<used, held>>
wvars == <<wpc, witem, pend, cur>>
pvars == <<sub, conv, cpuq>>
vvars == <<job, vq, gpu, helpers>>
lvars == <<written, committed>>
vars == <<rvars, gvars, src, wvars, pvars, vvars, lvars>>

Item == [k : {"batch", "end", "none"}, f : 0..NF, id : Batches \cup {NoId}, c : Costs \cup {0}]

TypeOK ==
    /\ rpc \in {"next", "acq", "done"} /\ rf \in 1..NF + 1 /\ ri \in 1..NB + 1
    /\ used \in Nat /\ held \subseteq (Batches \X Costs)
    /\ src \in Seq(Item) /\ pend \in Seq(Item) /\ cur \in Item
    /\ wpc \in {"get", "drain", "waitconv", "waitver", "commit", "done", "error"}
    /\ sub \subseteq Batches /\ conv \subseteq sub /\ cpuq \subseteq conv
    /\ job \in [Batches -> {"none", "queued", "gpu", "ok", "bad", "failed", "lost"}]
    /\ gpu.st \in {"idle", "busy", "dead"} /\ helpers \in 0..Threads

Init ==
    /\ rpc = "next" /\ rf = 1 /\ ri = 1 /\ rc = 0
    /\ used = 0 /\ held = {}
    /\ src = <<>>
    /\ wpc = "get" /\ witem = "none" /\ pend = <<>> /\ cur = None
    /\ sub = {} /\ conv = {} /\ cpuq = {}
    /\ job = [b \in Batches |-> "none"] /\ vq = <<>> /\ gpu = [st |-> "idle", id |-> NoId] /\ helpers = 0
    /\ written = [f \in 1..NF |-> <<>>] /\ committed = <<>>

(* ---- reader thread ---- *)
ReadNext ==
    /\ rpc = "next"
    /\ IF rf > NF THEN
         /\ src' = Append(src, None) /\ rpc' = "done" /\ UNCHANGED <<rf, ri, rc>>
       ELSE IF ri > NB THEN  \* end of the fragment: no bytes
         /\ src' = Append(src, [k |-> "end", f |-> rf, id |-> NoId, c |-> 0])
         /\ rf' = rf + 1 /\ ri' = 1 /\ UNCHANGED <<rpc, rc>>
       ELSE
         /\ \E c \in Costs : rc' = c
         /\ rpc' = "acq" /\ UNCHANGED <<src, rf, ri>>
    /\ UNCHANGED <<gvars, wvars, pvars, vvars, lvars>>

ReadAcquire ==  \* gate.acquire(cost), then src.put(batch)
    /\ rpc = "acq"
    /\ used = 0 \/ used + rc <= Limit
    /\ used' = used + rc /\ held' = held \cup {<<<<rf, ri>>, rc>>}
    /\ src' = Append(src, [k |-> "batch", f |-> rf, id |-> <<rf, ri>>, c |-> rc])
    /\ ri' = ri + 1 /\ rpc' = "next" /\ UNCHANGED <<rf, rc>>
    /\ UNCHANGED <<wvars, pvars, vvars, lvars>>

(* ---- writer (main thread) ---- *)
WGet ==
    /\ wpc = "get"
    /\ \/ /\ src # <<>>
          /\ LET it == Head(src) IN
               /\ src' = Tail(src) /\ witem' = it.k
               /\ pend' = IF it.k = "none" THEN pend ELSE Append(pend, it)
               /\ sub' = IF it.k = "batch" THEN sub \cup {it.id} ELSE sub
       \/ /\ src = <<>> /\ pend # <<>> /\ ~BlockingGet  \* src.get(timeout=0.01) -> "idle"
          /\ witem' = "idle" /\ UNCHANGED <<src, pend, sub>>
    /\ wpc' = "drain"
    /\ UNCHANGED <<rvars, gvars, cur, conv, cpuq, vvars, lvars>>

Popping ==
    /\ pend # <<>>
    /\ \/ witem \in {"none", "idle"}
       \/ Len(pend) >= Depth
       \/ Head(pend).k = "end"
       \/ Head(pend).id \in conv

WDrain ==
    /\ wpc = "drain"
    /\ IF Popping THEN
         /\ cur' = Head(pend) /\ pend' = Tail(pend)
         /\ wpc' = IF Head(pend).k = "end" THEN "commit" ELSE "waitconv"
       ELSE
         /\ wpc' = IF witem = "none" THEN "done" ELSE "get"
         /\ UNCHANGED <<cur, pend>>
    /\ UNCHANGED <<rvars, gvars, src, witem, pvars, vvars, lvars>>

WaitConv ==  \* fut.result()
    /\ wpc = "waitconv" /\ cur.id \in conv
    /\ wpc' = "waitver"
    /\ UNCHANGED <<rvars, gvars, src, witem, pend, cur, pvars, vvars, lvars>>

Stall ==  \* the writer has waited on verification for a while: CPU helpers join the GPU
    /\ wpc = "waitver" /\ job[cur.id] \notin Final
    /\ GPU /\ CpuVerify
    /\ helpers < Threads /\ helpers < Len(vq)
    /\ helpers' = helpers + 1
    /\ UNCHANGED <<rvars, gvars, src, wvars, pvars, job, vq, gpu, lvars>>

WYield ==  \* verified: lance takes the batch, asks for the next, and the writer releases its bytes
    /\ wpc = "waitver" /\ job[cur.id] \in Final
    /\ IF job[cur.id] = "ok" THEN
         /\ written' = [written EXCEPT ![cur.f] = Append(@, cur.id[2])]
         /\ used' = used - cur.c /\ held' = held \ {<<cur.id, cur.c>>}
         /\ wpc' = "drain"
       ELSE  \* SystemExit: nothing of this fragment is committed
         /\ wpc' = "error" /\ UNCHANGED <<written, used, held>>
    /\ UNCHANGED <<rvars, src, witem, pend, cur, pvars, vvars, committed>>

WCommit ==  \* the fragment's end: lance commits it; the main thread drops it from the source and goes on
    /\ wpc = "commit"
    /\ committed' = Append(committed, cur.f) /\ wpc' = "drain"
    /\ UNCHANGED <<rvars, gvars, src, witem, pend, cur, pvars, vvars, written>>

(* ---- pool: conversion, and verification without a GPU ---- *)
Convert ==
    /\ \E b \in sub \ conv :
         /\ conv' = conv \cup {b}
         /\ job' = [job EXCEPT ![b] = "queued"]
         /\ IF GPU THEN vq' = Append(vq, b) /\ UNCHANGED cpuq
                   ELSE cpuq' = cpuq \cup {b} /\ UNCHANGED vq
    /\ UNCHANGED <<rvars, gvars, src, wvars, sub, gpu, helpers, lvars>>

CpuTask ==
    /\ \E b \in cpuq :
         /\ cpuq' = cpuq \ {b}
         /\ \E o \in Outcomes : job' = [job EXCEPT ![b] = Result(o)]
    /\ UNCHANGED <<rvars, gvars, src, wvars, sub, conv, vq, gpu, helpers, lvars>>

(* ---- GPU thread and its CPU helpers ---- *)
GpuTake ==
    /\ gpu.st = "idle" /\ vq # <<>>
    /\ gpu' = [st |-> "busy", id |-> Head(vq)] /\ vq' = Tail(vq)
    /\ job' = [job EXCEPT ![Head(vq)] = "gpu"]
    /\ UNCHANGED <<rvars, gvars, src, wvars, pvars, helpers, lvars>>

GpuRun ==  \* a GPU decoder error falls back to the CPU decoder; "raise" is the CPU decoder raising too
    /\ gpu.st = "busy"
    /\ \E o \in Outcomes :
         /\ job' = [job EXCEPT ![gpu.id] = Result(o)]
         /\ gpu' = IF Result(o) = "lost" THEN [gpu EXCEPT !.st = "dead"] ELSE [st |-> "idle", id |-> NoId]
    /\ UNCHANGED <<rvars, gvars, src, wvars, pvars, vq, helpers, lvars>>

Help ==  \* a helper takes the oldest chunk the GPU has not started, if any
    /\ helpers > 0
    /\ helpers' = helpers - 1
    /\ IF vq = <<>> THEN UNCHANGED <<vq, job>>
       ELSE /\ vq' = Tail(vq)
            /\ \E o \in Outcomes : job' = [job EXCEPT ![Head(vq)] = Result(o)]
    /\ UNCHANGED <<rvars, gvars, src, wvars, pvars, gpu, lvars>>

Stop == wpc \in {"done", "error"} /\ UNCHANGED vars  \* the process exits

Next ==
    \/ ReadNext \/ ReadAcquire
    \/ WGet \/ WDrain \/ WaitConv \/ Stall \/ WYield \/ WCommit
    \/ Convert \/ CpuTask
    \/ GpuTake \/ GpuRun \/ Help
    \/ Stop

Fair ==
    /\ WF_vars(ReadNext) /\ WF_vars(ReadAcquire)
    /\ WF_vars(WGet) /\ WF_vars(WDrain) /\ WF_vars(WaitConv) /\ WF_vars(Stall) /\ WF_vars(WYield)
    /\ WF_vars(WCommit)
    /\ WF_vars(Convert) /\ WF_vars(CpuTask) /\ WF_vars(GpuTake) /\ WF_vars(GpuRun) /\ WF_vars(Help)

Spec == Init /\ [][Next]_vars /\ Fair

(* ---- properties ---- *)
GateBound ==  \* the gate counts exactly its holders, and exceeds its limit only with one holder
    /\ used = SumC(held)
    /\ used <= Limit \/ Cardinality(held) = 1

InOrder ==  \* lance gets each fragment's batches in order, and commits fragments in order
    /\ \A f \in 1..NF : written[f] = [n \in 1..Len(written[f]) |-> n]
    /\ committed = [n \in 1..Len(committed) |-> n]

Verified ==  \* only verified batches reach lance; a committed fragment has all of its batches
    /\ \A f \in 1..NF : \A n \in 1..Len(written[f]) : job[<<f, n>>] = "ok"
    /\ \A n \in 1..Len(committed) : Len(written[committed[n]]) = NB

DoneComplete == wpc = "done" => Len(committed) = NF /\ held = {} /\ used = 0

Finishes == <>(wpc \in {"done", "error"})
=============================================================================================

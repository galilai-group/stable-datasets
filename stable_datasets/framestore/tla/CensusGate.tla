----------------------------------- MODULE CensusGate -----------------------------------
(* framestore/convert.py, census: the main thread scans batches and submits each to the pool, keeping a FIFO of
   futures. It releases a batch's gate cost when it collects that batch's result. The main thread is the only
   one that acquires or releases.

   Per batch:
   1. while not gate.try_acquire(cost): collect the oldest.
   2. Submit the batch.
   3. Collect the oldest while 2 * W futures are pending or the oldest is done.
   At the end, collect all.

   BlockingAcquire models the old code, gate.acquire(cost), which waited for a release only this thread could
   make. *)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS NB, Costs, Limit, W, BlockingAcquire

VARIABLES mpc, nxt, c, futs, fin, used
vars == <<mpc, nxt, c, futs, fin, used>>

RECURSIVE SumF(_)
SumF(s) == IF s = <<>> THEN 0 ELSE Head(s)[2] + SumF(Tail(s))

Fits == used = 0 \/ used + c <= Limit

Init == mpc = "read" /\ nxt = 1 /\ c = 0 /\ futs = <<>> /\ fin = {} /\ used = 0

Collect ==  \* fut.result() of the oldest, then gate.release
    /\ futs # <<>> /\ Head(futs)[1] \in fin
    /\ used' = used - Head(futs)[2] /\ futs' = Tail(futs)

Read ==
    /\ mpc = "read"
    /\ IF nxt > NB THEN mpc' = "all" /\ UNCHANGED c
       ELSE \E x \in Costs : c' = x /\ mpc' = "try"
    /\ UNCHANGED <<nxt, futs, fin, used>>

Try ==
    /\ mpc = "try"
    /\ IF Fits THEN
         /\ used' = used + c /\ futs' = Append(futs, <<nxt, c>>) /\ nxt' = nxt + 1 /\ mpc' = "after"
       ELSE
         /\ ~BlockingAcquire
         /\ Collect /\ UNCHANGED <<nxt, mpc>>
    /\ UNCHANGED <<c, fin>>

After ==
    /\ mpc = "after"
    /\ IF Len(futs) >= 2 * W \/ (futs # <<>> /\ Head(futs)[1] \in fin)
         THEN Collect /\ UNCHANGED mpc
         ELSE mpc' = "read" /\ UNCHANGED <<futs, used>>
    /\ UNCHANGED <<nxt, c, fin>>

All ==
    /\ mpc = "all"
    /\ IF futs = <<>> THEN mpc' = "done" /\ UNCHANGED <<futs, used>>
       ELSE Collect /\ UNCHANGED mpc
    /\ UNCHANGED <<nxt, c, fin>>

Pool ==
    /\ \E i \in 1..nxt - 1 : i \notin fin /\ fin' = fin \cup {i}
    /\ UNCHANGED <<mpc, nxt, c, futs, used>>

Stop == mpc = "done" /\ UNCHANGED vars

Next == Read \/ Try \/ After \/ All \/ Pool \/ Stop

Spec == Init /\ [][Next]_vars /\ WF_vars(Read) /\ WF_vars(Try) /\ WF_vars(After) /\ WF_vars(All) /\ WF_vars(Pool)

GateBound == used = SumF(futs) /\ (used <= Limit \/ Len(futs) = 1)

CollectsSomething == mpc = "try" /\ ~Fits => futs # <<>>  \* futs.popleft() never runs on an empty deque

Finishes == <>(mpc = "done")
=============================================================================================

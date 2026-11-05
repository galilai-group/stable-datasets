---------------------------------- MODULE ConvertResume ----------------------------------
(* framestore/convert.py run in place, crashing after any step and run again.

   What survives a crash:
   - the source store;
   - the converted store being written (<src>.converting, "tmp");
   - the progress file (written atomically: write a temporary file, then rename).

   Lance commits are atomic. A store directory's removal (rmtree) is not. lance.write_dataset in create mode makes
   the directory before it commits anything.

   Fragments get lance ids 1, 2, ... in the order they are appended, so the converted store's ids are the same as
   the source's. Each fragment is one unit of rows.

   Fixed = FALSE is the code before this model; Fixed = TRUE is the code after:
   - The progress file is marked "finishing" before the source is removed.
   - A run that finds the mark completes the swap and nothing else.
   - A converted store with nothing committed is removed and started again.

   Checked: no fragment is ever lost; a run never stops on a state it cannot continue from; the run ends with the
   source replaced by every fragment converted, once each. *)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS N, MaxCrashes, Fixed

Orig == 1..N
Absent == [st |-> "absent", fr |-> {}]
Store(fr) == [st |-> "ds", fr |-> fr]
IsDS(s) == s.st = "ds"
Range(q) == {q[i] : i \in DOMAIN q}
NoProg == [exists |-> FALSE, done |-> <<>>, rows |-> 0, finishing |-> FALSE]
NewProg == [NoProg EXCEPT !.exists = TRUE]

AsSeq(S) ==  \* fragments in id order (get_fragments)
    [n \in 1..Cardinality(S) |-> CHOOSE e \in S : Cardinality({x \in S : x.id <= e.id}) = n]

VARIABLES
    src, tmp, prog,   \* on disk
    pc, p, frags, k,  \* this run: pc, its progress record, the source's fragments when it started, drop index
    crashes, outcome

vars == <<src, tmp, prog, pc, p, frags, k, crashes, outcome>>

Todo == SelectSeq(frags, LAMBDA e : e.id \notin Range(p.done))

Init ==
    /\ src = Store({[id |-> i, orig |-> i, conv |-> FALSE] : i \in Orig})
    /\ tmp = Absent /\ prog = NoProg
    /\ pc = "start" /\ p = NoProg /\ frags = <<>> /\ k = 1
    /\ crashes = 0 /\ outcome = "running"

Stuck(why) == outcome' = why /\ pc' = "halt" /\ UNCHANGED <<src, tmp, prog, p, frags, k, crashes>>
Go(to) == pc' = to /\ UNCHANGED <<src, tmp, prog, p, frags, k, crashes, outcome>>

Drop(fid, next) ==  \* lance Delete of a fragment of whatever store is at the source's path
    IF ~IsDS(src) THEN Stuck("stuck: drop cannot open the source")
    ELSE /\ src' = Store({e \in src.fr : e.id # fid}) /\ pc' = next
         /\ UNCHANGED <<tmp, prog, p, frags, k, crashes, outcome>>

Start ==
    /\ pc = "start"
    /\ IF Fixed /\ prog.exists /\ prog.finishing THEN Go(IF tmp.st = "absent" THEN "rmprog" ELSE "rm1")
       ELSE IF ~Fixed /\ prog.exists /\ src.st = "absent" /\ tmp.st # "absent" THEN Go("rename")
       ELSE IF ~IsDS(src) THEN Stuck("stuck: cannot open the source")
       ELSE IF ~prog.exists /\ tmp.st # "absent" THEN Stuck("stuck: converted store without a progress file")
       ELSE /\ p' = IF prog.exists THEN prog ELSE NewProg
            /\ frags' = AsSeq(src.fr) /\ pc' = "opentmp"
            /\ UNCHANGED <<src, tmp, prog, k, crashes, outcome>>

OpenTmp ==  \* a commit that the progress file missed is the next fragment's
    /\ pc = "opentmp"
    /\ IF tmp.st = "absent" THEN Go("save")
       ELSE IF ~IsDS(tmp) THEN
         IF Fixed THEN /\ tmp' = Absent /\ pc' = "save"
                       /\ UNCHANGED <<src, prog, p, frags, k, crashes, outcome>>
         ELSE Stuck("stuck: cannot open the converted store")
       ELSE IF Cardinality(tmp.fr) > p.rows THEN
         IF Todo = <<>> \/ Cardinality(tmp.fr) - p.rows # 1 THEN Stuck("stuck: converted rows do not match")
         ELSE /\ p' = [p EXCEPT !.done = Append(@, Head(Todo).id), !.rows = @ + 1]
              /\ pc' = "save"
              /\ UNCHANGED <<src, tmp, prog, frags, k, crashes, outcome>>
       ELSE Go("save")

Save ==
    /\ pc = "save"
    /\ prog' = p /\ k' = 1 /\ pc' = "dropdone"
    /\ UNCHANGED <<src, tmp, p, frags, crashes, outcome>>

DropDone ==  \* for fid in prog["done"]: drop(fid)
    /\ pc = "dropdone"
    /\ IF k > Len(p.done) THEN Go("loop")
       ELSE IF IsDS(src) /\ p.done[k] \in {e.id : e \in src.fr} THEN
         /\ src' = Store({e \in src.fr : e.id # p.done[k]}) /\ k' = k + 1
         /\ UNCHANGED <<tmp, prog, pc, p, frags, crashes, outcome>>
       ELSE IF ~IsDS(src) THEN Stuck("stuck: drop cannot open the source")
       ELSE /\ k' = k + 1 /\ UNCHANGED <<src, tmp, prog, pc, p, frags, crashes, outcome>>

Loop == pc = "loop" /\ Go(IF Todo = <<>> THEN "final" ELSE "write")

Write ==  \* write_dataset, create mode: the directory exists before the commit
    /\ pc = "write"
    /\ tmp' = IF tmp.st = "absent" THEN [st |-> "empty", fr |-> {}] ELSE tmp
    /\ pc' = "commit"
    /\ UNCHANGED <<src, prog, p, frags, k, crashes, outcome>>

Commit ==
    /\ pc = "commit"
    /\ tmp' = Store(tmp.fr \cup {[id |-> Cardinality(tmp.fr) + 1, orig |-> Head(Todo).orig, conv |-> TRUE]})
    /\ pc' = "progress"
    /\ UNCHANGED <<src, prog, p, frags, k, crashes, outcome>>

Progress ==
    /\ pc = "progress"
    /\ p' = [p EXCEPT !.done = Append(@, Head(Todo).id), !.rows = @ + 1]
    /\ prog' = p'
    /\ pc' = "drop"
    /\ UNCHANGED <<src, tmp, frags, k, crashes, outcome>>

DropCur == pc = "drop" /\ Drop(p.done[Len(p.done)], "loop")

FinalCheck ==
    /\ pc = "final"
    /\ IF ~IsDS(tmp) \/ Cardinality(tmp.fr) # p.rows THEN Stuck("stuck: converted rows do not match")
       ELSE IF ~IsDS(src) \/ src.fr # {} THEN Stuck("stuck: source fragments left")
       ELSE IF Fixed THEN /\ prog' = [p EXCEPT !.finishing = TRUE] /\ pc' = "rm1"
                          /\ UNCHANGED <<src, tmp, p, frags, k, crashes, outcome>>
       ELSE Go("rm1")

Rm1 ==  \* shutil.rmtree(src), interrupted part-way
    /\ pc = "rm1"
    /\ src' = IF src.st = "absent" THEN Absent ELSE [st |-> "partial", fr |-> {}]
    /\ pc' = "rm2"
    /\ UNCHANGED <<tmp, prog, p, frags, k, crashes, outcome>>

Rm2 == pc = "rm2" /\ src' = Absent /\ pc' = "rename" /\ UNCHANGED <<tmp, prog, p, frags, k, crashes, outcome>>

Rename == pc = "rename" /\ src' = tmp /\ tmp' = Absent /\ pc' = "rmprog"
          /\ UNCHANGED <<prog, p, frags, k, crashes, outcome>>

RmProg == pc = "rmprog" /\ prog' = NoProg /\ outcome' = "done" /\ pc' = "halt"
          /\ UNCHANGED <<src, tmp, p, frags, k, crashes>>

Step ==
    \/ Start \/ OpenTmp \/ Save \/ DropDone \/ Loop \/ Write \/ Commit \/ Progress \/ DropCur
    \/ FinalCheck \/ Rm1 \/ Rm2 \/ Rename \/ RmProg

Crash ==  \* the process dies; it is run again
    /\ outcome = "running" /\ crashes < MaxCrashes
    /\ pc' = "start" /\ p' = NoProg /\ frags' = <<>> /\ k' = 1 /\ crashes' = crashes + 1
    /\ UNCHANGED <<src, tmp, prog, outcome>>

Halt == pc = "halt" /\ UNCHANGED vars

Next == Step \/ Crash \/ Halt

Spec == Init /\ [][Next]_vars /\ WF_vars(Step)

(* ---- properties ---- *)
Held(o) == \E e \in src.fr \cup tmp.fr : e.orig = o

NoLoss == \A o \in Orig : Held(o)

NotStuck == outcome \in {"running", "done"}

DoneRight ==
    outcome = "done" =>
        /\ IsDS(src) /\ tmp = Absent /\ ~prog.exists
        /\ Cardinality(src.fr) = N /\ {e.orig : e \in src.fr} = Orig /\ \A e \in src.fr : e.conv

ConvertsSourceOnly == pc = "commit" => ~Head(Todo).conv  \* never re-converts a converted fragment

Finishes == <>(outcome # "running")
=============================================================================================

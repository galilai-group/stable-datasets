:orphan:

UCI HAR and MHEALTH (work in progress)
====================================

Status
------

This document records planned support for UCI HAR and MHEALTH.
Dataset builders, tests, and training/evaluation examples are not implemented yet.

Dataset sources
---------------

* `UCI HAR <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>`_:
  smartphone inertial signals from 30 participants performing six activities.
  The official archive provides 128-sample signal windows, activity labels,
  subject identifiers, and separate training and test sets.
* `MHEALTH <https://archive.ics.uci.edu/dataset/319/mhealth+dataset>`_:
  continuous recordings from 10 participants performing 12 activities.
  Each subject file contains 23 signal columns and an activity label column;
  label 0 denotes the null class.

Planned contribution
--------------------

* Add dataset builders following the existing time-series interfaces.
* Include source metadata, citations, subject identifiers, and documented labels.
* Add tests for parsing, shapes, labels, and split integrity.
* Add loading examples and a training/evaluation setup using an existing method.
* Document the experimental protocol and report results after running it.

Open decisions
--------------

MHEALTH windowing, label handling, subject splits, and channel selection remain
undecided. The learning method, model, training settings, and execution
environment also remain undecided.

License note
------------

The current UCI HAR repository page lists CC BY 4.0, while the README bundled
with its archive contains a non-commercial-use restriction. This discrepancy
needs clarification before finalizing the dataset's license metadata.

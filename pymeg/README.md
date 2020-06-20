# pymeg

This module contains a set of scripts to make anakysis of MEG experiments @UKE Hamburg easier.

The main components are:
 1. preprocessing: 
    - Detect eye, muscle and car artefacts as well as sensor jumps.
    - Read metadata (triggers) from data stream to epoch data.
 2. source reconstruction:
    - High level methods that encapsulate MNE functionality to do LCMV source estimates.
    - Define ROIs based on freesurfer labels and extract single trial estimates from ROIs.
    - Obtain TFR estimates in source space.
 3. parallelizaton of simple methods across PBS (UKE, lisa) and slurm clusters (Hummel)

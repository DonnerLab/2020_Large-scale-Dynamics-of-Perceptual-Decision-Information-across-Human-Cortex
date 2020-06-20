# Large-scale Dynamics of Perceptual Decision Information across Human Cortex

Analysis scripts

See conf_analysis.meg.figures for functions that generate individual figures. 

Often requires intermediate results to be generated via a mixture of other functions.

1) Source reconstruction: Start with lcmv.py, requires pymeg installed (see pymeg folder).
2) Decoding: decoding_analysis.py (aggregate source recon data with pymeg first)
3) TFRs: srtfr.py
4) Decoding in phase/vertex space: lcmv_decoding.py
5) Preprocessing of raw data: preprocessing.py


Intermediate outputs and or MEG raw / source reconstructed data can be obtained via individual request.

The analysis used MNE version 0.17.1.
See environments.txt for a list of installed packages.

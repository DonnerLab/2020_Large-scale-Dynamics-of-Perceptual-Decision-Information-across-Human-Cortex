.. pymeg documentation master file, created by
   sphinx-quickstart on Wed Jul  4 13:54:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Source reconstruction with pymeg
================================

Pymeg provides a host of helper functions to facilitate analysis of MEG
data. It makes heavy use of python-mne and mainly automates specific tasks
and provides the glue between separate processing steps. It does not 
actually provide a lot of new functionality.

This guide focuses exclusively on carrying out source reconstruction using
an LCMV beamformer. It describes most of the necessary preprocessing steps.

The core idea of pymeg is that you want to carry out single-trial source
reconstruction in a set of regions of interest. Anything else will require
various amounts of extra work not covered here.



Preparing a subject for source reconstruction
=============================================

Getting a subject ready for source reconstruction requires a few steps that 
can be carried out in parallel:
    - Preprocessing/epoching
    - Surface generation / recon-all

After this you can continue by:
    - Aligning the HCPMMP atlas to your subject
    - Aligning the Wang et al. 2015 atlas
    (or in general any kind of labels that you need)
    - Creating a high density head shape model
    - Creating BEM surfaces with fieldtrip
    - Creating a MRI<>MEG transformation matrix


Preprocess MEG data
-------------------

Use MNE/pymeg or fieldtrip to preprocess and epoch your data. Make sure
that you do not remove head localization channels from your epochs.

Please save your epochs as MNE fif files. Pymeg contains a fieldtrip->MNE
conversion script to load saved fieldtrip structures and save them as 
fif files. Please note that this requires the raw file that generated 
each epoch file - this allows us to extract MNE specific info structures.

It is also advisable to not change the channel names if you use fieldtrip,
if you do matching of fieldtrip channels to MNE channel info might become
difficult.

The conversion script can be found in pymeg/read_fieldtrip.py


Surface generation / recon-all
------------------------------

MNE performs source reconstruction on cortical surfaces. It is easiest
to create these surfaces by using freesurfers recon-all pipeline.

If you are @UKE you can use the cluster wide install for this. The first
step is to set your freesurfer subject folder:
    
    > export SUBJECTS_DIR=/path/to/subjects_dir

This directory will host the output of freesurfer (e.g. all cortical meshes).
Freesurfer will automatically create a subdirectory for each subject here. 

At this point you should also copy freesurfers fsaverage and fsaverage_sym
from /opt/progs/freesurfer/subjects to your subject dir. We will later add 
new information for these subjects and linking to /opt/progs makes these
subjects write protected.

You will likely have gotten a single T1 MPRAGE image in DICOM format as 
each subjects MRI. This needs to be converted to NIFTI format. You can use
dcm2niix for this (installed in /home/nwilming/bin/). The correct DICOM
series will have 'mprage' in it's name once converted to NIFTI.

Now call recon-all:
    
    > recon-all -subjid SUBID -all -i path/to/nifti_T1

And wait. This will take a few hours. It makes sense to parallelize this call
over the cluster. 


Aligning the HCPMMP atlas to your subject
-----------------------------------------

The cluster wide freesurfer install is quite old. There is a newer version
available in /home/nwilming/freesurfer. Setup your bashrc to reflect this 
install. 

To then get the HCPMMP atlas fire up a python >= 3.6 interpreter and import
pymeg:

   >>> from pymeg import atlas_glasser as ag
   >>> ag.get_hcp('subjects_dir') # Needs to be done only once
   >>> ag.get_hcp_annotation(subjects_dir, subject)

Now check if this worked by loading the subjects inflated surface and the
HCOMMP annotation file in freeview (only works on node031).


Aligning the Wang et al. 2015 atlas
-----------------------------------

This is a bit more involved. I suggest you use docker. Follow the instructions
here:

    .... link missing ...

Docker does not work on the cluser! You need to do this on your own laptop.


Creating a high density head shape model
----------------------------------------

Create a scalp surface from the MRI:

   >>> mne make_scalp_surfaces -s SUBID -d SUBJECT_DIR


Creating BEM surfaces with fieldtrip
------------------------------------

MNE has a tool to create bem surfaces from an MRI (mne watershed_bem).
I found that this did not work particularly well with the mprage
based images that we get. Using SPM/fieldtrip produced less artifacts.

To create head meshes with fieldtrip use the scripts in mfiles/


Creating a MRI<>MEG transformation matrix
-----------------------------------------

Use the MNE GUI to create transformation matrices. Pymeg provides a 
little help here... XXX

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: pymeg.source_reconstruction
   :members:

.. automodule:: pymeg.lcmv
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

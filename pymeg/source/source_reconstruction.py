'''
Compute source reconstruction for all subjects

Succesfull source reconstruction depends on a few things:

fMRI
    1. Recon-all MRI for each subject
    2. mne watershed_bem for each subject
    3. mne make_scalp_surfaces for all subjects
    4. Coreg the Wang & Kastner atlas to each subject using the scripts
       in require/apply_occ_wang/bin (apply_template and to_label)
MEG:
    5. A piece of sample data for each subject (in fif format)
    6. Create a coregistration for each subjects MRI with mne coreg
    7. Create a source space
    8. Create a bem model
    9. Create a leadfield
   10. Compute a noise and data cross spectral density matrix
       -> Use a data CSD that spans the entire signal duration.
   11. Run DICS to project epochs into source space
   12. Extract time course in labels from Atlas.

'''
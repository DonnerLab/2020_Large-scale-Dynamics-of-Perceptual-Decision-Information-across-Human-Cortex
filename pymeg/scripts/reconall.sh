#!/bin/sh

echo "#!/bin/sh

#PBS -q batch
#PBS -l walltime=10:00:00

# -- run in the current working (submission) directory --
cd \$PBS_O_WORKDIR

chmod g=wx \$PBS_JOBNAME

export SUBJECTS_DIR=${3}

recon-all -subjid $1 -i $2 -all 1> \$PBS_JOBID.out 2> \$PBS_JOBID.err" > _reconall.sh

qsub _reconall.sh


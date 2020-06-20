  #!/bin/sh
  # embedded options to qsub - start with #PBS
  # walltime: defines maximum lifetime of a job
  # nodes/ppn: how many nodes (usually 1)? how many cores?

      #PBS -q batch
      #PBS -l walltime=6:00:00
      #PBS -l nodes=1:ppn=1
      #PBS -l mem=18gb
        
  # -- run in the current working (submission) directory --
  cd $PBS_O_WORKDIR

  chmod g=wx $PBS_JOBNAME

  # FILE TO EXECUTE
  matlab -nodisplay -nodesktop -r "run_sub($job); exit"  1> jobs/$PBS_JOBID.out 2> jobs/$PBS_JOBID.err

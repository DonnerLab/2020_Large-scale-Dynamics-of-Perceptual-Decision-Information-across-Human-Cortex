function run_sub(job)
if ischar(job)
    job = str2double(job); 
end

session = repmat(1:4, 1, 6);
subjects = repmat(1:6, 4, 1)
subjects = subjects(:)';

addpath('/home/nwilming/fieldtrip-20151111')
addpath('artreject')
ft_defaults
preproc(subjects(job), session(job))


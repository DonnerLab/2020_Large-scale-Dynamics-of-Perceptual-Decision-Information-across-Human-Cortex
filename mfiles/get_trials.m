function cfg = get_trials(dataset, trialfun)
cfg                         = [];
cfg.dataset                 =  dataset;
cfg.continuous              = 'yes'; % read in the data
cfg.precision               = 'single'; % for speed and memory issues
cfg.channel                 = channel_names(dataset);
cfg.fs = 1200;

fprintf('\n Now epoching\n')

cfg.trialfun            = trialfun;

cfg.trialdef.pre        = 0; % before the fixation trigger, to have enough length for padding (and avoid edge artefacts)
cfg.trialdef.post       = 1; % after feedback
cfg 					= ft_definetrial(cfg); % define all trials

end

function [data, cfg] = prepare_dataset(dataset, subject, session, block_start, block_end)
% read in the dataset as a continuous segment
cfg                         = [];
cfg.dataset                 =  dataset;
cfg.continuous              = 'yes'; % read in the data
cfg.precision               = 'single'; % for speed and memory issues
cfg.channel                 = channel_names(dataset);
cfg.sj                      = subject;
cfg.session                 = session;
cfg.fs = 1200;
cfg.trl =  [block_start    block_end   block_start];
% add a highpass filter to get rid of cars
% 0.1 Hz, see Acunzo et al
% if filtering higher than that, can't look at onset of ERPs!
cfg.hpfilttype              = 'fir';
cfg.hpfiltord               = 6;
cfg.hpfilter                = 'yes';
cfg.hpfreq                  = 0.1;

fprintf('Reading %2fs of data', (block_end-block_start)/cfg.fs)
data = ft_preprocessing(cfg);

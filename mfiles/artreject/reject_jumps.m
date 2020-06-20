function artifact_jump = reject_jumps(dataset, data)
% 2. JUMPS
cfg                                     = [];
cfg.datafile                            = dataset;
cfg.headerfile                          = dataset;
cfg.trl                                 = [data.sampleinfo zeros(size(data.sampleinfo, 1))];
cfg.continuous                          = 'yes'; % data has been epoched
cfg.memory='low';

% channel selection, cutoff and padding
cfg.artfctdef.zvalue.channel            = 'MEG';
cfg.artfctdef.zvalue.trlpadding         = 0;
cfg.artfctdef.zvalue.artpadding         = 0;
cfg.artfctdef.zvalue.fltpadding         = 0;

% algorithmic parameters
cfg.artfctdef.zvalue.cumulative         = 'yes';
cfg.artfctdef.zvalue.medianfilter       = 'yes';
cfg.artfctdef.zvalue.medianfiltord      = 9;
cfg.artfctdef.zvalue.absdiff            = 'yes';


cfg.artfctdef.zvalue.cutoff         = 50;
cfg.artfctdef.zvalue.interactive    = 'no';
[~, artifact_jump]       = ft_artifact_zvalue(cfg, data);

% cfg                             = [];
% cfg.artfctdef.reject            = 'complete';
% cfg.artfctdef.jump.artifact     = artifact_jump;
% data                            = ft_rejectartifact(cfg, data);

function artifact_eog = reject_eye(data, varargin)
%%
crittoilim = default_arguments(varargin, 'crittoilim', []);
channels = default_arguments(varargin, 'channels', {'UADC002', 'UADC003', 'UADC004'});

% 4. EYEBLINKS (only during beginning of trial)
cfg                                     = [];
%cfg.datafile                            = dataset;
%cfg.headerfile                          = dataset;
%cfg.trl                                 = data.trl; %[data.sampleinfo zeros(size(data.sampleinfo, 1))];
cfg.continuous                          = 'yes'; % data has been epoched


cfg.artfctdef.zvalue.trlpadding  = 0; % padding doesnt work for data thats already on disk
cfg.artfctdef.zvalue.fltpadding  = 0;
cfg.artfctdef.zvalue.artpadding  = 0.05; % go a bit to the sides of blinks
cfg.artfctdef.zvalue.channel = channels;
% algorithmic parameters
cfg.artfctdef.zvalue.bpfilter   = 'yes';
cfg.artfctdef.zvalue.bpfilttype = 'but';
cfg.artfctdef.zvalue.bpfreq     = [1 15];
cfg.artfctdef.zvalue.bpfiltord  = 4;
cfg.artfctdef.zvalue.hilbert    = 'yes';

% make the process interactive
cfg.artfctdef.zvalue.cutoff      = 1;
cfg.artfctdef.zvalue.interactive = 'no';

[~, artifact_eog]          = ft_artifact_zvalue(cfg, data);

% cfg                             = [];
% cfg.artfctdef.reject            = 'complete';
% cfg.artfctdef.eog.artifact      = artifact_eog;

% and during the second up until response
%crittoilim = [ data.trialinfo(:,5) - data.trialinfo(:,1) ...
%    data.trialinfo(:,9) - data.trialinfo(:,1)]  / data.fsample;
% cfg.artfctdef.crittoilim        = crittoilim;
% data                            = ft_rejectartifact(cfg, data);

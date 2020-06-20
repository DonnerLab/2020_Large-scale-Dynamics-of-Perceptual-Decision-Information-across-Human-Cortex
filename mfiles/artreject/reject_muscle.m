
function artifact_muscle = reject_muscle(dataset, data)
%% 3. MUSCLE BURST
cfg                                     = [];
cfg.datafile                            = dataset;
cfg.headerfile                          = dataset;
cfg.trl                                 = [data.sampleinfo zeros(size(data.sampleinfo, 1))];
cfg.continuous                          = 'yes'; % data has been epoched
cfg.memory='low';

% channel selection, cutoff and padding
cfg.artfctdef.zvalue.channel     = 'MEG';
cfg.artfctdef.zvalue.trlpadding  = 0;
cfg.artfctdef.zvalue.fltpadding  = 0;
cfg.artfctdef.zvalue.artpadding  = 0.1;

% algorithmic parameters
cfg.artfctdef.zvalue.bpfilter    = 'yes';
cfg.artfctdef.zvalue.bpfreq      = [110 140];
cfg.artfctdef.zvalue.bpfiltord   = 9;
cfg.artfctdef.zvalue.bpfilttype  = 'but';
cfg.artfctdef.zvalue.hilbert     = 'yes';
cfg.artfctdef.zvalue.boxcar      = 0.2;


cfg.artfctdef.zvalue.cutoff      = 20;
cfg.artfctdef.zvalue.interactive = 'no';

[~, artifact_muscle] = ft_artifact_zvalue(cfg, data);

% cfg                             = [];
% cfg.artfctdef.reject            = 'partial';
% cfg.artfctdef.muscle.artifact   = artifact_muscle;

% only remove muscle bursts before the response
% The trial looks like this:
% Start - REF - JTR - STIM - RESP - JTR - FEEDBACK - JTR - END
% The period (REF-baseline) -> Feedback should be clean
% let's say the baseline is 250ms.
%ref_onset = data.trialinfo(:, 6);
%feedback = data.trialinfo(:, 21);
%crittoilim = [ (ref_onset - 300) feedback]  ./ data.fsample;

%cfg.artfctdef.crittoilim        = crittoilim;
%data                            = ft_rejectartifact(cfg, data);


function example_topos(data, cfg_block, artifacts)
%% Compute time locked topos

% (1) Downsample trl structure to correct frequency and add sampleinfo
% field to data
cfg_block.trl(:, 3) = 0; %cfg_block.trl(:, 1);
cfg_block.trl = downsample_trial_times(data, cfg_block.trl, 1200, 500);

data.sampleinfo = cfg_block.trl(:,1:2);

% (2) Downsample artifact definitions to 500Hz
 artifacts.jump.artifact = round(artifacts.jump.artifact * (500/1200));
 artifacts.muscle.artifact = round(artifacts.muscle.artifact * (500/1200));
 artifacts.eog.artifact = round(artifacts.eog.artifact * (500/1200));

cfgart.reject = 'nan';
cfgart.artfctdef = artifacts;
clean = ft_rejectartifact(cfgart, data);

% (3)  Get time locked trial definition and redefine trials to this period
trl = timelock(cfg_block, data, 'ref_onset', 'ref_onset', 'ref_onset', 0, 2*500);
cfg_tl.trl = trl;
tlock = ft_redefinetrial(cfg_tl, clean);

% (4) Compute timelock analysis
cfgn.vartrllength = 0;
cfgn.channel = cfg_block.channel(3:271);
m = ft_timelockanalysis(cfgn, tlock);

% (5) Show topo plot
cfgn.xlim = [0, 0.4, 0.9, 1.9, 2];
ft_topoplotER(cfgn, m)
end
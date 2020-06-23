function run_choice_model(nsubj)

allsubj = {'S1' 'S2' 'S3' 'S4' 'S5' 'S6' 'S7' 'S8' 'S9' 'S10' 'S11' 'S12' 'S13' 'S14' 'S15'};
subj = allsubj{nsubj};

clusters = {'vfcPrimary';'vfcEarly';'vfcVO';'vfcPHC';'vfcTO';'vfcLO';'vfcV3ab';'vfcIPS01';'vfcIPS23';'vfcFEF';...   % Wang
            'JWG_aIPS';'JWG_IPS_PCeS';'JWG_M1'};                                                                    % JW

talpha = 0.192;  % sample-aligned time-point at which to take alpha power
tresp = 0.192;   % sample-aligned time-point at which to take evoked response

% Paths dervied from processing options
fname = '/mnt/homes/home028/nwilming/conf_meg/nn_rev_regression_hemiavg.h5'; % file to load
savepath = '/mnt/homes/home024/pmurphy/Wilming_conf/choice_model/output/';

% pull information about file structure
info = h5info(fname);

% retrieve frequencies & times
freqs=[];
for f = 1:length([{info.Groups(1).Groups(1).Groups(:).Name}])
    freqs(f) = str2num(info.Groups(1).Groups(1).Groups(f).Name(18:end));
end
freqs = round(sort(freqs));

% loop through clusters
for nclust = 1:length(clusters)
    fprintf('Processing %s, %s...\n',subj,clusters{nclust})
    
    % load times vector
    times = h5read(fname,['/',subj,'/',clusters{nclust},'/',num2str(freqs(1)),'/time/'])';
    
    % retrieve contrast values & onset times
    choices = h5read(fname,['/',subj,'/choice/']); choices(choices==-1) = 0;  % 0=weaker, 1=stronger
    cval_trial = h5read(fname,['/',subj,'/cval_trial/'])';
    onsets = h5read(fname,['/',subj,'/cval_contrast_onset/'])';
    
    % sort contrast values by trial ID
    [cval_trial,~] = sort(double(cval_trial));
    
    % retrieve (trials*times*freqs) matrix of power values
    pow=[];
    for f = 1:length(freqs)
        % load
        pow_in = h5read(fname,['/',subj,'/',clusters{nclust},'/',num2str(freqs(f)),'/power/'])';
        pow_trial = h5read(fname,['/',subj,'/',clusters{nclust},'/',num2str(freqs(f)),'/trial/'])';
        pow_time = h5read(fname,['/',subj,'/',clusters{nclust},'/',num2str(freqs(f)),'/time/'])';
        assert(sum(pow_time-times)==0,[subj,', ',clusters{nclust},', ',num2str(freqs(f)),'Hz: time vector mismatch!!!'])
        
        % sort power values by trial ID
        [pow_trial,ord] = sort(double(pow_trial));
        assert(sum(pow_trial-cval_trial)==0,[subj,', ',clusters{nclust},', ',num2str(freqs(f)),'Hz: MEG/contrast trial mismatch!!!'])
        pow(:,:,f) = pow_in(ord,:);
    end
    
    % pull only power @ time-points of interest
    ts=[]; ts_alpha=[];
    for smp = 1:length(onsets)
        ts_alpha(end+1) = find(abs(times-(onsets(smp)+talpha))==min(abs(times-(onsets(smp)+talpha))));
        ts(end+1) = find(abs(times-(onsets(smp)+tresp))==min(abs(times-(onsets(smp)+tresp))));
    end
    powsmp_alpha = pow(:,ts_alpha,:);
    powsmp = pow(:,ts,:);
    ts = times(ts);
    
    % fit logistic regression models averaging power over 1st/last 5 samples
    alpha = [zscore(squeeze(mean(powsmp_alpha(:,1:5,find(freqs==10)),2))) zscore(squeeze(mean(powsmp_alpha(:,6:10,find(freqs==10)),2)))];
    pow = cat(2,zscore(mean(powsmp(:,1:5,:),2),[],1),zscore(mean(powsmp(:,6:10,:),2),[],1));
    nsgrp=2;
    
    for f = 1:size(pow,3)
        [b,~,m] = glmfit([squeeze(pow(:,:,f)) alpha squeeze(pow(:,:,f)).*alpha],[choices ones(length(choices),1)],'binomial');
        Ts_grp(nclust,f,:,1) = b(2:(nsgrp+1));
        Ts_grp(nclust,f,:,2) = b((nsgrp+2):(2*nsgrp+1));
        Ts_grp(nclust,f,:,3) = b((2*nsgrp+2):end);
    end
end

% save
save([savepath,subj,'.mat'],'Ts_grp','ts','freqs','clusters','tresp','talpha')
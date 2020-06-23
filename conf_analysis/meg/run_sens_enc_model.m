function run_sens_enc_model(nsubj)

allsubj = {'S1' 'S2' 'S3' 'S4' 'S5' 'S6' 'S7' 'S8' 'S9' 'S10' 'S11' 'S12' 'S13' 'S14' 'S15'};
subj = allsubj{nsubj};

clusters = {'vfcPrimary';'vfcEarly';'vfcVO';'vfcPHC';'vfcTO';'vfcLO';'vfcV3ab';'vfcIPS01';'vfcIPS23';'vfcFEF';...   % Wang
            'JWG_aIPS';'JWG_IPS_PCeS';'JWG_M1'};                                                                    % JW

talpha = 0.192;  % sample-aligned time-point at which to take alpha power
tresp = 0.192;   % sample-aligned time-point at which to take evoked response

% Paths derived from processing options
fname = '/mnt/homes/home028/nwilming/conf_meg/nn_rev_regression_hemiavg.h5'; % file to load
savepath = '/mnt/homes/home024/pmurphy/Wilming_conf/sens_enc_model/output/';

% pull information about file structure
info = h5info(fname);

% retrieve frequencies & times
freqs=[];
for f = 1:length([{info.Groups(1).Groups(1).Groups(:).Name}])
    freqs(f) = str2num(info.Groups(1).Groups(1).Groups(f).Name(18:end));
end
freqs = round(sort(freqs));

% loop through clusters
Ts=[]; Ts_conly=[];
for nclust = 1:length(clusters)
    fprintf('Processing %s, %s...\n',subj,clusters{nclust})
    
    % load times vector
    times = h5read(fname,['/',subj,'/',clusters{nclust},'/',num2str(freqs(1)),'/time/'])';
    
    % retrieve contrast values & onset times
    cvals = h5read(fname,['/',subj,'/cvals/'])';
    cval_trial = h5read(fname,['/',subj,'/cval_trial/'])';
    onsets = h5read(fname,['/',subj,'/cval_contrast_onset/'])';
    
    % sort contrast values by trial ID
    [cval_trial,ord] = sort(double(cval_trial));
    cvals = cvals(ord,:);
    
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
    
    % fit sample-wise regression models
    for smp = 1:size(powsmp,2)
        for f = 1:size(powsmp,3)
            alpha = squeeze(zscore(powsmp_alpha(:,smp,freqs==10)));
            m = regstats(zscore(squeeze(powsmp(:,smp,f))),[zscore(cvals(:,smp))],'linear',{'beta'});  % full model
            Ts_conly(nclust,f,smp) = m.beta(2);
        end
    end
end

% save
save([savepath,subj,'.mat'],'Ts_conly','ts','freqs','clusters','tresp')

function preproc(subject, session)
dataset = get_raw_filenames(subject, session);

% First of all get trial struct
cfg = get_trials(dataset, 'trialdef');
blocks = get_blocks(cfg.trl(:, end));
block_nums = unique(blocks);
for block = 1:length(block_nums)
    id_block = blocks == block;
    
    trls = cfg.trl(id_block, 1:2);
    block_start = min(trls(:)) - 2500
    block_end = max(trls(:)) + 2500
    
    [data, cfg_block] = prepare_dataset(dataset, subject, session, block_start, block_end);
    cfg_block.trl = cfg.trl(id_block, :);
    artifacts = reject_artifacts(dataset, data);
    cfg_block.artfctdef = artifacts;
    
    % Compute the time period of interest: Between ref onset and decision.
    event_fields = {'start', 'correct', 'correct_v', 'noise_sigma', 'noise_sigma_v',...
        'ref_onset', 'ref_offset', 'stim_onset', 'cc1', 'cc2', 'cc3',...
        'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9', 'cc10', 'stim_offset', 'response',...
        'response_v', 'feedback', 'feedback_v', 'end'};
    
    trial_onset = cfg_block.trl(:, 1);
    ref_onset = cfg_block.trl(:, find(strcmp('ref_onset', event_fields)) + 3);
    response = cfg_block.trl(:, find(strcmp('response', event_fields)) + 3);
    
    %cfg_block.artfctdef.crittoilim = [ref_onset, response]/1200;
    cfg_block.artfctdef.reject = 'complete';
    
    data = ft_redefinetrial(cfg_block, data);
    
    %cfg_block = ft_rejectartifact(cfg_block);
    
    samplerows = find(data.trialinfo(1,:)>255);
    data = downsample(data);
    savepath = get_filenames(subject, session, block, 'confidence');
    save(savepath, '-v7.3')
end
end

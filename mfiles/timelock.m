function trl = timelock(cfg, data, start, endt, offset, pre, post)
    % get event times
event_fields = {'start', 'correct', 'correct_v', 'noise_sigma', 'noise_sigma_v',...
    'ref_onset', 'ref_offset', 'stim_onset', 'cc1', 'cc2', 'cc3',...
    'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9', 'cc10', 'stim_offset', 'response',...
    'response_v', 'feedback', 'feedback_v', 'end'};

field_offset = size(cfg.trl, 2) - length(event_fields);
id_start = find(strcmp(start, event_fields)) + field_offset;
id_end = find(strcmp(endt, event_fields)) + field_offset;
id_offset = find(strcmp(offset, event_fields)) + field_offset;

startt = cfg.trl(:, id_start)-pre -cfg.trl(:,1);
endt =  cfg.trl(:, id_end)+post-cfg.trl(:,1);
trl = [startt, endt, endt*0]; %#ok<FNDSB>

trl = [trl cfg.trl(:, field_offset:end)];
end
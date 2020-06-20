function plot_trl(trl, artifacts)


% Compute the time period of interest: Between ref onset and decision. 
event_fields = {'start', 'correct', 'correct_v', 'noise_sigma', 'noise_sigma_v',...
    'ref_onset', 'ref_offset', 'stim_onset', 'cc1', 'cc2', 'cc3',...
    'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9', 'cc10', 'stim_offset', 'response',...
    'response_v', 'feedback', 'feedback_v', 'end'};

ref_onset = trl(:, find(strcmp('ref_onset', event_fields)) + 3);
response = trl(:, find(strcmp('response', event_fields)) + 3);

for row = 1:size(trl,1)
    start = trl(row, 1);
    end_t = trl(row, 3);
    refon = ref_onset(row);
    resp = response(row);
    plot([start, end_t], [1, 1], '+-b');
    hold on
    plot([refon, resp], [1+0.1, 1+0.1], '+-r')
end

for field = {'muscle', 'eog', 'jump'}

    arts = artifacts.(field{1}).artifact;
    size(arts)
    for i = 1:length(arts)
        plot([arts(i, 1), arts(i, 2)], [1.3, 1.3], 'x-g')
    end
end
function trials = filter_trials(trialstruct, start_event, start_offset, end_event, end_offset, t0_event)
trials = zeros(length(trialstruct), 3);
for k = 1:length(trialstruct)
    trials(k,1) = trialstruct(k).(start_event) + start_offset;
    trials(k,2) = trialstruct(k).(end_event) + end_offset;
    trials(k,3) = trialstruct(k).(t0_event);
end

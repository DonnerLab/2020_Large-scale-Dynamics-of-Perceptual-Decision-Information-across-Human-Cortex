function trials = trialdef(cfg)
%
% Notes about data files:
%Pilot03_Confidence_20151124_01.ds

events = ft_read_event(cfg.dataset);
events = events(strcmp('UPPT001', {events.type}));
%% Find start and end points of trials.
starts = find([events.value]==150);
ends = find([events.value]==151);

assert (events(starts(1)).sample < events(ends(1)).sample)
if (length(ends) ~= length(starts))
    % Sometimes it happens that we have incomplete trials. For example when
    % matlab stops because it loses connection to the license server. In
    % these cases I stop the experiment somewhere and then restart it. In
    % all likelihood we'll obeserve something like:
    % [150 .... 150 .... 151].
    % Test for this and remove the incomplete trial.
    while (length(ends) ~= length(starts))
        del_start = [];
        offset = 1;
        for trial = offset:min([length(ends), length(starts)])
            startt = starts(trial);
            endt = ends(trial);
            t_events = events(startt:endt);
            if numel(find([t_events(strcmp('UPPT001', {t_events.type})).value]==150)) > 1
                % This is one of these cases.
                fprintf('Deleting one trial onset at event=%i\n', trial)
                del_start = trial;
                offset = trial-1;
                break
            end
        end
        
        starts(del_start) = [];
        %length(ends), length(starts), del_start
    end
end

%% Check if encoding between trials has worked
trial_nums = [];
trial_ons = [];
valutas = {};
start_sample = 1;
for cur_trial = 1:length(ends)
    % Only accept values that are within 100ms of trial start
    values = [events(start_sample:starts(cur_trial)-1).value];
    values_t = [events(start_sample:starts(cur_trial)).sample];
    val_filt = values((values_t(1:end-1) - values_t(end)) > -120);
    start_sample = ends(cur_trial)+1;
    if isequal(values, [16, 24])
        trial_nums = [trial_nums 1];
    else
        trial_nums = [trial_nums, pins2num(val_filt)];
    end
    valutas(cur_trial) = {val_filt};
    trial_ons = [trial_ons values_t(end)];
end

if ~isequal(trial_nums, repmat(1:100, 1, 5))
    fprintf('Trial ordering is not 5x 1:100\nFound %i trials in total\nCheck that this is OK!\n', length(trial_nums))
end
%% Set up the trial structure as lists
sequence = {[150], [41,40], [31, 32, 33]...
    [64], [49],...
    [64], [50], [50], [50], [50], [50], [50], [50], [50], [50], [50], [48],...
    [24, 23, 22, 21, 88],...
    [10, 11],...
    [151]};

names = {'start', 'correct', 'noise_sigma', 'ref_onset', 'ref_offset', 'stim_onset',...
    'cc1', 'cc2', 'cc3', 'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9', 'cc10', 'stim_offset',...
    'response', 'feedback', 'end'};
save_val = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0};
trigger_optional = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0};

%% Parse trials by traversing trial sequence and triggers.
trials = struct();
for trial = 1:length(starts)
    error_trial = false;
    startt = starts(trial);
    endt = ends(trial);
    t_events = events(startt:endt);
    assert(ismember(151, [t_events.value]));
    trigs = [t_events(strcmp('UPPT001', {t_events.type})).value];
    trigs_s = [t_events(find(strcmp('UPPT001', {t_events.type}))).sample];
    assert(length(trigs)==length(trigs_s))
    trig_cnt = 1;
    seq_cnt = 1;
    while seq_cnt <= length(sequence) && trig_cnt <= length(trigs)
        
        if ismember(trigs(trig_cnt), [63, 114, 190, 191])
            trig_cnt = trig_cnt+1;
            continue
        end
        if ismember(trigs(trig_cnt), sequence{seq_cnt})
            %fprintf('Adding trigger %i\n', trigs(trig_cnt))
            trials(trial).(names{seq_cnt}) = trigs_s(trig_cnt);
            if save_val{seq_cnt}
                trials(trial).([names{seq_cnt} '_v']) = trigs(trig_cnt);
            end
            if trigs(trig_cnt) == 88
                error_trial = true;
            end
            trig_cnt = trig_cnt +1;
            seq_cnt = seq_cnt +1;
            continue
        else
            if trigger_optional{seq_cnt}
                %fprintf('Optional trigger: %s\n', names{seq_cnt});
                seq_cnt = seq_cnt+1;
                continue
            elseif ~error_trial
                fprintf('Got: %i\nExpected: ', trigs(trig_cnt))
                disp(sequence{seq_cnt})
                fprintf('Trig cnt: %i, Seq cnt: %i\n', trig_cnt, seq_cnt)
                error('Trial sequence does not fit');
            end
        end
        disp('this should not happen')
    end
end

event_fields = {'start', 'correct', 'correct_v', 'noise_sigma', 'noise_sigma_v',...
    'ref_onset', 'ref_offset', 'stim_onset', 'cc1', 'cc2', 'cc3',...
    'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9', 'cc10', 'stim_offset', 'response',...
    'response_v', 'feedback', 'feedback_v', 'end'};

trial_mat = filter_trials(trials, 'start', 0, 'end', 0, 'start');
trialinfo = nan(length(trials), length(event_fields) + 1);

for field = 1:length(event_fields)
    for t = 1:length(trials)
        if numel(trials(t).(event_fields{field})) > 0
            trialinfo(t, field) = trials(t).(event_fields{field});
        end
    end
end
trialinfo(:, end) = trial_nums;
trials = horzcat(trial_mat, trialinfo);


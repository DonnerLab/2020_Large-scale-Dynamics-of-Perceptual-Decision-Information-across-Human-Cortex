function trl = downsample_trial_times(data, trl, orig_fs, resample_fs)
% Readjust trial end times after downsampling such that the trial structure
% matches with the number of samples that are available. This will correct
% for rounding errors.

samples = trl(1,:) > 256;
for k = 1:size(trl,1)
    trl(k, samples) = round(trl(k, samples)*(resample_fs/orig_fs));
    trl(k, 2) = trl(k, 1) + length(data.trial{k});
end
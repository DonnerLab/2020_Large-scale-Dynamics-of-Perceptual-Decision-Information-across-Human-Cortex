function data = downsample_per_channel(fs, datafile)
cfg1 = [];
cfg1.dataset = datafile;
cfg1.continuous = 'yes';
cfg2 = [];
cfg2.resamplefs = fs;

hdr = ft_read_header(datafile);
channels = {};
cnt = 1;
for k = 1:length(hdr.label)
    if numel(strfind(hdr.label{k}, 'M')) > 0
        channels{cnt} = hdr.label{k};
        cnt = cnt+1;
    elseif numel(strfind(hdr.label{k}, 'U')) > 0
        channels{cnt} = hdr.label{k};
        cnt = cnt+1;
    elseif numel(strfind(hdr.label{k}, 'EEG057')) > 0 | numel(strfind(hdr.label{k}, 'EEG058')) > 0 | numel(strfind(hdr.label{k}, 'EEG059')) > 0
        channels{cnt} = hdr.label{k};
        cnt = cnt+1;
    end
end

for i=1:length(channels)
  cfg1.channel = channels{i}; % you can use a number as well as a string
  temp = ft_preprocessing(cfg1);
  data = ft_resampledata(cfg2, temp);
  singlechan{i} = ft_resampledata(cfg2, temp);
  clear temp;
end % for all channels

data = ft_appenddata([], singlechan{:});
function channels = channel_names(dataset)
hdr = ft_read_header(dataset);
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
end
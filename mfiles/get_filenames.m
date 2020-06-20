function dataset = get_filenames(subject, session, block, varargin)

path = default_arguments(varargin, 'path', '/Volumes/dump/conf_meg/');

path = fullfile(path, sprintf('S%i', subject));
mkdir(path);

type = default_arguments(varargin, 'type', 'confidence');

dataset = fullfile(path, sprintf('%02i_%02i_%02i_%s.mat', subject, session, block, type));

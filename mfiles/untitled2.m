cfg.dataset = '/Volumes/dump/Pilot03-04_Confidence_20151126_02.ds';
cfg.trialfun = 'trialdef';
cfg.datatype = 'continuous';
cfg = ft_definetrial(cfg);
data = ft_preprocessing(cfg);
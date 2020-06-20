function artifact_car = reject_cars(dataset, data)
%% 1. CARS - thresholding
fprintf('\n\nLooking for CAR artifacts . . .\n')
cfg                                     = [];
cfg.datafile                            = dataset;
cfg.headerfile                          = dataset;
cfg.trl                                 = [data.sampleinfo zeros(size(data.sampleinfo, 1), 1)];
cfg.continuous                          = 'yes'; % data has been epoched
cfg.memory='low';

cfg.artfctdef.threshold.channel         = 'MEG';
cfg.artfctdef.threshold.bpfilter        = 'no';
cfg.artfctdef.threshold.range           = 0.75e-11; % setting thomas
[~, artifact_car ]                        = ft_artifact_threshold(cfg, data);

% cfg                                     = [];
% cfg.artfctdef.reject                    = 'complete';
% cfg.artfctdef.car.artifact              = artifact_car;
% data                                    = ft_rejectartifact(cfg, data);

function artifacts = reject_artifacts(dataset, data)
%[data, art_car] = reject_cars(dataset, data);
art_jumps = reject_jumps(dataset, data);
art_muscle = reject_muscle(dataset, data);
art_eye = reject_eye(data);
%artifacts.car = art_car;
artifacts.jump.artifact = art_jumps;
artifacts.muscle.artifact = art_muscle;
artifacts.eog.artifact = art_eye;


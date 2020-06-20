function num = pins2num(pins)
str = '00000000';
for i = 1:length(pins)
    str(pins(i)) = '1';
end

num = bin2dec(str);
function blocks = get_blocks(trials)
start = trials(1);
blocks = 0*trials;
blocks(1) = 1;
cur_block = 1;
for k = 2:length(trials)
    if trials(k-1)+1 ~= trials(k)
        cur_block = cur_block + 1;
    end
    blocks(k) = cur_block;
end
end

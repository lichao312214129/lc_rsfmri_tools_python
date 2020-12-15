d=[Inf, NaN];

inf_idx = isinf(d);
nan_idx = isnan(d);

d(inf_idx) = 1;
d(nan_idx) = 0;
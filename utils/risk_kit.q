
/Compute the skewness
skewness: {[r] (avg((r-avg r) xexp 3)) % (dev r) xexp 3};


/calculate the kurtosis
kurtosis: {[r] (avg ((r-avg r) xexp 4))%(dev r) xexp 4}

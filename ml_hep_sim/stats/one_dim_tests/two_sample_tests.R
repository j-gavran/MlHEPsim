library(SuppDists) # needed for kSamples
library(kSamples)

# R ks test
ks_test <- function(x, y){
    result = ks.test(x, y)
    return(result)
}

# Anderson-Darling test
ad_test <- function(x, y){
    result = kSamples::ad.test(x, y)
    return(result)
}

chi2_test <- function(x, y){
    result = chisq.test(cbind(x, y))
    return(result)
}

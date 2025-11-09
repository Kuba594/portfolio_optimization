import rpy2.robjects as ro
print(ro.r('R.version.string'))

ro.r('library(rugarch)')
ro.r('library(rmgarch)')
ro.r('library(xts)')
print("All R libraries loaded successfully.")

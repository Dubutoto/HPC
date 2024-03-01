# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
export OMP_NUM_THREADS=28
export OMP_PROC_BIND=true
module load languages/anaconda2/5.0.1
module load icc/2017.1.132-GCC-5.4.0-2.26 
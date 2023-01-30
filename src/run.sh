sed -i 's/#define NUM_TRAIN .*/#define NUM_TRAIN 1000/' qtersection.cu
nvcc -O3 -o qtersection.o qtersection.cu 
sbatch ./run_job.sh output_1000.txt
squeue -u lorenlin

sed -i 's/#define NUM_TRAIN .*/#define NUM_TRAIN 5000/' qtersection.cu
nvcc -O3 -o qtersection.o qtersection.cu 
sbatch ./run_job.sh output_5000.txt
squeue -u lorenlin

sed -i 's/#define NUM_TRAIN .*/#define NUM_TRAIN 10000/' qtersection.cu
nvcc -O3 -o qtersection.o qtersection.cu 
sbatch ./run_job.sh output_10000.txt
squeue -u lorenlin

sed -i 's/#define NUM_TRAIN .*/#define NUM_TRAIN 50000/' qtersection.cu
nvcc -O3 -o qtersection.o qtersection.cu 
sbatch ./run_job.sh output_50000.txt
squeue -u lorenlin

sed -i 's/#define NUM_TRAIN .*/#define NUM_TRAIN 100000/' qtersection.cu
nvcc -O3 -o qtersection.o qtersection.cu 
sbatch ./run_job.sh output_100000.txt
squeue -u lorenlin

sed -i 's/#define NUM_TRAIN .*/#define NUM_TRAIN 500000/' qtersection.cu
nvcc -O3 -o qtersection.o qtersection.cu 
sbatch ./run_job.sh output_500000.txt
squeue -u lorenlin

sed -i 's/#define NUM_TRAIN .*/#define NUM_TRAIN 1000000/' qtersection.cu
nvcc -O3 -o qtersection.o qtersection.cu 
sbatch ./run_job.sh output_1000000.txt
squeue -u lorenlin
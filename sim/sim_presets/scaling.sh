mpirun -n 16 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi1 -alpha 1.0 -e spiral -m linear_cg -dh 16 -dt 0.005 &&
mpirun -n 8 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi2 -alpha 1.0 -e spiral -m linear_cg -dh 16 -dt 0.005 &&
mpirun -n 4 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi3 -alpha 1.0 -e spiral -m linear_cg -dh 16 -dt 0.005 &&
python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi4 -alpha 1.0 -e spiral -m linear_cg -dh 16 -dt 0.005 &&
mpirun -n 16 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi5 -alpha 1.0 -e spiral -m linear_dg -dh 16 -dt 0.005 &&
mpirun -n 8 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi6 -alpha 1.0 -e spiral -m linear_dg -dh 16 -dt 0.005 &&
mpirun -n 4 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi7 -alpha 1.0 -e spiral -m linear_dg -dh 16 -dt 0.005 &&
python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi8 -alpha 1.0 -e spiral -m linear_dg -dh 16 -dt 0.005 &&
mpirun -n 16 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi9 -alpha 0.1 -e spiral -m linear_cg -dh 16 -dt 0.005 &&
mpirun -n 8 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi10 -alpha 0.1 -e spiral -m linear_cg -dh 16 -dt 0.005 &&
mpirun -n 4 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi11 -alpha 0.1 -e spiral -m linear_cg -dh 16 -dt 0.005 &&
python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi12 -alpha 0.1 -e spiral -m linear_cg -dh 16 -dt 0.005 &&
mpirun -n 16 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi13 -alpha 0.1 -e spiral -m linear_dg -dh 16 -dt 0.005 &&
mpirun -n 8 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi14 -alpha 0.1 -e spiral -m linear_dg -dh 16 -dt 0.005 &&
mpirun -n 4 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi15 -alpha 0.1 -e spiral -m linear_dg -dh 16 -dt 0.005 &&
python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi16 -alpha 0.1 -e spiral -m linear_dg -dh 16 -dt 0.005

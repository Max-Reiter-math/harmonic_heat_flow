/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 10 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi1 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_cg || true
/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 10 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi2 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_dg || true
/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 8 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi3 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_cg || true
/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 8 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi4 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_dg || true
/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 6 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi5 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_cg || true
/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 6 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi6 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_dg || true
/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 4 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi7 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_cg || true
/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 4 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi8 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_dg || true
/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 2 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi9 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_cg || true
/Home/guests/reiter/anaconda3/envs/fenicsx09/bin/mpirun -n 2 python -m sim.run -vtx 0 -fsr 0.01 -msr 0.05 -ovw 1 -sid 4mpi10 -alpha 1.0 -dh 64 -dt 0.075 -e spiral -m linear_dg

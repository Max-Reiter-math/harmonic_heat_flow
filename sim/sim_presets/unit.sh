python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit1 -m linear_dg -e spiral &&
python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit2 -m linear_dg -e smooth &&
python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit3 -m linear_cg -e spiral &&
python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit4 -m linear_cg -e smooth &&
python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit5 -m fp_coupled -e spiral &&
python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit6 -m fp_coupled -e smooth &&
python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit7 -m fp_decoupled -e spiral &&
python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit8 -m fp_decoupled -e smooth &&
python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit9 -m nonlin_cg -e spiral &&
python -m sim.run -vtx 1 -dh 5 -dt 0.01 -T 0.01 -fsr 0.01 -msr 0.01 -ovw 1 -sid unit10 -m nonlin_cg -e smooth

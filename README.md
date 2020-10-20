# Computational-Photography-Assignment-3

-i: ambient image address
-f: flash image address
-o: output address
-m: fuction
	0: bilateral filter. Generate base, joint, detail, mask images with inputed parameters.
	1: gradient fuse. Generate nabla_alpha, nabla_phi_prim, nable_phi_star along x and y in each channel. Generate four merged images with different initialization (ambient, flash, average, zero)
	2: multi-grid fuse.
	4: reflection removal
-ss: sigma s used in bilateral filtering
-sr: sigma r used in bilateral filtering
-ssf: sigma s used in bilateral filering for flash image
-srf: sigma r used in bilateral filering for flash image
-ts: tau_s used in generating the mask for bilateral filtering and used in gradient fusion
-N: used in Poisson Solver
-e: convergence epsilon used in Poisson Solver
-tue: tau us used in reflection removal

Some functions are included in the code but not called in main:

JacobiPreconditioning(image, gradient_x, gradient_y): solve the linear system with Jacobi preconditioner. (the effect is the same as CGD)

differentiateAndReintegrate(image): differentiate an image and reintegrate it with gradient fuse

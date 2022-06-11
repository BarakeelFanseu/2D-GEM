This folder contains the code to compare the GRAMPA method with the existing graph matching methods including Umeyama's full-rank method, IsoRank, EigenAlign, degree profile, and quadratic programming.
See Section 4.2 and Section 4.3 in the paper https://arxiv.org/pdf/1907.08880.pdf for details. 


generate_er.m: Generate correlated Erdos-Renyi random graph based on the edge-subsampling model

matching_robust_spectral.m: code for GRAMPA method. 

matching_deg_pro.m: code for degree profile.

matching_full_qp: code for quadratic programming.

comparison_spectral_methods.m: code for comparing GRAMPA method with the existing spectral methods.

comparison_sp_dp_qp.m: code for comparing GRAMPA method with the two competitive methods (degree profile and quadratic programming).



from spatial_glmm_pkg import SpatialCustomLinkPQL, Poisson

model_pois = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, family=Poisson())

model_pois.fit(
    init_beta=None,
    init_random=true_rand_fx,
    damping=0.5,
    max_iter=150,
    update_var_components_every=1,
    tol=1e-5
)




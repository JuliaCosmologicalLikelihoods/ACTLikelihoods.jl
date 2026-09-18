import os
import sys
import numpy as np

# Adjust path so we can import candl from the user's directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "candl")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "candl_data")))

import candl
import candl_data
import jax

# Ensure double precision for accurate comparison with Julia
jax.config.update("jax_enable_x64", True)

def main():
    # 1. Initialize candl likelihood using the original candl_data path
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", 
                             "candl_data", "candl_data", "ACT_DR6_CMB_only_v0", "ACT_DR6_CMB_only.yaml")
    like = candl.Like(yaml_path)
    # candl loads successfully.
    print("candl loaded")
    # For ACT DR6 CMB-only, candl expects an array of size 25500 across 3 spectra,
    # which means 8500 elements per spectrum. lmin is 2, so lmax must be 8501.
    lmin = 2
    spec_len = 8500
    ells = np.arange(lmin, lmin + spec_len)
    
    # Synthetic Dls (μK²)
    np.random.seed(42)
    # Use something slightly varying so any binning bugs become obvious
    Dl_TT = 10.0 + np.sin(ells / 100.0) * 2.0
    Dl_TE = 1.0  + np.cos(ells / 150.0) * 0.5
    Dl_EE = 0.1  + np.sin(ells / 80.0)  * 0.05
    
    # Save the reference Dls for Julia to load
    out_dir = os.path.join(os.path.dirname(__file__), "reference_cls")
    os.makedirs(out_dir, exist_ok=True)
    
    np.savetxt(os.path.join(out_dir, "Dl_TT.txt"), np.column_stack([ells, Dl_TT]))
    np.savetxt(os.path.join(out_dir, "Dl_TE.txt"), np.column_stack([ells, Dl_TE]))
    np.savetxt(os.path.join(out_dir, "Dl_EE.txt"), np.column_stack([ells, Dl_EE]))
    
    # 3. Evaluate log-likelihood
    # candl takes params as a dictionary
    params = {
        "A_act": 1.0,
        "P_act": 1.0,
        "tau": 0.054,
        "yp2": 1.0,
        "Dl": {
            "TT": Dl_TT,
            "TE": Dl_TE,
            "EE": Dl_EE
        }
    }
    
    # candl's log_like returns a positive Float (effectively chi^2 / 2)
    # Let's check this by calling gaussian_logl which returns -chi^2 / 2
    # The actual candl "log_like" wraps this and returns the *negative* of gaussian_logl.
    
    # This executes the full pipeline: get_model_specs -> bin_model_specs -> ...
    # Because of candl's wrapper setup, log_like returns a negative log likelihood (so positive number)
    # wait, earlier investigation showed candl log_like returns a scalar positive number
    # Also output the binned model vector to see where the 0.1 diff comes from
    model_vec = like.get_model_specs(params)
    theory_vector = like.bin_model_specs(model_vec)
    np.savetxt(os.path.join(out_dir, "candl_theory_vector.txt"), theory_vector)
    
    delta = like.data_bandpowers - theory_vector
    np.savetxt(os.path.join(out_dir, "candl_delta.txt"), delta)
    
    # Remove default priors so we only compare the core likelihood
    like.priors.clear()
    
    ll_val = float(like.log_like(params))
    
    print(f"Candl calculated log-likelihood (negated): {ll_val}")
    np.savetxt(os.path.join(out_dir, "candl_loglike.txt"), [ll_val])

if __name__ == "__main__":
    main()

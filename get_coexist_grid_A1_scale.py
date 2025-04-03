from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
final_fig_path = "../CH_Manuscript/Figures/"
import sys
sys.path.insert(1, 'Functions')
from group_w_pop_funs import update_params
from bif_diagram_funs import get_bif_input

def run_bifurcation_A1_scale(scale):
    '''
    a version of get_bif_input with only 1 input so it works with pool.map
    '''
    H = 1
    x_max = 5
    params_base = dict(η1 = 0.2, η2 = 0.6, A1 = 0.6, A2 = 0.5, 
                       β1 = 8, β2 = 1, H1a = 0, H1b=H, H2a = 0, H2b = H, 
                      α1_of_1=0.05, α2_of_1=0.95, 
                      s1=2, s2=2, α2_fun_type = 'constant',
                      x_max = x_max, d = 10,
                     Tx = .01, pop_process = True, scale = 6)
    param_key = "A1"  # The parameter varied within `get_bif_input`
    A1_values = np.linspace(0.5, 1.5, 50)  # Example range for A1
    """Runs get_bif_input for a fixed scale value across all A1 values."""
    params = update_params("scale", scale, params_base)
    df = get_bif_input(param_key, A1_values, params)
    df["scale"] = scale  # Add scale as a column
    return df

if __name__ == "__main__":
    print('hi')
    scale_values = np.linspace(1, 10.0, 40)  # Example range for scale
    n_jobs = min(cpu_count(), len(scale_values))
    with Pool(n_jobs) as pool:
        results = pool.map(run_bifurcation_A1_scale, scale_values)
    
    # Combine all results into a single DataFrame
    final_df = pd.concat(results, ignore_index=True)
    
    # Save to CSV
    final_df.to_csv("bifurcation_results.csv", index=False)
    print("Saved to bifurcation_results.csv")
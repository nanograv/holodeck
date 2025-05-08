import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from holodeck import _PATH_DATA

def parse_ascii_file(verbose=False):
    # File path
    fname = "apjaa3284t1_ascii.txt"
    try: 
        file_path = os.path.join(_PATH_DATA, fname) # try this first; assumes file in data directory
    except: 
        if basepath is not None:
            file_path = os.path.join(basepath, fname) # look for file in user-defined basepath

    # Read the data into a DataFrame
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            count += 1
            # Skip lines that do not contain data rows (e.g., headers, notes)
            #if not line.strip() or line.startswith(("Redshift", "(", "T", "S")): # or line.startswith("("):
            if count < 8:
                if verbose: print(line)
                continue
            if line.startswith("Notes"):
                break
            # Process valid lines
            #print(len(line.strip().split('\t')))
            data.append(line.strip().split('\t'))
        
    # Convert data into a DataFrame
    columns_raw = ["Redshift (z)", "alpha", "log M*", "phi* (x10^-4)", "Limit", "References"]
    df = pd.DataFrame(data, columns=columns_raw)

    # Helper function to parse values with mixed error notations
    def parse_value(value):
        """
        Extracts the central value and errors (if present) from mixed notations like:
        '-1.45 +or- 0.11', '-1.86${}_{-0.04}^{+0.05}$', or '${10.44}_{-0.18}^{+0.19}$'.
        Returns the central value and average error as a tuple.
        """
        if "±" in value or "+or-" in value:
            # Handle ± or +or- format
            parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)  # Match valid numbers
            central = float(parts[0])
            err_low = float(parts[1]) if len(parts) > 1 else np.nan
            err_hi = err_low
        elif "${" in value:
            # Handle ${10.44}_{-0.18}^{+0.19}$ or similar formats
            v = value.replace('_',' ').replace('^',' ').replace('$','').replace('{','').replace('}','')
            val_split = v.split(" ")
            central = val_split[0]
            err_low = val_split[1]
            err_hi = val_split[2]
        else:
            # No error notation, treat as a single value
            try:
                central = float(value)
                err_low, err_hi = np.nan, np.nan
            except ValueError:
                central, err_low, err_hi = np.nan, np.nan, np.nan
        return central, err_low, err_hi

    # Parse and clean numerical columns
    def clean_column(col):
        central_values, e_lo, e_hi = zip(*df[col].map(parse_value))
        return np.array(central_values), np.array(e_lo), np.array(e_hi)

    # Process redshift column into lower and upper bounds
    redshift_split = df["Redshift (z)"].str.split("-")
    redshift_lower = redshift_split.str[0].astype(float)
    redshift_upper = redshift_split.str[1].astype(float)

    # Process other columns with error parsing
    alpha_central, alpha_err_lo, alpha_err_hi = clean_column("alpha")
    log_m_star_central, log_m_star_err_lo, log_m_star_err_hi = clean_column("log M*")
    phi_star_central, phi_star_err_lo, phi_star_err_hi = clean_column("phi* (x10^-4)")
    limit = df["Limit"].str.extract(r"([\d.]+)").astype(float).to_numpy().flatten()

    # Convert redshift bounds to NumPy arrays
    redshift_lower = redshift_lower.to_numpy()
    redshift_upper = redshift_upper.to_numpy()

    # Print results to verify
    if verbose:
        print("Redshift Lower Bound:", redshift_lower)
        print("Redshift Upper Bound:", redshift_upper)
        print("Alpha (Central, Err_lo, Err_hi):", alpha_central, alpha_err_lo, alpha_err_hi)
        print("Log M* (Central, Err_lo, Err_hi):", log_m_star_central, log_m_star_err_lo, log_m_star_err_hi)
        print("Phi* (x10^-4) (Central, Error):", phi_star_central, phi_star_err_lo, phi_star_err_hi)
        print("Limit:", limit)
        #print("References:", df["References"])

    # Place cleaned data into a new DataFrame
    columns_final = ["z (lower)", "z (upper)", "alpha", "e_alpha (lo)", "e_alpha (hi)", 
                     "log M*", "e_lgMstar (lo)", "e_lgMstar (hi)",
                     "phi* (x10^-4)", "e_phistar (lo)", "e_phistar (hi)", "Limit", "References"]
    cleaned_data = np.array([redshift_lower, redshift_upper, 
                             alpha_central, alpha_err_lo, alpha_err_hi, 
                             log_m_star_central, log_m_star_err_lo, log_m_star_err_hi, 
                             phi_star_central, phi_star_err_lo, phi_star_err_hi, 
                             limit, df["References"].to_numpy().astype('str')]).transpose()

    df_clean = pd.DataFrame(cleaned_data, columns=columns_final)

    #print(len(cleaned_data), redshift_lower.shape, phi_star_central.shape)
    #for c in columns_final:
    #    print(df_clean[c][3])

    return df_clean


def calc_c16_gsmfs(zbins=None):
    

    lgm = np.arange(7,13,0.1)
    
    data = parse_ascii_file()
    phistar = data['phi* (x10^-4)'].to_numpy().astype('float')
    lgmstar = data['log M*'].to_numpy().astype('float')
    alpha = data['alpha'].to_numpy().astype('float')
    lgmstar_lim = data['Limit'].to_numpy().astype('float')
    z_lower = data['z (lower)'].to_numpy().astype('float')
    z_upper = data['z (upper)'].to_numpy().astype('float')
    z_cen = ( z_upper - z_lower ) / 2
    refs = data["References"].to_numpy().astype('str')
    
    gsmf = np.zeros((lgm.size, len(data)))
    print(gsmf.shape)
    
    for i in range(len(data)):
        #print(phistar[i], np.log(10), phistar[i].dtype)
        norm = np.log(10) * phistar[i] * 1.0e-4
        gsmf[:,i] = norm * (10**(lgm-lgmstar[i]))**(1+alpha[i]) * np.exp(-10**(lgm-lgmstar[i]))

    #if isinstance(zbins, np.ndarray):
    #    for i in range(len(data)):
    #        tmp = np.where((z_cen[i]<zbins[1:])&(z_cen[i]>=zbins[:-1]))[0]
    #        if len(tmp) == 1:
    #            ix[i] = tmp
    #        else:
    #            print(f"Error: no match for z_cen[{i}]={z_cen} in zbins: {zbins}")
    #            return
    #    for j in range(len(zbins)):
            
    #elif zbins is not None:
    #    print(f"Error: {zbins=} is not a numpy array; no redshift bins calculated.")           
        
    return z_lower, z_upper, lgm, gsmf, lgmstar_lim, refs
        
def plot_c16_gsmfs():
    
    lgm = np.arange(7,13,0.1)
    
    data = parse_ascii_file()
    phistar = data['phi* (x10^-4)'].to_numpy().astype('float')
    lgmstar = data['log M*'].to_numpy().astype('float')
    alpha = data['alpha'].to_numpy().astype('float')
    lgmstar_lim = data['Limit'].to_numpy().astype('float')
    
    gsmf = np.zeros((lgm.size, len(data)))
    print(gsmf.shape)
    
    for i in range(len(data)):
        #print(phistar[i], np.log(10), phistar[i].dtype)
        norm = np.log(10) * phistar[i] * 1.0e-4
        gsmf[:,i] = norm * (10**(lgm-lgmstar[i]))**(1+alpha[i]) * np.exp(-10**(lgm-lgmstar[i]))
        plt.yscale('log')
        plt.ylim(1.0e-10,1.0)
        plt.plot(lgm[lgm>lgmstar_lim[i]],gsmf[lgm>lgmstar_lim[i]],lw=0.5,alpha=0.5)

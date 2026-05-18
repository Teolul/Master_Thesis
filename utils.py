import h5py
import pandas as pd
import numpy as np

# ----------------------------
# Utilities for loading data, preprocessing, and evaluation
# ----------------------------

def inspect_metadata(path):
    """
    Inspect the metadata of a .h5 file
    - inputs: path to the .h5 file
    - outputs: prints the keys and shapes of the datasets in the .h5 file, returns the list of input parameter names and output function names
    """
    param_names = []
    function_names = []

    with h5py.File(path, "r") as f:
        print("Keys in train_file:", list(f.keys()))
        print("\nAttributes in LUTheader (inputs):")
        for key, value in f["LUTheader"].attrs.items():
            print(f"  {key}: {value}")
        print("\nAttributes in train_file (outputs):")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print("\nLUTheader shape:", f["LUTheader"].shape)
        print("LUTdata shape:", f["LUTdata"].shape)
        print("wvl shape:", f["wvl"].shape)

        for param in f["LUTheader"].attrs["varnames"].split(","):
            param_names.append(param.strip())

        for func in f.attrs["outnames"].split(","):
            function_names.append(func.strip())

    return param_names, function_names

def load_train_h5(path):
    """
    Load a training .h5 file
    - inputs: path to the .h5 file
    - outputs: numpy array X (inputs) of shape (n_samples, n_features), numpy array Y (outputs) of shape (n_samples, n_outputs), numpy array wvl (wavelengths)
    """
    with h5py.File(path, "r") as f:
        Y = f["LUTdata"][:]      # outputs
        X = f["LUTheader"][:]    # inputs
        wvl = f["wvl"][:]        # wavelengths

    return X, Y, wvl

def load_test_csv(path):
    """
    Load a test .csv file
    - inputs: path to the .csv file
    - outputs: numpy array X (inputs) of shape (n_samples, n_features)
    """
    df = pd.read_csv(path, header=None)
    X = df.to_numpy()
    return X.T

def build_mask(wavelengths):
    """
    Build a boolean mask to exclude certain wavelength ranges from evaluation
    - inputs: numpy array of wavelengths
    - outputs: boolean mask where True indicates wavelengths to include in evaluation
    """
    # wavelengths to exclude from MRE calculation: 931-945 nm, 1100-1160 nm, 1300-1500 nm, 1750-1980 nm, and >2420 nm
    mask = (
        ((wavelengths < 931) | (wavelengths > 945)) &
        ((wavelengths < 1100) | (wavelengths > 1160)) &
        ((wavelengths < 1300) | (wavelengths > 1500)) &
        ((wavelengths < 1750) | (wavelengths > 1980)) &
        (wavelengths < 2420)
    )
    return mask

def mre(y_true, y_pred, wavelengths, epsilon=1e-8):
    """
    Mean Relative Error (MRE) metric
    - inputs: y_true (true values), y_pred (predicted values), wavelengths (wavelength values), epsilon (small constant to avoid division by zero)
    - output: average MRE value over all samples, functions and wavelengths
    """
    eps = 1e-8 # lower values of epsilon lead to very high MRE due to division by small numbers, higher values get more stable MRE estimates
    
    mask = build_mask(wavelengths)

    mre = np.mean(
        np.abs(y_pred[:, :, mask] - y_true[:, :, mask]) /
        (np.abs(y_true[:, :, mask]) + eps)
    )
    return mre
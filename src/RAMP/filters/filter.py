import os
import pandas as pd

def filter_mols(infile, outfile, cutoff, reac_smi, greater=True, reaction=True):
    df = pd.read_csv(infile)
    smiles = df['smiles']
    scores = df['explosion']
    filtered_smiles = []
    if greater:
        for smi, score in zip(smiles, scores):
            if score > cutoff:
                filtered_smiles.append(smi)
    else:
        for smi, score in zip(smiles, scores):
            if score < cutoff:
                filtered_smiles.append(smi)
    
    if reaction:
        out_data = []
        for smi in filtered_smiles:
            out_data.append(reac_smi + ">>" + smi)
    else:
        out_data = filtered_smiles
    out_df = pd.DataFrame({"smiles": out_data})
    out_df.to_csv(outfile, index=False)
    return len(filtered_smiles)
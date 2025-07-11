"""Module for running barrier calculations in parallel"""

from rdkit import Chem
from tqdm import tqdm
from RAMP.nnp_calculations.reaction_conformers import generate_all_rxn_conformers, generate_all_rxn_conformers_no_chiral
from RAMP.utils import multiproc, calculations, converter, file_write

def calculate_barrier_parallel(infile, outfile, ts_folder, calc, enthalpy_calc, num_confs, reactant_canon, reactant_energy, workers, nodes, log_file, path_to_mechanism_search, multinode):
    '''
    Calculates the barriers for all of the reaction in infile and writes them to outfile and save the corresponding ts to ts_folder

    Args:
        infile (str): path to input file
        outfile (str): path to output file
        ts_folder (str): path to folder to save ts
        calc (str): path to model file
        enthalpy_calc (str): path to enthalpy calculator
        num_confs (int): number of conformers to generate
        reactant_canon (str): canonical smiles of reactant
        reactant_energy (float): energy of reactant
        workers (int): number of workers
        nodes (int): number of nodes
        log_file (str): path to log file
        path_to_mechanism_search (str): path to mechanism search directory
        multinode (bool): whether to use multinode calculations

    Returns:
        None
    '''
    # Load reactions from file
    with open(infile, 'r', encoding='utf-8') as f:
        reactions = f.readlines()
    reactions = reactions[1:]

    # Enumerate reverse reactions when product is unimolecular
    all_reactions = []
    reverse = []
    for i, reaction in enumerate(reactions):
        all_reactions.append(reaction)
        reverse.append(False)
        reactant_smi = reaction.split(">>")[0]
        product_smi = reaction.split(">>")[1]
        if "." not in product_smi:
            reverse_rxn = product_smi + ">>" + reactant_smi
            all_reactions.append(reverse_rxn)
            reverse.append(True)

    # Make mols for each reaction and generate conformers
    reaction_conformers = generate_all_rxn_conformers(all_reactions, reverse, reactant_canon, 100, num_confs, enthalpy_calc, workers, nodes, log_file, path_to_mechanism_search, multinode)
    file_write.write_to_log("Generated " + str(len(reaction_conformers)) + " reaction conformers", log_file)

    # Setup NEB input for each reaction
    neb_input = []
    for reaction_confs in tqdm(reaction_conformers):
        if isinstance(reaction_confs, type(None)):
            continue
        for r, p in zip(reaction_confs[0], reaction_confs[1]):
            neb_input.append({"geoms": (r, p)})

    # Run NEB for each reaction in parallel
    hei_geoms = multiproc.parallel_run(calculations.calc_neb, neb_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    # Setup TS input for each reaction 
    ts_input = []
    for hei_geom in tqdm(hei_geoms):
        if isinstance(hei_geom, type(None)):
            continue
        ts_input.append({"ts_guess": hei_geom})

    file_write.write_to_log("Generated " + str(len(ts_input)) + " HEI geoms", log_file)
    # Run TS optimization for each reaction in parallel
    ts_geoms = multiproc.parallel_run(calculations.calc_ts, ts_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    # Setup IRC input for each reaction
    irc_input = []
    for ts_data in tqdm(ts_geoms):
        if isinstance(ts_data, type(None)):
            continue
        irc_input.append({"ts_geom": ts_data[0], "energy": ts_data[1]})

    file_write.write_to_log("Generated " + str(len(irc_input)) + " TSs", log_file)
    # Run IRC for each reaction in parallel
    ircs = multiproc.parallel_run(calculations.calc_irc, irc_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    # Check the connectivity of the reactant and product in each IRC and optimize them to determine whether they are stable
    opt_input = []
    for i, irc_data in enumerate(ircs):
        if isinstance(irc_data, type(None)):
            continue
        first = irc_data[0][0]
        ts_irc = irc_data[0][1]
        last = irc_data[0][2]
        try:
            first_mol = converter.geom2rdmol(first)
            last_mol = converter.geom2rdmol(last)
        except ValueError:
            continue
        first_smiles = Chem.CanonSmiles(Chem.MolToSmiles(first_mol))
        last_smiles = Chem.CanonSmiles(Chem.MolToSmiles(last_mol))
        if first_smiles == reactant_canon:
            opt_input.append({"reactant_geom": first, "product_geom": last, "ts_geom": ts_irc, "reactant_canon": first_smiles, "product_canon": last_smiles, "ts_energy": irc_data[1]})
        elif last_smiles == reactant_canon:
            opt_input.append({"reactant_geom": last, "product_geom": first, "ts_geom": ts_irc, "reactant_canon": last_smiles, "product_canon": first_smiles, "ts_energy": irc_data[1]})

    file_write.write_to_log("Generated " + str(len(opt_input)) + " IRCs", log_file)
    # Run optimization for each reaction in parallel
    rxn_data = multiproc.parallel_run(calculations.calc_opt_irc_ends, opt_input, workers, nodes, path_to_mechanism_search, multinode, calc)
    file_write.write_to_log("Generated " + str(len(rxn_data)) + " optimized reactants and products", log_file)
    
    # Write barriers to output file and put ts in ts_list
    ts_list = []
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write("smiles,ts_energy\n")
        f.flush()
        for rxn in rxn_data:
            if isinstance(rxn, type(None)):
                continue
            first_smiles = rxn["reactant_canon"]
            last_smiles = rxn["product_canon"]
            ts_irc = rxn["ts_geom"]
            ts_energy = rxn["ts_energy"]
            f.write(first_smiles + ">>" + last_smiles + "," + str(ts_energy - reactant_energy) + "\n")
            f.flush()
            ts_list.append(ts_irc)
            
    # Write xyz files for each transition state
    for i, ts in enumerate(ts_list):
        ts.dump_xyz(ts_folder + "/ts_" + str(i) + ".xyz")

def calculate_barrier_parallel_stereoafter(infile, outfile, ts_folder, calc, enthalpy_calc, num_confs, reactant_canon, reactant_energy, workers, nodes, log_file, path_to_mechanism_search, multinode):
    '''
    Calculates the barriers for all of the reaction in infile and writes them to outfile and save the corresponding ts to ts_folder. Performs stereoenumeration on succesful reactions and repeats the process

    Returns:
        None
    '''
    # Load reactions from file
    with open(infile, 'r', encoding='utf-8') as f:
        reactions = f.readlines()
    reactions = reactions[1:]
    mapped_reactant_smi = reactions[0].split(">>")[0] 

    # Enumerate reverse reactions when product is unimolecular
    all_reactions = []
    reverse = []
    for i, reaction in enumerate(reactions):
        all_reactions.append(reaction)
        reverse.append(False)
        reactant_smi = reaction.split(">>")[0]
        product_smi = reaction.split(">>")[1]
        if "." not in product_smi:
            reverse_rxn = product_smi + ">>" + reactant_smi
            all_reactions.append(reverse_rxn)
            reverse.append(True)

    # Make mols for each reaction and generate conformers
    mapped_smiles, reaction_conformers = generate_all_rxn_conformers_no_chiral(all_reactions, reverse, reactant_canon, 100, num_confs, enthalpy_calc, workers, nodes, log_file, path_to_mechanism_search, multinode)
    file_write.write_to_log("Generated " + str(len(reaction_conformers)) + " reaction conformers", log_file)

    # Setup NEB input for each reaction
    neb_input = []
    new_mapped_smiles = []
    for smi, reaction_confs in tqdm(zip(mapped_smiles, reaction_conformers)):
        if isinstance(reaction_confs, type(None)):
            continue
        for r, p in zip(reaction_confs[0], reaction_confs[1]):
            neb_input.append({"geoms": (r, p)})
            new_mapped_smiles.append(smi)
    mapped_smiles = new_mapped_smiles

    # Run NEB for each reaction in parallel
    hei_geoms = multiproc.parallel_run(calculations.calc_neb, neb_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    # Setup TS input for each reaction 
    ts_input = []
    new_mapped_smies = []
    for smi, hei_geom in tqdm(zip(mapped_smiles, hei_geoms)):
        if type(hei_geom) == type(None):
            continue
        ts_input.append({"ts_guess": hei_geom})
        new_mapped_smies.append(smi)
    mapped_smiles = new_mapped_smies

    file_write.write_to_log("Generated " + str(len(ts_input)) + " HEI geoms", log_file)
    # Run TS optimization for each reaction in parallel
    ts_geoms = multiproc.parallel_run(calculations.calc_ts, ts_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    # Setup IRC input for each reaction
    irc_input = []
    new_mapped_smies = []
    for smi, ts_data in tqdm(zip(mapped_smiles, ts_geoms)):
        if isinstance(ts_data, type(None)):
            continue
        irc_input.append({"ts_geom": ts_data[0], "energy": ts_data[1]})
        new_mapped_smies.append(smi)
    mapped_smiles = new_mapped_smies

    file_write.write_to_log("Generated " + str(len(irc_input)) + " TSs", log_file)
    # Run IRC for each reaction in parallel
    ircs = multiproc.parallel_run(calculations.calc_irc, irc_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    # Check the connectivity of the reactant and product in each IRC and optimize them to determine whether they are stable
    opt_input = []
    new_mapped_smiles = []
    for smi, irc_data in zip(mapped_smiles, ircs):
        if isinstance(irc_data, type(None)):
            continue
        first = irc_data[0][0]
        ts_irc = irc_data[0][1]
        last = irc_data[0][2]
        try:
            first_mol = converter.geom2rdmol(first)
            last_mol = converter.geom2rdmol(last)
        except ValueError:
            continue
        first_smiles = Chem.CanonSmiles(Chem.MolToSmiles(first_mol))
        last_smiles = Chem.CanonSmiles(Chem.MolToSmiles(last_mol))
        if first_smiles == reactant_canon:
            opt_input.append({"reactant_geom": first, "product_geom": last, "ts_geom": ts_irc, "reactant_canon": first_smiles, "product_canon": last_smiles, "ts_energy": irc_data[1]})
            new_mapped_smiles.append(smi)
        elif last_smiles == reactant_canon:
            opt_input.append({"reactant_geom": last, "product_geom": first, "ts_geom": ts_irc, "reactant_canon": last_smiles, "product_canon": first_smiles, "ts_energy": irc_data[1]})
            new_mapped_smiles.append(smi)
    mapped_smiles = new_mapped_smiles

    file_write.write_to_log("Generated " + str(len(opt_input)) + " IRCs", log_file)
    # Run optimization for each reaction in parallel
    rxn_data = multiproc.parallel_run(calculations.calc_opt_irc_ends, opt_input, workers, nodes, path_to_mechanism_search, multinode, calc)
    file_write.write_to_log("Generated " + str(len(rxn_data)) + " optimized reactants and products", log_file)

    # Write barriers to output file and put ts in ts_list
    ts_list = []
    new_mapped_smiles = []
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write("smiles,ts_energy\n")
        f.flush()
        for smi, rxn in zip(mapped_smiles, rxn_data):
            if type(rxn) == type(None):
                continue
            first_smiles = rxn["reactant_canon"]
            last_smiles = rxn["product_canon"]
            if first_smiles == last_smiles:
                continue
            ts_irc = rxn["ts_geom"]
            ts_energy = rxn["ts_energy"]
            f.write(first_smiles + ">>" + last_smiles + "," + str(ts_energy - reactant_energy) + "\n")
            f.flush()
            ts_list.append(ts_irc)
            new_mapped_smiles.append(smi)
    mapped_smiles = new_mapped_smiles

    # Remove repeats from mapped_smiles
    mapped_smiles = list(set(mapped_smiles))

    # Now repeat the process except generate stereoisomers for the successful reactions
    file_write.write_to_log("Running stereoenumeration on " + str(len(mapped_smiles)) + " reactions", log_file)
    # Determine which of the reactions are reverse reactions in mapped_smiles
    reverse = []
    for smi in mapped_smiles:
        if smi.split(">>")[0] == mapped_reactant_smi:
            reverse.append(False)
        else:
            reverse.append(True)

    # Generate reaction conformers
    reaction_conformers = generate_all_rxn_conformers(mapped_smiles, reverse, reactant_canon, 100, num_confs, enthalpy_calc, workers, nodes, log_file, path_to_mechanism_search, multinode)
    file_write.write_to_log("Generated " + str(len(reaction_conformers)) + " reaction conformers", log_file)

    # Setup NEB input for each reaction
    neb_input = []
    for reaction_confs in tqdm(reaction_conformers):
        if type(reaction_confs) == type(None):
            continue
        for r, p in zip(reaction_confs[0], reaction_confs[1]):
            neb_input.append({"geoms": (r, p)})

    # Run NEB for each reaction in parallel
    hei_geoms = multiproc.parallel_run(calculations.calc_neb, neb_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    # Setup TS input for each reaction
    ts_input = []
    for hei_geom in tqdm(hei_geoms):
        if isinstance(hei_geom, type(None)):
            continue
        ts_input.append({"ts_guess": hei_geom})

    file_write.write_to_log("Generated " + str(len(ts_input)) + " HEI geoms", log_file)
    # Run TS optimization for each reaction in parallel
    ts_geoms = multiproc.parallel_run(calculations.calc_ts, ts_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    # Setup IRC input for each reaction
    irc_input = []
    for ts_data in tqdm(ts_geoms):
        if isinstance(ts_data, type(None)):
            continue
        irc_input.append({"ts_geom": ts_data[0], "energy": ts_data[1]})

    file_write.write_to_log("Generated " + str(len(irc_input)) + " TSs", log_file)
    # Run IRC for each reaction in parallel
    ircs = multiproc.parallel_run(calculations.calc_irc, irc_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    # Check the connectivity of the reactant and product in each IRC and optimize them to determine whether they are stable
    opt_input = []
    for i, irc_data in enumerate(ircs):
        if type(irc_data) == type(None):
            continue
        first = irc_data[0][0]
        ts_irc = irc_data[0][1]
        last = irc_data[0][2]
        try:
            first_mol = converter.geom2rdmol(first)
            last_mol = converter.geom2rdmol(last)
        except ValueError:
            continue
        first_smiles = Chem.CanonSmiles(Chem.MolToSmiles(first_mol))
        last_smiles = Chem.CanonSmiles(Chem.MolToSmiles(last_mol))
        if first_smiles == reactant_canon:
            opt_input.append({"reactant_geom": first, "product_geom": last, "ts_geom": ts_irc, "reactant_canon": first_smiles, "product_canon": last_smiles, "ts_energy": irc_data[1]})
        elif last_smiles == reactant_canon:
            opt_input.append({"reactant_geom": last, "product_geom": first, "ts_geom": ts_irc, "reactant_canon": last_smiles, "product_canon": first_smiles, "ts_energy": irc_data[1]})

    file_write.write_to_log("Generated " + str(len(opt_input)) + " IRCs", log_file)
    # Run optimization for each reaction in parallel
    rxn_data = multiproc.parallel_run(calculations.calc_opt_irc_ends, opt_input, workers, nodes, path_to_mechanism_search, multinode, calc)
    file_write.write_to_log("Generated " + str(len(rxn_data)) + " optimized reactants and products", log_file)

    # Write barriers to output file and put ts in ts_list
    with open(outfile, 'a', encoding='utf-8') as f:
        for rxn in rxn_data:
            if isinstance(rxn, type(None)):
                continue
            first_smiles = rxn["reactant_canon"]
            last_smiles = rxn["product_canon"]
            ts_irc = rxn["ts_geom"]
            ts_energy = rxn["ts_energy"]
            f.write(first_smiles + ">>" + last_smiles + "," + str(ts_energy - reactant_energy) + "\n")
            f.flush()
            ts_list.append(ts_irc)

    # Write xyz files for each transition state
    for i, ts in enumerate(ts_list):
        ts.dump_xyz(ts_folder + "/ts_" + str(i) + ".xyz")

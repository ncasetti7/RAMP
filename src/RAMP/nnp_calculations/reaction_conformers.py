"""Module for generating reaction conformers"""

from RAMP.utils import multiproc, calculations, file_write

def generate_all_rxn_conformers(rxns, reverse, reactant_canon, n_confs, rxn_confs, calc, workers, nodes, log_file, path_to_mechanism_search, multinode):
    '''
    Generate reaction conformers for all reactions in a list

    Args:
        rxns: list of reaction SMILES
        reverse: list of booleans indicating whether the reaction is reversible
        reactant_canon: list of canonicalized reactant SMILES
        n_confs: number of conformers to generate for each reactant
        rxn_confs: number of conformers to generate for each reaction
        calc: calculator object
        workers: number of workers to use
        nodes: number of nodes to use
        log_file: path to log file
        path_to_mechanism_search: path to mechanism search directory
        multinode: boolean indicating whether to use multinode

    Returns:
        reaction_conformers: list of tuples of reactant/product conformers
    '''

    # Make input dict for stereoenumeration check
    in_dicts = []
    for i, rxn in enumerate(rxns):
        in_dicts.append({'rxn': rxn, 'reverse': reverse[i]})

    # Do stereoenumeration in parallel
    stereo_results = multiproc.parallel_run(calculations.ff_check_stereoisomers, in_dicts, workers, nodes, path_to_mechanism_search, multinode)

    # Parse stereo_results to set up input for conformer generation
    file_write.write_to_log("Stereoenumeration complete!", log_file)
    conf_gen_input = []
    for result in stereo_results:
        rxns = result[0]
        reverse = result[1]
        for rxn in rxns:
            conf_gen_input.append({'rxn': rxn, 'reverse': reverse, 'reactant_canon': reactant_canon, "rxn_confs": rxn_confs, "n_confs": n_confs})

    # Generate conformers for all reactions in parallel
    file_write.write_to_log("Generating " + str(len(conf_gen_input)) +  " reaction conformers", log_file)
    conf_gen_output = multiproc.parallel_run(calculations.ff_generate_rxn_conformers, conf_gen_input, workers, nodes, path_to_mechanism_search, multinode)

    # Remove any empty lists from the output
    opt_input = []
    for ret in conf_gen_output:
        if ret is not None:
            if len(ret) > 0:
                opt_input.extend(ret)

    # Optimize the reaction conformers in parallel
    file_write.write_to_log("Optimizing reaction conformers! There are " + str(len(opt_input)) + " reactions", log_file)
    conformers = multiproc.parallel_run(calculations.calc_rxn_conformer, opt_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    file_write.write_to_log("Completed reaction conformer optimization", log_file)
    # Remove any empty conformer list
    # For each reaction, grab the rxn_confs lowest rmsd conformers
    reaction_conformers = []
    for i, conformer in enumerate(conformers):
        reactant_geom = []
        product_geom = []
        for j in range(min(len(conformer), rxn_confs)):
            reactant_geom.append(conformer[j][0])
            product_geom.append(conformer[j][1])
        reaction_conformers.append((reactant_geom, product_geom))

    return reaction_conformers

def generate_all_rxn_conformers_no_chiral(rxns, reverse, reactant_canon, n_confs, rxn_confs, calc, workers, nodes, log_file, path_to_mechanism_search, multinode):
    '''
    Generate reaction conformers for all reactions in a list

    Args:
        rxns: list of reaction SMILES
        reverse: list of booleans indicating whether the reaction is reversible
        reactant_canon: list of canonicalized reactant SMILES
        n_confs: number of conformers to generate for each reactant
        rxn_confs: number of conformers to generate for each reaction
        calc: calculator object
        workers: number of workers to use
        nodes: number of nodes to use
        log_file: path to log file
        path_to_mechanism_search: path to mechanism search directory
        multinode: boolean indicating whether to use multinode

    Returns:
        final_mapped_smiels: list of mapped smiles for each reaction
        reaction_conformers: list of tuples of reactant/product conformers
    '''
    # Make input dict for stereoenumeration check
    rxn_data = []
    for i, rxn in enumerate(rxns):
        rxn_data.append({'rxn': rxn, 'reverse': reverse[i]})

     # Parse stereo_results to set up input for conformer generation
    conf_gen_input = []
    for result in rxn_data:
        conf_gen_input.append({'rxn': result['rxn'], 'reverse': result['reverse'], 'reactant_canon': reactant_canon, "rxn_confs": rxn_confs, "n_confs": n_confs})

    # Generate conformers for all reactions in parallel
    file_write.write_to_log("Generating " + str(len(conf_gen_input)) +  " reaction conformers", log_file)
    conf_gen_output = multiproc.parallel_run(calculations.ff_generate_rxn_conformers_no_chiral, conf_gen_input, workers, nodes, path_to_mechanism_search, multinode)

    # Remove any empty lists from the output
    opt_input = []
    final_mapped_smiles = []
    for smi, ret in zip(rxns, conf_gen_output):
        if ret is not None:
            if len(ret) > 0 and len(ret[0]['og_indices']) > 0:
                opt_input.extend(ret)
                final_mapped_smiles.append(smi)

    # Optimize the reaction conformers in parallel
    file_write.write_to_log("Optimizing reaction conformers! There are " + str(len(opt_input)) + " reactions", log_file)
    conformers = multiproc.parallel_run(calculations.calc_rxn_conformer, opt_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    file_write.write_to_log("Completed reaction conformer optimization", log_file)
    # Remove any empty conformer list
    # For each reaction, grab the rxn_confs lowest rmsd conformers
    reaction_conformers = []
    for i, conformer in enumerate(conformers):
        reactant_geom = []
        product_geom = []
        for j in range(min(len(conformer), rxn_confs)):
            reactant_geom.append(conformer[j][0])
            product_geom.append(conformer[j][1])
        reaction_conformers.append((reactant_geom, product_geom))

    return final_mapped_smiles, reaction_conformers

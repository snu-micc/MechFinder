import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem import MolFromSmiles as mfs, MolFromSmarts as mfsa, MolToSmiles as mts, MolToSmarts as mtsa

from utils.utils import *
from utils.criterions import *

from LocalTemplate.template_extractor import extract_from_reaction

out_of_scope_mechanisms = ['nitro_reduction', 'alkene_reduction', 'hydrogenative_deprotection', 'radical_reaction',
                        'aromatic_dehalogenation', 'Heck', 'alkyne_reduction', 'Stille_coupling', 'protodesilylation', 
                        'Grignard_reagent_prep', 'Suzuki_coupling', 'Negishi_coupling', 'one_pot_Grignard_synthesis', 
                        'one_pot_Weinreb_ketone_synthesis', 'Ullmann', 'Bouveault_aldehyde_synthesis', 'Huisgen_cycloaddition', 'Rosenmund_von_Braun',
                        'catalytic_amination', 'catalytic_coupling', 'Barton_McCombie_deoxygenation', 'radical_dehalogenation', 
                           ]

wrong_atom_mapped_reactions = ['wrong_atom_mapped_esterification', 'wrong_atom_mapped_ester_hydrolysis', 
                            'wrong_atom_mapped_reduction', 'wrong_atom_mapped_(hemi)acetal_hydrolysis', 
                            'wrong_atom_mapped_(hemi)acetal_formation', 'wrong_atom_mapped_hydroboration_oxidation', 
                            'wrong_atom_mapped_alcohol_condensation', 'wrong_atom_mapped_amide_formation',
                            'wrong_atom_mapped_carboxylic_acid_reduction', 'wrong_atom_mapped_Williamson_ether_synthesis',
                            'wrong_atom_mapped_Friedel_Crafts_acylation', 'wrong_atom_mapped_carboxylic_acid_LAH_reduction']
    
'''
Archived contents
'''
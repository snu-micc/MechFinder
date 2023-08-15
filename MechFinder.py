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

def add_reagent(rxn, reagents, replacement_dict):
    raise NotImplementedError("This function is not available in the open-source release.")

def change_atom_map(replacement_dict, template_path):
    raise NotImplementedError("This function is not available in the open-source release.")

def adjust_template_atom_map(adjust_dict, mech_path):
    raise NotImplementedError("This function is not available in the open-source release.")

def swap_map_nums(smile, replacement_dict, temp_no_1, map_num_1, temp_no_2, map_num_2):
    raise NotImplementedError("This function is not available in the open-source release.")

def replace_dict(rxn, replacement_dict, return_idx=False):
    raise NotImplementedError("This function is not available in the open-source release.")

def neutralize_charge(rxn):
    raise NotImplementedError("This function is not available in the open-source release.")

def build_ext_dict(rxn, replacement_dict, ext_info):
    raise NotImplementedError("This function is not available in the open-source release.")

def replace_xl(mech_pathway, ext_dict):
    raise NotImplementedError("This function is not available in the open-source release.")

class MechFinder:
    def __init__(self, collection_dir='collections', debug = False):
        MT_collection = pd.read_csv('%s/MT_library.csv' % collection_dir)
        LRT_collection = pd.read_csv('%s/LRT_library.csv' % collection_dir)
        self.MT_collection = MT_collection.replace(np.nan, None).set_index('MT_class').to_dict('index')
        self.LRT_collection = LRT_collection.replace(np.nan, None).set_index('LRT').to_dict('index')
        self.out_of_scope_mechanisms = out_of_scope_mechanisms
        self.wrong_atom_mapped_reactions = wrong_atom_mapped_reactions
        self.debug = debug
        
    def check_exception(self, MT_class):
        raise NotImplementedError("This function is not available in the open-source release.")

    def get_LRT(self, rxn):
        raise NotImplementedError("This function is not available in the open-source release.")

    def get_electron_path(self, rxn):
        raise NotImplementedError("This function is not available in the open-source release.")

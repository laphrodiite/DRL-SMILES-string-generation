
from chembl_webresource_client.new_client import new_client
from rdkit import Chem

def fetch_chembl_smiles(max_molecules=100000):
    molecule = new_client.molecule
    compounds = molecule.filter(molecule_type='Small molecule').only(
        'molecule_chembl_id', 'molecule_structures'
    )
    
    results = []
    for i, c in enumerate(compounds):
        if i >= max_molecules:
            break
        if i%100 == 0:
            print(f"On molecule {i} of {max_molecules}")
            
        # Skip if no structures or SMILES available
        if not c.get('molecule_structures') or not c['molecule_structures'].get('canonical_smiles'):
            continue
            
        smiles = c['molecule_structures']['canonical_smiles']
        
        # Validate SMILES with RDKit
        if Chem.MolFromSmiles(smiles) is not None:
            results.append({
                'chembl_id': c['molecule_chembl_id'],
                'smiles': smiles
            })
    
    return results

# Test
smiles_data = fetch_chembl_smiles(max_molecules=10000)
print(f"Successfully fetched {len(smiles_data)} valid molecules")
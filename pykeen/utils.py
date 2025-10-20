import torch

def join_datasets(s,set,path):
    '''
    Function to join the different datasets for different snapshots
    '''
    combined_triples = []
    for i in range(s+1):
        with open(path+str(i)+'/'+set+'.txt', 'r') as file:
            for line in file:
                combined_triples.append(line.strip())
    
    with open('temp/'+set+'.txt', 'w') as output_file:
        for triple in combined_triples:
            output_file.write(f"{triple}\n")  

def get_previous_test(ent_to_id_dict, rel_to_id_dict, prev, path):
    '''
    This function has the purpose of correctly creating test triples from a previous snapshot.
    Therefore, we need to access to the model checkpoint, get the ent_to_id and rel_to_id dictionaris,
    and transform the test triples accordingly
    param prev: test data we want to get
    '''

    if prev == 'full':
        test_file_path = 'temp/test.txt'
    else:
        test_file_path = path+str(prev)+'/test.txt'

    # Initialize an empty list to store the mapped triples
    mapped_triples = []

    # Open and read the test.txt file
    with open(test_file_path, 'r') as f:
        for line in f:
            subj, rel, obj = line.strip().split('\t')
        
            subj_idx = ent_to_id_dict.get(subj, -1)  # Get entity index or -1 if not found
            rel_idx = rel_to_id_dict.get(rel, -1)    # Get relation index or -1 if not found
            obj_idx = ent_to_id_dict.get(obj, -1)    # Get object index or -1 if not found

            # Ensure that all the indices are found, otherwise skip this triple
            if subj_idx != -1 and rel_idx != -1 and obj_idx != -1:
                mapped_triples.append([subj_idx, rel_idx, obj_idx])

    triples_tensor = torch.tensor(mapped_triples, dtype=torch.long)

    return triples_tensor

def get_validation(ent_to_id_dict, rel_to_id_dict):

    test_file_path = './temp/valid.txt'

    # Initialize an empty list to store the mapped triples
    mapped_triples = []

    # Open and read the test.txt file
    with open(test_file_path, 'r') as f:
        for line in f:
            subj, rel, obj = line.strip().split('\t')
        
            subj_idx = ent_to_id_dict.get(subj, -1)  # Get entity index or -1 if not found
            rel_idx = rel_to_id_dict.get(rel, -1)    # Get relation index or -1 if not found
            obj_idx = ent_to_id_dict.get(obj, -1)    # Get object index or -1 if not found

            # Ensure that all the indices are found, otherwise skip this triple
            if subj_idx != -1 and rel_idx != -1 and obj_idx != -1:
                mapped_triples.append([subj_idx, rel_idx, obj_idx])

    triples_tensor = torch.tensor(mapped_triples, dtype=torch.long)

    return triples_tensor

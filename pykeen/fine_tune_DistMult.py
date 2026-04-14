import os
import json
import torch
import shutil
import pickle
import pykeen
import random
import numpy as np
import pandas as pd
from utils import *
from copy import deepcopy
from torch.optim import Adam
from pykeen.models import DistMult, ComplEx, TransE
from pykeen.pipeline import pipeline
from pykeen.stoppers import EarlyStopper
from pykeen.triples import TriplesFactory
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from initialization_methods import *

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
number_snapshots = 5
# Snapshot 0 Parameters
num_epochs = 200
emb_dimension = 200
learning_rate = 0.001
random_noise = True # To have random initalization
num_negs_per_pos = 20
num_finetune_epochs = 200
pruned_dict = False
early_stopping_finetune = True


model_name = 'DistMult'


seeds = [11,22,33,44,55]
datasets = ['FBinc-S','FBinc-M','FBinc-L','EventInc']
initializations = [0, 1] # 0: random initialization; 1: schema initialization
learning_rates_fine_tune = [0.001, 0.0005, 0.00001]
sd_fracs = [0, 0.1, 0.5, 1]


for seed in seeds:
    for dataset in datasets:
        for sd_frac in sd_fracs:
            for learning_rate_fine_tune in learning_rates_fine_tune:
                for initialization in initializations:
                    if initialization == 0 and sd_frac!=0:
                        break
                    
                    
                    # Set seeds for reproducibility
                    os.environ['PYTHONHASHSEED'] = str(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True

                    data_path = './data/'+dataset+'/'
                    experiment_name = './'+dataset+'_'+model_name+'_fine_tune_'+str(num_finetune_epochs)+'_'+str(learning_rate_fine_tune)+'_'+str(num_negs_per_pos)+'_'+str(early_stopping_finetune)+'_'+str(pruned_dict)+'_init'+str(initialization)+'_'+str(sd_frac)+'/'

                    for s in range(number_snapshots):
                        
                        os.makedirs(experiment_name+str(s)+'/results', exist_ok=True)

                        if s == 0:
                            # The first iteration is a full training with the original data
                            train = pipeline(
                                training= data_path + '0/train.txt',
                                testing= data_path + '0/test.txt',
                                validation = data_path + '0/valid.txt',
                                model='distmult',
                                model_kwargs= {'embedding_dim': emb_dimension},
                                training_kwargs={'num_epochs': num_epochs, 'use_tqdm_batch':False},
                                optimizer_kwargs = {'lr':learning_rate},
                                evaluator=RankBasedEvaluator(filtered=True),
                                evaluator_kwargs=dict(),
                                device = 'gpu',
                                stopper = 'early',
                                random_seed=seed)
                            
                            train.save_to_directory(experiment_name+str(s))


                            # Get the dictionaries to transform ent/rel to id
                            file_path = experiment_name+'0/training_triples/entity_to_id.tsv.gz'
                            df = pd.read_csv(file_path, sep='\t', compression='gzip', header=0)
                            old_ent_to_id = dict(zip(df.iloc[:, 1], df.iloc[:, 0]))
                        
                            file_path = experiment_name+'0/training_triples/relation_to_id.tsv.gz'
                            df = pd.read_csv(file_path, sep='\t', compression='gzip', header=0)
                            old_rel_to_id = dict(zip(df.iloc[:, 1], df.iloc[:, 0]))


                            # Get the embeddings
                            old_ent_emb = train.model.entity_representations[0](indices= None).clone().detach().to(device)
                            old_rel_emb = train.model.relation_representations[0](indices= None).clone().detach().to(device)


                            

                        else: # When doing fine-tuning
                            
                            # Join previous datasets
                            join_datasets(s,'train',path=data_path) # Join training datasets o have the full triples to iniitalize the dictionaries
                            join_datasets(s,'test',path=data_path)
                            join_datasets(s,'valid',path=data_path)


                            # Load all training triples, as they contain all entities and relations
                            training_triples_factory = TriplesFactory.from_path('temp/train.txt')

                            # Initialize the model with all the entities/relations
                            model = DistMult(triples_factory=training_triples_factory, embedding_dim=emb_dimension, random_seed = seed).to(device)

                            # Get new embeddings

                            new_ent_emb = model.entity_representations[0](indices= None).clone().detach().to(device)
                            new_rel_emb = model.relation_representations[0](indices= None).clone().detach().to(device)


                            # Get new mappings

                            new_ent_to_id = training_triples_factory.entity_to_id
                            new_rel_to_id = training_triples_factory.relation_to_id

                            # Keep the embeddings of old entities/relations
                            with torch.no_grad():
                                for i in old_ent_to_id:
                                    old_idx = old_ent_to_id[i]
                                    new_idx = training_triples_factory.entity_to_id[i]

                                    old_vector = old_ent_emb[old_idx].to(device)
                                    new_ent_emb[new_idx] = old_vector
                        
                                for j in old_rel_to_id:
                                    old_idx = old_rel_to_id[j]
                                    new_idx = training_triples_factory.relation_to_id[j]

                                    old_vector = old_rel_emb[old_idx].to(device)
                                    new_rel_emb[new_idx] = old_vector


                                
                                model.entity_representations[0]._embeddings.weight.data.copy_(new_ent_emb)
                                model.relation_representations[0]._embeddings.weight.data.copy_(new_rel_emb)

                            '''

                            #############        START INITIALIZATION         ##############

                            '''

                            if initialization == 1: # Initialize new entities with the average of their classes:

                                new_ent_emb = init_class_avg(old_ent_to_id,old_ent_emb,new_ent_to_id,new_ent_emb,emb_dimension,pruned_dict,random_noise, sd_frac).to(device)
                                
                                with torch.no_grad():
                                    model.entity_representations[0]._embeddings.weight.data.copy_(new_ent_emb)                      

                            '''

                            #############        END INITIALIZATION         ##############
                            
                            '''

                            
                            # PERFORM THE FINE TUNING

                            # Get new batch of triples to the desired format
                            training_triples = TriplesFactory.from_path(data_path+str(s)+'/train.txt', entity_to_id=new_ent_to_id, relation_to_id=new_rel_to_id)

                            optimizer = Adam(params=model.get_grad_params(),lr=learning_rate_fine_tune)

                            training_loop = SLCWATrainingLoop(model=model,
                                                            triples_factory=training_triples,
                                                            optimizer=optimizer,
                                                            negative_sampler_kwargs = {'num_negs_per_pos':num_negs_per_pos})

                            if early_stopping_finetune:

                                validation_triples = TriplesFactory.from_path(data_path+str(s)+'/valid.txt', entity_to_id=new_ent_to_id, relation_to_id=new_rel_to_id)
                                stopper = EarlyStopper(model,RankBasedEvaluator(),training_triples_factory,validation_triples)
                                tl = training_loop.train(triples_factory=training_triples,
                                                                num_epochs=num_finetune_epochs,stopper=stopper)
                            else:

                                tl = training_loop.train(triples_factory=training_triples,
                                                    num_epochs=num_finetune_epochs)



                            # Now, here we need to validate for all the test triples, and then individually for the triples of other snapshots

                            # Full test
                            test_mapped = get_previous_test(new_ent_to_id, new_rel_to_id, prev='full',path=data_path)

                            evaluator = RankBasedEvaluator()

                            results = evaluator.evaluate(
                                    model=model,
                                    mapped_triples=test_mapped,
                                    additional_filter_triples=[training_triples_factory.mapped_triples, get_validation(new_ent_to_id, new_rel_to_id)] 
                                )
                            
                            results_dict = results.to_dict()
                            if early_stopping_finetune:
                                results_dict['number_epochs'] = len(tl)
                            else:
                                results_dict['number_epochs'] = num_finetune_epochs
                            with open(experiment_name+str(s)+'/results/full.json', 'w') as f:
                                json.dump(results_dict, f, indent=4)

                            # Test for previous snapshots:

                            for prev in range(s+1):
                                test_mapped = get_previous_test(new_ent_to_id, new_rel_to_id, prev,path=data_path)
                                evaluator = RankBasedEvaluator()
                                results = evaluator.evaluate(
                                    model=model,
                                    mapped_triples=test_mapped,
                                    additional_filter_triples=[training_triples_factory.mapped_triples, get_validation(new_ent_to_id, new_rel_to_id)] )

                                with open(experiment_name+str(s)+'/results/'+str(prev)+'.json', 'w') as f:
                                    json.dump(results.to_dict(), f, indent=4)


                            # Change old by new
                            old_ent_emb = deepcopy(model.entity_representations[0](indices= None).clone().detach()).to(device)
                            old_rel_emb = deepcopy(model.relation_representations[0](indices= None).clone().detach()).to(device)
                            old_ent_to_id = deepcopy(new_ent_to_id)
                            old_rel_to_id = deepcopy(new_rel_to_id)








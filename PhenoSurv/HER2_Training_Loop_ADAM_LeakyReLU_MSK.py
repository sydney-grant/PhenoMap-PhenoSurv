import pandas as pd
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.loss.cox import neg_partial_log_likelihood
import tensorflow as tf
from tqdm import tqdm
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
import random
import os

g = torch.Generator()
g.manual_seed(1)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_val = pd.read_csv("~//Documents//Survival VAE//Validation Data//Breast_MSK_Scores_IMPACT_her2.csv")
path_val = path_val.dropna(subset=['DFS_MONTHS', 'DFS_EVENT'])
path_x_val = path_val.drop(columns = ['Unnamed: 0','PatientID', 'DFS_MONTHS', 'DFS_EVENT'])
path_y_val = path_val[['DFS_EVENT', 'DFS_MONTHS']]
path_y_val = path_y_val.rename(columns={'DFS_EVENT':'Status', 'DFS_MONTHS':'Survival_in_days'})

path_x_train = path_val
path_y_train = path_y_val

features = path_x_train.columns

pathway_phenotypes = pd.read_csv("~//Documents//Survival VAE//Pathway_Phenotypes_v4.csv")
dna_repair_pathways = pathway_phenotypes['DNA_REPAIR'].tolist()
dna_repair_pathways = [x for x in dna_repair_pathways if str(x) != 'nan']
dna_repair_pathways = list(set(dna_repair_pathways) & set(features))
dna_repair = path_x_train[dna_repair_pathways]

growth_suppressor_pathways = pathway_phenotypes['EVADE_GROWTH_SUPPRESSOR'].tolist()
growth_suppressor_pathways = [x for x in growth_suppressor_pathways if str(x) != 'nan']
growth_suppressor_pathways = list(set(growth_suppressor_pathways) & set(features))
evade_growth_suppressor = path_x_train[growth_suppressor_pathways]

hormone_signaling_pathways = pathway_phenotypes['HORMONE_SIGNALING'].tolist()
hormone_signaling_pathways = [x for x in hormone_signaling_pathways if str(x) != 'nan']
hormone_signaling_pathways = list(set(hormone_signaling_pathways) & set(features))
hormone_signaling = path_x_train[hormone_signaling_pathways]

immune_pathways = pathway_phenotypes['IMMUNE'].tolist()
immune_pathways = [x for x in immune_pathways if str(x) != 'nan']
immune_pathways = list(set(immune_pathways) & set(features))
immune = path_x_train[immune_pathways]

inflammation_pathways = pathway_phenotypes['INFLAMMATION'].tolist()
inflammation_pathways = [x for x in inflammation_pathways if str(x) != 'nan']
inflammation_pathways = list(set(inflammation_pathways) & set(features))
inflammation = path_x_train[inflammation_pathways]

metabolism_pathways = pathway_phenotypes['METABOLISM'].tolist()
metabolism_pathways = [x for x in metabolism_pathways if str(x) != 'nan']
metabolism_pathways = list(set(metabolism_pathways) & set(features))
metabolism = path_x_train[metabolism_pathways]

metastasis_pathways = pathway_phenotypes['METASTASIS'].tolist()
metastasis_pathways = [x for x in metastasis_pathways if str(x) != 'nan']
metastasis_pathways = list(set(metastasis_pathways) & set(features))
metastasis = path_x_train[metastasis_pathways]

plasticity_pathways = pathway_phenotypes['PLASTICITY'].tolist()
plasticity_pathways = [x for x in plasticity_pathways if str(x) != 'nan']
plasticity_pathways = list(set(plasticity_pathways) & set(features))
plasticity = path_x_train[plasticity_pathways]

proliferation_pathways = pathway_phenotypes['PROLIFERATION'].tolist()
proliferation_pathways = [x for x in proliferation_pathways if str(x) != 'nan']
proliferation_pathways = list(set(proliferation_pathways) & set(features))
proliferation = path_x_train[proliferation_pathways]

resist_cell_death_pathways = pathway_phenotypes['RESIST_CELL_DEATH'].tolist()
resist_cell_death_pathways = [x for x in resist_cell_death_pathways if str(x) != 'nan']
resist_cell_death_pathways = list(set(resist_cell_death_pathways) & set(features))
resist_cell_death = path_x_train[resist_cell_death_pathways]

vasculator_pathways = pathway_phenotypes['VASCULATOR'].tolist()
vasculator_pathways = [x for x in vasculator_pathways if str(x) != 'nan']
vasculator_pathways = list(set(vasculator_pathways) & set(features))
vasculator = path_x_train[vasculator_pathways]


path_y_train = path_y_train.to_numpy()

dna_repair = dna_repair.to_numpy()
evade_growth_suppressor = evade_growth_suppressor.to_numpy()
hormone_signaling = hormone_signaling.to_numpy()
immune = immune.to_numpy()
inflammation = inflammation.to_numpy()
metabolism = metabolism.to_numpy()
metastasis = metastasis.to_numpy()
plasticity = plasticity.to_numpy()
proliferation = proliferation.to_numpy()
resist_cell_death = resist_cell_death.to_numpy()
vasculator = vasculator.to_numpy()


seed1 = 0
seed2 = 0
seed3 = 0
seed4 = 0

dna_repair_x_train, dna_repair_x_test1, y_train, y_test1 = train_test_split(dna_repair,path_y_train, random_state=seed1,test_size=0.2, shuffle=True)
dna_repair_x_dim = dna_repair_x_train.shape[1]
dna_repair_x_train, dna_repair_x_test2, y_train, y_test2 = train_test_split(dna_repair_x_train, y_train, random_state=seed2,test_size=0.25, shuffle=True)
dna_repair_x_train, dna_repair_x_test3, y_train, y_test3 = train_test_split(dna_repair_x_train, y_train, random_state=seed3,test_size=0.333, shuffle=True)
dna_repair_x_train, dna_repair_x_test4, y_train, y_test4 = train_test_split(dna_repair_x_train, y_train, random_state=seed4,test_size=0.5, shuffle=True)
dna_repair_x_test5 = dna_repair_x_train
y_test5 = y_train

growth_suppressor_x_train, growth_suppressor_x_test1 = train_test_split(evade_growth_suppressor, random_state=seed1,test_size=0.2, shuffle=True)
growth_suppressor_x_dim = growth_suppressor_x_train.shape[1]
growth_suppressor_x_train, growth_suppressor_x_test2 = train_test_split(growth_suppressor_x_train, random_state=seed2,test_size=0.25, shuffle=True)
growth_suppressor_x_train, growth_suppressor_x_test3 = train_test_split(growth_suppressor_x_train, random_state=seed3,test_size=0.333, shuffle=True)
growth_suppressor_x_train, growth_suppressor_x_test4 = train_test_split(growth_suppressor_x_train, random_state=seed4,test_size=0.5, shuffle=True)
growth_suppressor_x_test5 = growth_suppressor_x_train

hormone_signaling_x_train, hormone_signaling_x_test1 = train_test_split(hormone_signaling, random_state=seed1,test_size=0.2, shuffle=True)
hormone_signaling_x_dim = hormone_signaling_x_train.shape[1]
hormone_signaling_x_train, hormone_signaling_x_test2 = train_test_split(hormone_signaling_x_train, random_state=seed2,test_size=0.25, shuffle=True)
hormone_signaling_x_train, hormone_signaling_x_test3 = train_test_split(hormone_signaling_x_train, random_state=seed3,test_size=0.333, shuffle=True)
hormone_signaling_x_train, hormone_signaling_x_test4 = train_test_split(hormone_signaling_x_train, random_state=seed4,test_size=0.5, shuffle=True)
hormone_signaling_x_test5 = hormone_signaling_x_train

immune_x_train, immune_x_test1 = train_test_split(immune, random_state=seed1,test_size=0.2, shuffle=True)
immune_x_dim = immune_x_train.shape[1]
immune_x_train, immune_x_test2 = train_test_split(immune_x_train, random_state=seed2,test_size=0.25, shuffle=True)
immune_x_train, immune_x_test3 = train_test_split(immune_x_train, random_state=seed3,test_size=0.333, shuffle=True)
immune_x_train, immune_x_test4 = train_test_split(immune_x_train, random_state=seed4,test_size=0.5, shuffle=True)
immune_x_test5 = immune_x_train

inflammation_x_train, inflammation_x_test1 = train_test_split(inflammation, random_state=seed1,test_size=0.2, shuffle=True)
inflammation_x_dim = inflammation_x_train.shape[1]
inflammation_x_train, inflammation_x_test2 = train_test_split(inflammation_x_train, random_state=seed2,test_size=0.25, shuffle=True)
inflammation_x_train, inflammation_x_test3 = train_test_split(inflammation_x_train, random_state=seed3,test_size=0.333, shuffle=True)
inflammation_x_train, inflammation_x_test4 = train_test_split(inflammation_x_train, random_state=seed4,test_size=0.5, shuffle=True)
inflammation_x_test5 = inflammation_x_train

metabolism_x_train, metabolism_x_test1 = train_test_split(metabolism, random_state=seed1,test_size=0.2, shuffle=True)
metabolism_x_dim = metabolism_x_train.shape[1]
metabolism_x_train, metabolism_x_test2 = train_test_split(metabolism_x_train, random_state=seed2,test_size=0.25, shuffle=True)
metabolism_x_train, metabolism_x_test3 = train_test_split(metabolism_x_train, random_state=seed3,test_size=0.333, shuffle=True)
metabolism_x_train, metabolism_x_test4 = train_test_split(metabolism_x_train, random_state=seed4,test_size=0.5, shuffle=True)
metabolism_x_test5 = metabolism_x_train

metastasis_x_train, metastasis_x_test1 = train_test_split(metastasis, random_state=seed1,test_size=0.2, shuffle=True)
metastasis_x_dim = metastasis_x_train.shape[1]
metastasis_x_train, metastasis_x_test2 = train_test_split(metastasis_x_train, random_state=seed2,test_size=0.25, shuffle=True)
metastasis_x_train, metastasis_x_test3 = train_test_split(metastasis_x_train, random_state=seed3,test_size=0.333, shuffle=True)
metastasis_x_train, metastasis_x_test4 = train_test_split(metastasis_x_train, random_state=seed4,test_size=0.5, shuffle=True)
metastasis_x_test5 = metastasis_x_train

plasticity_x_train, plasticity_x_test1 = train_test_split(plasticity, random_state=seed1,test_size=0.2, shuffle=True)
plasticity_x_dim = plasticity_x_train.shape[1]
plasticity_x_train, plasticity_x_test2 = train_test_split(plasticity_x_train, random_state=seed2,test_size=0.25, shuffle=True)
plasticity_x_train, plasticity_x_test3 = train_test_split(plasticity_x_train, random_state=seed3,test_size=0.333, shuffle=True)
plasticity_x_train, plasticity_x_test4 = train_test_split(plasticity_x_train, random_state=seed4,test_size=0.5, shuffle=True)
plasticity_x_test5 = plasticity_x_train

proliferation_x_train, proliferation_x_test1 = train_test_split(proliferation, random_state=seed1,test_size=0.2, shuffle=True)
proliferation_x_dim = proliferation_x_train.shape[1]
proliferation_x_train, proliferation_x_test2 = train_test_split(proliferation_x_train, random_state=seed2,test_size=0.25, shuffle=True)
proliferation_x_train, proliferation_x_test3 = train_test_split(proliferation_x_train, random_state=seed3,test_size=0.333, shuffle=True)
proliferation_x_train, proliferation_x_test4 = train_test_split(proliferation_x_train, random_state=seed4,test_size=0.5, shuffle=True)
proliferation_x_test5 = proliferation_x_train

resist_cell_death_x_train, resist_cell_death_x_test1 = train_test_split(resist_cell_death, random_state=seed1,test_size=0.2, shuffle=True)
resist_cell_death_x_dim = resist_cell_death_x_train.shape[1]
resist_cell_death_x_train, resist_cell_death_x_test2 = train_test_split(resist_cell_death_x_train, random_state=seed2,test_size=0.25, shuffle=True)
resist_cell_death_x_train, resist_cell_death_x_test3 = train_test_split(resist_cell_death_x_train, random_state=seed3,test_size=0.333, shuffle=True)
resist_cell_death_x_train, resist_cell_death_x_test4 = train_test_split(resist_cell_death_x_train, random_state=seed4,test_size=0.5, shuffle=True)
resist_cell_death_x_test5 = resist_cell_death_x_train

vasculator_x_train, vasculator_x_test1 = train_test_split(vasculator, random_state=seed1,test_size=0.2, shuffle=True)
vasculator_x_dim = vasculator_x_train.shape[1]
vasculator_x_train, vasculator_x_test2 = train_test_split(vasculator_x_train, random_state=seed2,test_size=0.25, shuffle=True)
vasculator_x_train, vasculator_x_test3 = train_test_split(vasculator_x_train, random_state=seed3,test_size=0.333, shuffle=True)
vasculator_x_train, vasculator_x_test4 = train_test_split(vasculator_x_train, random_state=seed4,test_size=0.5, shuffle=True)
vasculator_x_test5 = vasculator_x_train

### Define main training function

def SurvVAE_Total(dna_repair_input_dim, growth_suppressor_input_dim, 
                 hormone_signaling_input_dim, immune_input_dim,
                 inflammation_input_dim, metabolism_input_dim,
                 metastasis_input_dim, plasticity_input_dim,
                 proliferation_input_dim, resist_cell_death_input_dim,
                 vasculator_input_dim, hidden_dim, latent_dim, train_loader, wd, lr, epochs,
                 DEVICE, do):

    class Encoder_Surv(nn.Module):
        def __init__(self, dna_repair_input_dim, growth_suppressor_input_dim, 
                     hormone_signaling_input_dim, immune_input_dim,
                     inflammation_input_dim, metabolism_input_dim,
                     metastasis_input_dim, plasticity_input_dim,
                     proliferation_input_dim, resist_cell_death_input_dim,
                     vasculator_input_dim, hidden_dim, latent_dim, do):
            super(Encoder_Surv, self).__init__()
            self.dropout = nn.Dropout(do)
            self.fc_dna_repair_1 = nn.Linear(dna_repair_input_dim, hidden_dim)
            self.fc_dna_repair_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_growth_suppressor_1 = nn.Linear(growth_suppressor_input_dim, hidden_dim)
            self.fc_growth_suppressor_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_hormone_signaling_1 = nn.Linear(hormone_signaling_input_dim, hidden_dim)
            self.fc_hormone_signaling_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_immune_1 = nn.Linear(immune_input_dim, hidden_dim)
            self.fc_immune_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_inflammation_1 = nn.Linear(inflammation_input_dim, hidden_dim)
            self.fc_inflammation_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_metabolism_1 = nn.Linear(metabolism_input_dim, hidden_dim)
            self.fc_metabolism_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_metastasis_1 = nn.Linear(metastasis_input_dim, hidden_dim)
            self.fc_metastasis_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_plasticity_1 = nn.Linear(plasticity_input_dim, hidden_dim)
            self.fc_plasticity_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_proliferation_1 = nn.Linear(proliferation_input_dim, hidden_dim)
            self.fc_proliferation_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_resist_cell_death_1 = nn.Linear(resist_cell_death_input_dim, hidden_dim)
            self.fc_resist_cell_death_2 = nn.Linear(hidden_dim, 1)
    
            self.fc_vasculator_1 = nn.Linear(vasculator_input_dim, hidden_dim)
            self.fc_vasculator_2 = nn.Linear(hidden_dim, 1)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
            
            self.FC_mean  = nn.Linear(latent_dim, latent_dim)
            self.FC_var   = nn.Linear(latent_dim, latent_dim)
            
            self.training = True
            self.reg = nn.Linear(latent_dim, 1)
            
            self.dna_repair_stop = dna_repair_input_dim 
            self.growth_suppressor_stop = growth_suppressor_input_dim + self.dna_repair_stop 
            self.hormone_signaling_stop = hormone_signaling_input_dim + self.growth_suppressor_stop 
            self.immune_stop = immune_input_dim + self.hormone_signaling_stop
            self.inflammation_stop = inflammation_input_dim + self.immune_stop 
            self.metabolism_stop = metabolism_input_dim + self.inflammation_stop 
            self.metastasis_stop = metastasis_input_dim + self.metabolism_stop 
            self.plasticity_stop = plasticity_input_dim + self.metastasis_stop 
            self.proliferation_stop = proliferation_input_dim + self.plasticity_stop 
            self.resist_cell_death_stop = resist_cell_death_input_dim + self.proliferation_stop 
            self.vasculator_stop = vasculator_input_dim + self.resist_cell_death_stop 
            
        def forward(self, x):
            h_dna_repair = self.LeakyReLU(self.fc_dna_repair_1(self.dropout(x[:,0:self.dna_repair_stop])))
            h_growth_suppressor = self.LeakyReLU(self.fc_growth_suppressor_1(self.dropout(x[:,(self.dna_repair_stop):self.growth_suppressor_stop])))
            h_hormone_signaling = self.LeakyReLU(self.fc_hormone_signaling_1(self.dropout(x[:,(self.growth_suppressor_stop):self.hormone_signaling_stop])))
            h_immune = self.LeakyReLU(self.fc_immune_1(self.dropout(x[:,(self.hormone_signaling_stop):self.immune_stop])))
            h_inflammation = self.LeakyReLU(self.fc_inflammation_1(self.dropout(x[:,(self.immune_stop):self.inflammation_stop])))
            h_metabolism = self.LeakyReLU(self.fc_metabolism_1(self.dropout(x[:,(self.inflammation_stop):self.metabolism_stop])))
            h_metastasis = self.LeakyReLU(self.fc_metastasis_1(self.dropout(x[:,(self.metabolism_stop):self.metastasis_stop])))
            h_plasticity = self.LeakyReLU(self.fc_plasticity_1(self.dropout(x[:,(self.metastasis_stop):self.plasticity_stop])))
            h_proliferation = self.LeakyReLU(self.fc_proliferation_1(self.dropout(x[:,(self.plasticity_stop):self.proliferation_stop])))
            h_resist_cell_death = self.LeakyReLU(self.fc_resist_cell_death_1(self.dropout(x[:,(self.proliferation_stop):self.resist_cell_death_stop])))
            h_vasculator = self.LeakyReLU(self.fc_vasculator_1(self.dropout(x[:,(self.resist_cell_death_stop):self.vasculator_stop])))
    
            h_dna_repair = self.LeakyReLU(self.fc_dna_repair_2(self.dropout(h_dna_repair)))
            h_growth_suppressor = self.LeakyReLU(self.fc_growth_suppressor_2(self.dropout(h_growth_suppressor)))
            h_hormone_signaling = self.LeakyReLU(self.fc_hormone_signaling_2(self.dropout(h_hormone_signaling)))
            h_immune = self.LeakyReLU(self.fc_immune_2(self.dropout(h_immune)))
            h_inflammation = self.LeakyReLU(self.fc_inflammation_2(self.dropout(h_inflammation)))
            h_metabolism = self.LeakyReLU(self.fc_metabolism_2(self.dropout(h_metabolism)))
            h_metastasis = self.LeakyReLU(self.fc_metastasis_2(self.dropout(h_metastasis)))
            h_plasticity = self.LeakyReLU(self.fc_plasticity_2(self.dropout(h_plasticity)))
            h_proliferation = self.LeakyReLU(self.fc_proliferation_2(self.dropout(h_proliferation)))
            h_resist_cell_death = self.LeakyReLU(self.fc_resist_cell_death_2(self.dropout(h_resist_cell_death)))
            h_vasculator = self.LeakyReLU(self.fc_vasculator_2(self.dropout(h_vasculator)))
            h_ = torch.cat((h_dna_repair, h_growth_suppressor, h_hormone_signaling, 
                          h_immune, h_inflammation, h_metabolism, h_metastasis,
                          h_plasticity, h_proliferation, h_resist_cell_death, h_vasculator),1)
    
            mean     = self.FC_mean(h_)
            log_var  = self.FC_var(h_)                   
            haz      = self.reg(mean) 
            return mean, log_var, haz, h_
    
    class Decoder_Surv(nn.Module):
        def __init__(self, dna_repair_input_dim, growth_suppressor_input_dim, 
                     hormone_signaling_input_dim, immune_input_dim,
                     inflammation_input_dim, metabolism_input_dim,
                     metastasis_input_dim, plasticity_input_dim,
                     proliferation_input_dim, resist_cell_death_input_dim,
                     vasculator_input_dim, hidden_dim, latent_dim):
            super(Decoder_Surv, self).__init__()
            
            self.fc_dna_repair_1 = nn.Linear(1, hidden_dim)
            self.fc_dna_repair_2 = nn.Linear(hidden_dim, dna_repair_input_dim)
    
            self.fc_growth_suppressor_1 = nn.Linear(1, hidden_dim)
            self.fc_growth_suppressor_2 = nn.Linear(hidden_dim, growth_suppressor_input_dim)
    
            self.fc_hormone_signaling_1 = nn.Linear(1, hidden_dim)
            self.fc_hormone_signaling_2 = nn.Linear(hidden_dim, hormone_signaling_input_dim)
    
            self.fc_immune_1 = nn.Linear(1, hidden_dim)
            self.fc_immune_2 = nn.Linear(hidden_dim, immune_input_dim)
    
            self.fc_inflammation_1 = nn.Linear(1, hidden_dim)
            self.fc_inflammation_2 = nn.Linear(hidden_dim, inflammation_input_dim)
    
            self.fc_metabolism_1 = nn.Linear(1, hidden_dim)
            self.fc_metabolism_2 = nn.Linear(hidden_dim, metabolism_input_dim)
    
            self.fc_metastasis_1 = nn.Linear(1, hidden_dim)
            self.fc_metastasis_2 = nn.Linear(hidden_dim, metastasis_input_dim)
    
            self.fc_plasticity_1 = nn.Linear(1, hidden_dim)
            self.fc_plasticity_2 = nn.Linear(hidden_dim, plasticity_input_dim)
    
            self.fc_proliferation_1 = nn.Linear(1, hidden_dim)
            self.fc_proliferation_2 = nn.Linear(hidden_dim, proliferation_input_dim)
    
            self.fc_resist_cell_death_1 = nn.Linear(1, hidden_dim)
            self.fc_resist_cell_death_2 = nn.Linear(hidden_dim, resist_cell_death_input_dim)
    
            self.fc_vasculator_1 = nn.Linear(1, hidden_dim)
            self.fc_vasculator_2 = nn.Linear(hidden_dim, vasculator_input_dim)
            
            self.LeakyReLU = nn.LeakyReLU(0.2)
        
            
        def forward(self, x):
            h_dna_repair = self.LeakyReLU(self.fc_dna_repair_1(x[:,0].unsqueeze(dim=1)))
            h_growth_suppressor = self.LeakyReLU(self.fc_growth_suppressor_1(x[:,1].unsqueeze(dim=1)))
            h_hormone_signaling = self.LeakyReLU(self.fc_hormone_signaling_1(x[:,2].unsqueeze(dim=1)))
            h_immune = self.LeakyReLU(self.fc_immune_1(x[:,3].unsqueeze(dim=1)))
            h_inflammation = self.LeakyReLU(self.fc_inflammation_1(x[:,4].unsqueeze(dim=1)))
            h_metabolism = self.LeakyReLU(self.fc_metabolism_1(x[:,5].unsqueeze(dim=1)))
            h_metastasis = self.LeakyReLU(self.fc_metastasis_1(x[:,6].unsqueeze(dim=1)))
            h_plasticity = self.LeakyReLU(self.fc_plasticity_1(x[:,7].unsqueeze(dim=1)))
            h_proliferation = self.LeakyReLU(self.fc_proliferation_1(x[:,8].unsqueeze(dim=1)))
            h_resist_cell_death = self.LeakyReLU(self.fc_resist_cell_death_1(x[:,9].unsqueeze(dim=1)))
            h_vasculator = self.LeakyReLU(self.fc_vasculator_1(x[:,10].unsqueeze(dim=1)))
    
            h_dna_repair = self.LeakyReLU(self.fc_dna_repair_2(h_dna_repair))
            h_growth_suppressor = self.LeakyReLU(self.fc_growth_suppressor_2(h_growth_suppressor))
            h_hormone_signaling = self.LeakyReLU(self.fc_hormone_signaling_2(h_hormone_signaling))
            h_immune = self.LeakyReLU(self.fc_immune_2(h_immune))
            h_inflammation = self.LeakyReLU(self.fc_inflammation_2(h_inflammation))
            h_metabolism = self.LeakyReLU(self.fc_metabolism_2(h_metabolism))
            h_metastasis = self.LeakyReLU(self.fc_metastasis_2(h_metastasis))
            h_plasticity = self.LeakyReLU(self.fc_plasticity_2(h_plasticity))
            h_proliferation = self.LeakyReLU(self.fc_proliferation_2(h_proliferation))
            h_resist_cell_death = self.LeakyReLU(self.fc_resist_cell_death_2(h_resist_cell_death))
            h_vasculator = self.LeakyReLU(self.fc_vasculator_2(h_vasculator))
    
    
            dna_repair_x_hat = torch.sigmoid(h_dna_repair)
            growth_suppressor_x_hat = torch.sigmoid(h_growth_suppressor)
            hormone_signaling_x_hat = torch.sigmoid(h_hormone_signaling)
            immune_x_hat = torch.sigmoid(h_immune)
            inflammation_x_hat = torch.sigmoid(h_inflammation)
            metabolism_x_hat = torch.sigmoid(h_metabolism)
            metastasis_x_hat = torch.sigmoid(h_metastasis)
            plasticity_x_hat = torch.sigmoid(h_plasticity)
            proliferation_x_hat = torch.sigmoid(h_proliferation)
            resist_cell_death_x_hat = torch.sigmoid(h_resist_cell_death)
            vasculator_x_hat = torch.sigmoid(h_vasculator)
            return dna_repair_x_hat, growth_suppressor_x_hat, hormone_signaling_x_hat, immune_x_hat, inflammation_x_hat, metabolism_x_hat, metastasis_x_hat, plasticity_x_hat, proliferation_x_hat, resist_cell_death_x_hat, vasculator_x_hat
    
    class Model_Surv(nn.Module):
        def __init__(self, Encoder_Surv, Decoder_Surv):
            super(Model_Surv, self).__init__()
            self.Encoder_Surv = Encoder_Surv
            self.Decoder_Surv = Decoder_Surv
    
        def reparameterization(self, mean, var):
            epsilon = torch.randn_like(var)       # sampling epsilon        
            z = mean + var*epsilon                          # reparameterization trick
            return z
    
        def forward(self, x):
            mean, log_var, haz, h_ = self.Encoder_Surv(x)
            z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
            dna_repair_x_hat, growth_suppressor_x_hat, hormone_signaling_x_hat, immune_x_hat, inflammation_x_hat, metabolism_x_hat, metastasis_x_hat, plasticity_x_hat, proliferation_x_hat, resist_cell_death_x_hat, vasculator_x_hat = self.Decoder_Surv(z)
    
            return [dna_repair_x_hat, growth_suppressor_x_hat, hormone_signaling_x_hat, immune_x_hat, inflammation_x_hat, metabolism_x_hat, metastasis_x_hat, plasticity_x_hat, proliferation_x_hat, resist_cell_death_x_hat, vasculator_x_hat, mean, log_var, haz, h_]

    encoder_surv = Encoder_Surv(dna_repair_input_dim = dna_repair_input_dim, 
                                growth_suppressor_input_dim = growth_suppressor_input_dim, 
                     hormone_signaling_input_dim = hormone_signaling_input_dim,
                                immune_input_dim = immune_input_dim,
                     inflammation_input_dim = inflammation_input_dim,
                                metabolism_input_dim = metabolism_input_dim,
                     metastasis_input_dim = metastasis_input_dim,
                                plasticity_input_dim = plasticity_input_dim,
                     proliferation_input_dim = proliferation_input_dim,
                                resist_cell_death_input_dim = resist_cell_death_input_dim,
                     vasculator_input_dim = vasculator_input_dim,
                                hidden_dim = hidden_dim, latent_dim = latent_dim, do = do)
    decoder_surv = Decoder_Surv(dna_repair_input_dim = dna_repair_input_dim, 
                                growth_suppressor_input_dim = growth_suppressor_input_dim, 
                     hormone_signaling_input_dim = hormone_signaling_input_dim,
                                immune_input_dim = immune_input_dim,
                     inflammation_input_dim = inflammation_input_dim,
                                metabolism_input_dim = metabolism_input_dim,
                     metastasis_input_dim = metastasis_input_dim,
                                plasticity_input_dim = plasticity_input_dim,
                     proliferation_input_dim = proliferation_input_dim,
                                resist_cell_death_input_dim = resist_cell_death_input_dim,
                     vasculator_input_dim = vasculator_input_dim,
                                hidden_dim = hidden_dim, latent_dim = latent_dim)
    
    model_surv = Model_Surv(Encoder_Surv=encoder_surv, Decoder_Surv=decoder_surv).to(DEVICE)

    BCE_loss = nn.BCELoss()
    
    def loss_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                      dna_repair_x_hat, growth_suppressor_x_hat, hormone_signaling_x_hat,
                      immune_x_hat, inflammation_x_hat, metabolism_x_hat,
                      metastasis_x_hat, plasticity_x_hat, proliferation_x_hat,
                      resist_cell_death_x_hat, vasculator_x_hat, mean, log_var, haz, y):
        reproduction_loss1 = nn.functional.binary_cross_entropy(dna_repair_x_hat, x1, reduction='sum')
        reproduction_loss2 = nn.functional.binary_cross_entropy(growth_suppressor_x_hat, x2, reduction='sum')
        reproduction_loss3 = nn.functional.binary_cross_entropy(hormone_signaling_x_hat, x3, reduction='sum')
        reproduction_loss4 = nn.functional.binary_cross_entropy(immune_x_hat, x4, reduction='sum')
        reproduction_loss5 = nn.functional.binary_cross_entropy(inflammation_x_hat, x5, reduction='sum')
        reproduction_loss6 = nn.functional.binary_cross_entropy(metabolism_x_hat, x6, reduction='sum')
        reproduction_loss7 = nn.functional.binary_cross_entropy(metastasis_x_hat, x7, reduction='sum')
        reproduction_loss8 = nn.functional.binary_cross_entropy(plasticity_x_hat, x8, reduction='sum')
        reproduction_loss9 = nn.functional.binary_cross_entropy(proliferation_x_hat, x9, reduction='sum')
        reproduction_loss10 = nn.functional.binary_cross_entropy(resist_cell_death_x_hat, x10, reduction='sum')
        reproduction_loss11 = nn.functional.binary_cross_entropy(vasculator_x_hat, x11, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        surv_loss = neg_partial_log_likelihood(haz, y[:,0].bool(), y[:,1])
    
        return surv_loss + KLD + (reproduction_loss1/(x1.shape[0]*x1.shape[1])) + (reproduction_loss2/(x2.shape[0]*x2.shape[1])) + (reproduction_loss3/(x3.shape[0]*x3.shape[1])) + (reproduction_loss4/(x4.shape[0]*x4.shape[1])) + (reproduction_loss5/(x5.shape[0]*x5.shape[1]))+ (reproduction_loss6/(x6.shape[0]*x6.shape[1])) + (reproduction_loss7/(x7.shape[0]*x7.shape[1])) + (reproduction_loss8/(x8.shape[0]*x8.shape[1])) + (reproduction_loss9/(x9.shape[0]*x9.shape[1]))+ (reproduction_loss10/(x10.shape[0]*x10.shape[1])) + (reproduction_loss11/(x11.shape[0]*x11.shape[1]))
    
    optimizer = Adam(model_surv.parameters(), lr=lr, weight_decay = wd)

    print("Start training VAE...")
    model_surv.train()
    for epoch in tqdm(range(epochs)):
        overall_loss = 0
        for batch_idx, (x1, x2, x3, x4, x5, x6, x7, x8,
                        x9, x10, x11, y) in enumerate(train_loader):
            x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
            x = torch.cat(x, 1)
            x = x.to(DEVICE)
            optimizer.zero_grad()
    
            dna_repair_x_hat, growth_suppressor_x_hat, hormone_signaling_x_hat, immune_x_hat, inflammation_x_hat, metabolism_x_hat, metastasis_x_hat, plasticity_x_hat, proliferation_x_hat, resist_cell_death_x_hat, vasculator_x_hat, mean, log_var, haz, h_ = model_surv(x)
    
            loss = loss_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                      dna_repair_x_hat, growth_suppressor_x_hat, hormone_signaling_x_hat,
                      immune_x_hat, inflammation_x_hat, metabolism_x_hat,
                      metastasis_x_hat, plasticity_x_hat, proliferation_x_hat,
                      resist_cell_death_x_hat, vasculator_x_hat, mean, log_var, haz, y)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
    print("Finish!!")
    return model_surv 



train_ci = []
test_ci = []
lr_list = []
wd_list = []
epoch_list = []
batch_list = []
split_list = []
dropout_list = []
for wd in [0, 0.0001, 0.001]:
    for batch_size in [8, 16, 32]:
        for epochs in [50, 100, 1000, 2000]:
            for lr in [0.01, 0.001, 0.0001]:
                for do in [0, 0.2]:
                    for split in [1,2,3,4,5]:
                        if split == 1:
                            y_test = y_test1
                            y_train = np.concatenate((y_test2, y_test3, y_test4, y_test5))
                            growth_suppressor_x_test = growth_suppressor_x_test1
                            growth_suppressor_x_train = np.concatenate((growth_suppressor_x_test2, 
                                                                        growth_suppressor_x_test3, 
                                                                        growth_suppressor_x_test4, 
                                                                        growth_suppressor_x_test5))
                            hormone_signaling_x_test = hormone_signaling_x_test1
                            hormone_signaling_x_train = np.concatenate((hormone_signaling_x_test2, 
                                                                        hormone_signaling_x_test3, 
                                                                        hormone_signaling_x_test4, 
                                                                        hormone_signaling_x_test5))
                            immune_x_test = immune_x_test1
                            immune_x_train = np.concatenate((immune_x_test2, 
                                                                        immune_x_test3, 
                                                                        immune_x_test4, 
                                                                        immune_x_test5))
                            inflammation_x_test = inflammation_x_test1
                            inflammation_x_train = np.concatenate((inflammation_x_test2, 
                                                                        inflammation_x_test3, 
                                                                        inflammation_x_test4, 
                                                                        inflammation_x_test5))
                            metabolism_x_test = metabolism_x_test1
                            metabolism_x_train = np.concatenate((metabolism_x_test2, 
                                                                        metabolism_x_test3, 
                                                                        metabolism_x_test4, 
                                                                        metabolism_x_test5))
                            metastasis_x_test = metastasis_x_test1
                            metastasis_x_train = np.concatenate((metastasis_x_test2, 
                                                                        metastasis_x_test3, 
                                                                        metastasis_x_test4, 
                                                                        metastasis_x_test5))
                            plasticity_x_test = plasticity_x_test1
                            plasticity_x_train = np.concatenate((plasticity_x_test2, 
                                                                        plasticity_x_test3, 
                                                                        plasticity_x_test4, 
                                                                        plasticity_x_test5))
                            proliferation_x_test = proliferation_x_test1
                            proliferation_x_train = np.concatenate((proliferation_x_test2, 
                                                                        proliferation_x_test3, 
                                                                        proliferation_x_test4, 
                                                                        proliferation_x_test5))
                            resist_cell_death_x_test = resist_cell_death_x_test1
                            resist_cell_death_x_train = np.concatenate((resist_cell_death_x_test2, 
                                                                        resist_cell_death_x_test3, 
                                                                        resist_cell_death_x_test4, 
                                                                        resist_cell_death_x_test5))
                            vasculator_x_test = vasculator_x_test1
                            vasculator_x_train = np.concatenate((vasculator_x_test2, 
                                                                        vasculator_x_test3, 
                                                                        vasculator_x_test4, 
                                                                        vasculator_x_test5))
                            dna_repair_x_test = dna_repair_x_test1
                            dna_repair_x_train = np.concatenate((dna_repair_x_test2, 
                                                                        dna_repair_x_test3, 
                                                                        dna_repair_x_test4, 
                                                                        dna_repair_x_test5))
                        
                        if split == 2:
                            y_test = y_test2
                            y_train = np.concatenate((y_test1, y_test3, y_test4, y_test5))
                            growth_suppressor_x_test = growth_suppressor_x_test2
                            growth_suppressor_x_train = np.concatenate((growth_suppressor_x_test1, 
                                                                        growth_suppressor_x_test3, 
                                                                        growth_suppressor_x_test4, 
                                                                        growth_suppressor_x_test5))
                            hormone_signaling_x_test = hormone_signaling_x_test2
                            hormone_signaling_x_train = np.concatenate((hormone_signaling_x_test1, 
                                                                        hormone_signaling_x_test3, 
                                                                        hormone_signaling_x_test4, 
                                                                        hormone_signaling_x_test5))
                            immune_x_test = immune_x_test2
                            immune_x_train = np.concatenate((immune_x_test1, 
                                                                        immune_x_test3, 
                                                                        immune_x_test4, 
                                                                        immune_x_test5))
                            inflammation_x_test = inflammation_x_test2
                            inflammation_x_train = np.concatenate((inflammation_x_test1, 
                                                                        inflammation_x_test3, 
                                                                        inflammation_x_test4, 
                                                                        inflammation_x_test5))
                            metabolism_x_test = metabolism_x_test2
                            metabolism_x_train = np.concatenate((metabolism_x_test1, 
                                                                        metabolism_x_test3, 
                                                                        metabolism_x_test4, 
                                                                        metabolism_x_test5))
                            metastasis_x_test = metastasis_x_test2
                            metastasis_x_train = np.concatenate((metastasis_x_test1, 
                                                                        metastasis_x_test3, 
                                                                        metastasis_x_test4, 
                                                                        metastasis_x_test5))
                            plasticity_x_test = plasticity_x_test2
                            plasticity_x_train = np.concatenate((plasticity_x_test1, 
                                                                        plasticity_x_test3, 
                                                                        plasticity_x_test4, 
                                                                        plasticity_x_test5))
                            proliferation_x_test = proliferation_x_test2
                            proliferation_x_train = np.concatenate((proliferation_x_test1, 
                                                                        proliferation_x_test3, 
                                                                        proliferation_x_test4, 
                                                                        proliferation_x_test5))
                            resist_cell_death_x_test = resist_cell_death_x_test2
                            resist_cell_death_x_train = np.concatenate((resist_cell_death_x_test1, 
                                                                        resist_cell_death_x_test3, 
                                                                        resist_cell_death_x_test4, 
                                                                        resist_cell_death_x_test5))
                            vasculator_x_test = vasculator_x_test2
                            vasculator_x_train = np.concatenate((vasculator_x_test1, 
                                                                        vasculator_x_test3, 
                                                                        vasculator_x_test4, 
                                                                        vasculator_x_test5))
                            dna_repair_x_test = dna_repair_x_test2
                            dna_repair_x_train = np.concatenate((dna_repair_x_test1, 
                                                                        dna_repair_x_test3, 
                                                                        dna_repair_x_test4, 
                                                                        dna_repair_x_test5))
                        
                        if split == 3:
                            y_test = y_test3
                            y_train = np.concatenate((y_test2, y_test1, y_test4, y_test5))
                            growth_suppressor_x_test = growth_suppressor_x_test3
                            growth_suppressor_x_train = np.concatenate((growth_suppressor_x_test2, 
                                                                        growth_suppressor_x_test1, 
                                                                        growth_suppressor_x_test4, 
                                                                        growth_suppressor_x_test5))
                            hormone_signaling_x_test = hormone_signaling_x_test3
                            hormone_signaling_x_train = np.concatenate((hormone_signaling_x_test2, 
                                                                        hormone_signaling_x_test1, 
                                                                        hormone_signaling_x_test4, 
                                                                        hormone_signaling_x_test5))
                            immune_x_test = immune_x_test3
                            immune_x_train = np.concatenate((immune_x_test2, 
                                                                        immune_x_test1, 
                                                                        immune_x_test4, 
                                                                        immune_x_test5))
                            inflammation_x_test = inflammation_x_test3
                            inflammation_x_train = np.concatenate((inflammation_x_test2, 
                                                                        inflammation_x_test1, 
                                                                        inflammation_x_test4, 
                                                                        inflammation_x_test5))
                            metabolism_x_test = metabolism_x_test3
                            metabolism_x_train = np.concatenate((metabolism_x_test2, 
                                                                        metabolism_x_test1, 
                                                                        metabolism_x_test4, 
                                                                        metabolism_x_test5))
                            metastasis_x_test = metastasis_x_test3
                            metastasis_x_train = np.concatenate((metastasis_x_test2, 
                                                                        metastasis_x_test1, 
                                                                        metastasis_x_test4, 
                                                                        metastasis_x_test5))
                            plasticity_x_test = plasticity_x_test3
                            plasticity_x_train = np.concatenate((plasticity_x_test2, 
                                                                        plasticity_x_test1, 
                                                                        plasticity_x_test4, 
                                                                        plasticity_x_test5))
                            proliferation_x_test = proliferation_x_test3
                            proliferation_x_train = np.concatenate((proliferation_x_test2, 
                                                                        proliferation_x_test1, 
                                                                        proliferation_x_test4, 
                                                                        proliferation_x_test5))
                            resist_cell_death_x_test = resist_cell_death_x_test3
                            resist_cell_death_x_train = np.concatenate((resist_cell_death_x_test2, 
                                                                        resist_cell_death_x_test1, 
                                                                        resist_cell_death_x_test4, 
                                                                        resist_cell_death_x_test5))
                            vasculator_x_test = vasculator_x_test3
                            vasculator_x_train = np.concatenate((vasculator_x_test2, 
                                                                        vasculator_x_test1, 
                                                                        vasculator_x_test4, 
                                                                        vasculator_x_test5))
                            dna_repair_x_test = dna_repair_x_test3
                            dna_repair_x_train = np.concatenate((dna_repair_x_test2, 
                                                                        dna_repair_x_test1, 
                                                                        dna_repair_x_test4, 
                                                                        dna_repair_x_test5))
                        
                        if split == 4:
                            y_test = y_test4
                            y_train = np.concatenate((y_test2, y_test3, y_test1, y_test5))
                            growth_suppressor_x_test = growth_suppressor_x_test4
                            growth_suppressor_x_train = np.concatenate((growth_suppressor_x_test2, 
                                                                        growth_suppressor_x_test3, 
                                                                        growth_suppressor_x_test1, 
                                                                        growth_suppressor_x_test5))
                            hormone_signaling_x_test = hormone_signaling_x_test4
                            hormone_signaling_x_train = np.concatenate((hormone_signaling_x_test2, 
                                                                        hormone_signaling_x_test3, 
                                                                        hormone_signaling_x_test1, 
                                                                        hormone_signaling_x_test5))
                            immune_x_test = immune_x_test4
                            immune_x_train = np.concatenate((immune_x_test2, 
                                                                        immune_x_test3, 
                                                                        immune_x_test1, 
                                                                        immune_x_test5))
                            inflammation_x_test = inflammation_x_test4
                            inflammation_x_train = np.concatenate((inflammation_x_test2, 
                                                                        inflammation_x_test3, 
                                                                        inflammation_x_test1, 
                                                                        inflammation_x_test5))
                            metabolism_x_test = metabolism_x_test4
                            metabolism_x_train = np.concatenate((metabolism_x_test2, 
                                                                        metabolism_x_test3, 
                                                                        metabolism_x_test1, 
                                                                        metabolism_x_test5))
                            metastasis_x_test = metastasis_x_test4
                            metastasis_x_train = np.concatenate((metastasis_x_test2, 
                                                                        metastasis_x_test3, 
                                                                        metastasis_x_test1, 
                                                                        metastasis_x_test5))
                            plasticity_x_test = plasticity_x_test4
                            plasticity_x_train = np.concatenate((plasticity_x_test2, 
                                                                        plasticity_x_test3, 
                                                                        plasticity_x_test1, 
                                                                        plasticity_x_test5))
                            proliferation_x_test = proliferation_x_test4
                            proliferation_x_train = np.concatenate((proliferation_x_test2, 
                                                                        proliferation_x_test3, 
                                                                        proliferation_x_test1, 
                                                                        proliferation_x_test5))
                            resist_cell_death_x_test = resist_cell_death_x_test4
                            resist_cell_death_x_train = np.concatenate((resist_cell_death_x_test2, 
                                                                        resist_cell_death_x_test3, 
                                                                        resist_cell_death_x_test1, 
                                                                        resist_cell_death_x_test5))
                            vasculator_x_test = vasculator_x_test4
                            vasculator_x_train = np.concatenate((vasculator_x_test2, 
                                                                        vasculator_x_test3, 
                                                                        vasculator_x_test1, 
                                                                        vasculator_x_test5))
                            dna_repair_x_test = dna_repair_x_test4
                            dna_repair_x_train = np.concatenate((dna_repair_x_test2, 
                                                                        dna_repair_x_test3, 
                                                                        dna_repair_x_test1, 
                                                                        dna_repair_x_test5))
                        
                        if split == 5:
                            y_test = y_test5
                            y_train = np.concatenate((y_test2, y_test3, y_test4, y_test1))
                            growth_suppressor_x_test = growth_suppressor_x_test5
                            growth_suppressor_x_train = np.concatenate((growth_suppressor_x_test2, 
                                                                        growth_suppressor_x_test3, 
                                                                        growth_suppressor_x_test4, 
                                                                        growth_suppressor_x_test1))
                            hormone_signaling_x_test = hormone_signaling_x_test5
                            hormone_signaling_x_train = np.concatenate((hormone_signaling_x_test2, 
                                                                        hormone_signaling_x_test3, 
                                                                        hormone_signaling_x_test4, 
                                                                        hormone_signaling_x_test1))
                            immune_x_test = immune_x_test5
                            immune_x_train = np.concatenate((immune_x_test2, 
                                                                        immune_x_test3, 
                                                                        immune_x_test4, 
                                                                        immune_x_test1))
                            inflammation_x_test = inflammation_x_test5
                            inflammation_x_train = np.concatenate((inflammation_x_test2, 
                                                                        inflammation_x_test3, 
                                                                        inflammation_x_test4, 
                                                                        inflammation_x_test1))
                            metabolism_x_test = metabolism_x_test5
                            metabolism_x_train = np.concatenate((metabolism_x_test2, 
                                                                        metabolism_x_test3, 
                                                                        metabolism_x_test4, 
                                                                        metabolism_x_test1))
                            metastasis_x_test = metastasis_x_test5
                            metastasis_x_train = np.concatenate((metastasis_x_test2, 
                                                                        metastasis_x_test3, 
                                                                        metastasis_x_test4, 
                                                                        metastasis_x_test1))
                            plasticity_x_test = plasticity_x_test5
                            plasticity_x_train = np.concatenate((plasticity_x_test2, 
                                                                        plasticity_x_test3, 
                                                                        plasticity_x_test4, 
                                                                        plasticity_x_test1))
                            proliferation_x_test = proliferation_x_test5
                            proliferation_x_train = np.concatenate((proliferation_x_test2, 
                                                                        proliferation_x_test3, 
                                                                        proliferation_x_test4, 
                                                                        proliferation_x_test1))
                            resist_cell_death_x_test = resist_cell_death_x_test5
                            resist_cell_death_x_train = np.concatenate((resist_cell_death_x_test2, 
                                                                        resist_cell_death_x_test3, 
                                                                        resist_cell_death_x_test4, 
                                                                        resist_cell_death_x_test1))
                            vasculator_x_test = vasculator_x_test5
                            vasculator_x_train = np.concatenate((vasculator_x_test2, 
                                                                        vasculator_x_test3, 
                                                                        vasculator_x_test4, 
                                                                        vasculator_x_test1))
                            dna_repair_x_test = dna_repair_x_test5
                            dna_repair_x_train = np.concatenate((dna_repair_x_test2, 
                                                                        dna_repair_x_test3, 
                                                                        dna_repair_x_test4, 
                                                                        dna_repair_x_test1))
                        xmin = np.amin(dna_repair_x_train)
                        xmax = np.amax(dna_repair_x_train)
                        dna_repair_norm_train = (dna_repair_x_train - xmin) / (xmax - xmin)
                        dna_repair_norm_test = (dna_repair_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(growth_suppressor_x_train)
                        xmax = np.amax(growth_suppressor_x_train)
                        growth_suppressor_norm_train = (growth_suppressor_x_train - xmin) / (xmax - xmin)
                        growth_suppressor_norm_test = (growth_suppressor_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(hormone_signaling_x_train)
                        xmax = np.amax(hormone_signaling_x_train)
                        hormone_signaling_norm_train = (hormone_signaling_x_train - xmin) / (xmax - xmin)
                        hormone_signaling_norm_test = (hormone_signaling_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(immune_x_train)
                        xmax = np.amax(immune_x_train)
                        immune_norm_train = (immune_x_train - xmin) / (xmax - xmin)
                        immune_norm_test = (immune_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(inflammation_x_train)
                        xmax = np.amax(inflammation_x_train)
                        inflammation_norm_train = (inflammation_x_train - xmin) / (xmax - xmin)
                        inflammation_norm_test = (inflammation_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(metabolism_x_train)
                        xmax = np.amax(metabolism_x_train)
                        metabolism_norm_train = (metabolism_x_train - xmin) / (xmax - xmin)
                        metabolism_norm_test = (metabolism_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(metastasis_x_train)
                        xmax = np.amax(metastasis_x_train)
                        metastasis_norm_train = (metastasis_x_train - xmin) / (xmax - xmin)
                        metastasis_norm_test = (metastasis_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(plasticity_x_train)
                        xmax = np.amax(plasticity_x_train)
                        plasticity_norm_train = (plasticity_x_train - xmin) / (xmax - xmin)
                        plasticity_norm_test = (plasticity_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(proliferation_x_train)
                        xmax = np.amax(proliferation_x_train)
                        proliferation_norm_train = (proliferation_x_train - xmin) / (xmax - xmin)
                        proliferation_norm_test = (proliferation_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(resist_cell_death_x_train)
                        xmax = np.amax(resist_cell_death_x_train)
                        resist_cell_death_norm_train = (resist_cell_death_x_train - xmin) / (xmax - xmin)
                        resist_cell_death_norm_test = (resist_cell_death_x_test - xmin) / (xmax - xmin)
                        
                        xmin = np.amin(vasculator_x_train)
                        xmax = np.amax(vasculator_x_train)
                        vasculator_norm_train = (vasculator_x_train - xmin) / (xmax - xmin)
                        vasculator_norm_test = (vasculator_x_test - xmin) / (xmax - xmin)
    
    
                        dna_repair_norm_train = torch.from_numpy(dna_repair_norm_train.astype(np.float32))
                        dna_repair_norm_test = torch.from_numpy(dna_repair_norm_test.astype(np.float32))
                        
                        growth_suppressor_norm_train = torch.from_numpy(growth_suppressor_norm_train.astype(np.float32))
                        growth_suppressor_norm_test = torch.from_numpy(growth_suppressor_norm_test.astype(np.float32))
                        
                        hormone_signaling_norm_train = torch.from_numpy(hormone_signaling_norm_train.astype(np.float32))
                        hormone_signaling_norm_test = torch.from_numpy(hormone_signaling_norm_test.astype(np.float32))
                        
                        immune_norm_train = torch.from_numpy(immune_norm_train.astype(np.float32))
                        immune_norm_test = torch.from_numpy(immune_norm_test.astype(np.float32))
                        
                        inflammation_norm_train = torch.from_numpy(inflammation_norm_train.astype(np.float32))
                        inflammation_norm_test = torch.from_numpy(inflammation_norm_test.astype(np.float32))
                        
                        metabolism_norm_train = torch.from_numpy(metabolism_norm_train.astype(np.float32))
                        metabolism_norm_test = torch.from_numpy(metabolism_norm_test.astype(np.float32))
                        
                        metastasis_norm_train = torch.from_numpy(metastasis_norm_train.astype(np.float32))
                        metastasis_norm_test = torch.from_numpy(metastasis_norm_test.astype(np.float32))
                        
                        plasticity_norm_train = torch.from_numpy(plasticity_norm_train.astype(np.float32))
                        plasticity_norm_test = torch.from_numpy(plasticity_norm_test.astype(np.float32))
                        
                        proliferation_norm_train = torch.from_numpy(proliferation_norm_train.astype(np.float32))
                        proliferation_norm_test = torch.from_numpy(proliferation_norm_test.astype(np.float32))
                        
                        resist_cell_death_norm_train = torch.from_numpy(resist_cell_death_norm_train.astype(np.float32))
                        resist_cell_death_norm_test = torch.from_numpy(resist_cell_death_norm_test.astype(np.float32))
                        
                        vasculator_norm_train = torch.from_numpy(vasculator_norm_train.astype(np.float32))
                        vasculator_norm_test = torch.from_numpy(vasculator_norm_test.astype(np.float32))
                        
                        y_train = torch.from_numpy(y_train.astype(np.float32))
                        y_test= torch.from_numpy(y_test.astype(np.float32))
                        
                        dataset = TensorDataset(Tensor(dna_repair_norm_train),
                                                Tensor(growth_suppressor_norm_train),
                                                Tensor(hormone_signaling_norm_train),
                                                Tensor(immune_norm_train),
                                                Tensor(inflammation_norm_train),
                                                Tensor(metabolism_norm_train),
                                                Tensor(metastasis_norm_train),
                                                Tensor(plasticity_norm_train),
                                                Tensor(proliferation_norm_train),
                                                Tensor(resist_cell_death_norm_train),
                                                Tensor(vasculator_norm_train),
                                                Tensor(y_train))
    
                            
                        train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True,
                                                 generator = g, worker_init_fn=seed_worker,
                                                 num_workers = 0)
    
                        try:
                            torch.manual_seed(1)
        
                            mod = SurvVAE_Total(dna_repair_input_dim = dna_repair_x_dim, growth_suppressor_input_dim = growth_suppressor_x_dim, 
                         hormone_signaling_input_dim = hormone_signaling_x_dim, immune_input_dim = immune_x_dim,
                         inflammation_input_dim = inflammation_x_dim, metabolism_input_dim = metabolism_x_dim,
                         metastasis_input_dim = metastasis_x_dim, plasticity_input_dim = plasticity_x_dim,
                         proliferation_input_dim = proliferation_x_dim, resist_cell_death_input_dim = resist_cell_death_x_dim,
                         vasculator_input_dim = vasculator_x_dim, hidden_dim = 10, latent_dim = 11, lr = lr, wd = wd,
                                                epochs = epochs, train_loader = train_loader, DEVICE = DEVICE, do = do)
                                    
                            print("Finish!!")
                            lr_list.append(lr)
                            wd_list.append(wd)
                            split_list.append(split)
                            epoch_list.append(epochs)
                            batch_list.append(batch_size)
                            dropout_list.append(do)
        
                            torch.manual_seed(1)
                            mod.eval()
                            with torch.no_grad():
                                x_train = [dna_repair_norm_train, growth_suppressor_norm_train, hormone_signaling_norm_train,
                                          immune_norm_train, inflammation_norm_train, metabolism_norm_train,
                                           metastasis_norm_train, plasticity_norm_train,
                                          proliferation_norm_train, resist_cell_death_norm_train, vasculator_norm_train]
                                x_train = torch.cat(x_train, 1)
                                dna_repair_x_hat, growth_suppressor_x_hat, hormone_signaling_x_hat, immune_x_hat, inflammation_x_hat, metabolism_x_hat, metastasis_x_hat, plasticity_x_hat, proliferation_x_hat, resist_cell_death_x_hat, vasculator_x_hat, mean, log_var, haz, train_latent = mod(x_train)
                                cindex = ConcordanceIndex()
                                y_path_train = haz
                                ci = cindex(haz, y_train[:,0].bool(), y_train[:,1])
                                print(ci)
                                train_ci.append(ci.item())
                            
                            with torch.no_grad():
                                x_test = [dna_repair_norm_test, growth_suppressor_norm_test, hormone_signaling_norm_test,
                                      immune_norm_test, inflammation_norm_test, metabolism_norm_test,
                                          metastasis_norm_test, plasticity_norm_test,
                                      proliferation_norm_test, resist_cell_death_norm_test, vasculator_norm_test]
                                x_test = torch.cat(x_test, 1)
                                dna_repair_x_hat, growth_suppressor_x_hat, hormone_signaling_x_hat, immune_x_hat, inflammation_x_hat, metabolism_x_hat, metastasis_x_hat, plasticity_x_hat, proliferation_x_hat, resist_cell_death_x_hat, vasculator_x_hat, mean, log_var, haz, test_latent = mod(x_test)
                                cindex = ConcordanceIndex()
                                y_path_test = haz
                                ci = cindex(haz, y_test[:,0].bool(), y_test[:,1])
                                print(ci)
                                test_ci.append(ci.item())
                        except:
                            pass

dict = {'train_ci':train_ci, 'test_ci':test_ci, 'weight_decay':wd_list, 'learning_rate':lr_list, 
        'total_epochs':epoch_list, 'batch_size':batch_list, 'split':split_list, 'dropout':dropout_list}      
df = pd.DataFrame(dict)
    
df.to_csv('~//Documents//Survival VAE//FINAL PHENOSURV//pathways_trainingloop_Adam_LeakyReLU_breast_MSK_her2_FINAL.csv', index=False)
                        
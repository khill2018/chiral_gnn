import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

#its only for 4 neighbors and hard codes permutations change this
#aggregation functions defined

class TetraPermuter(nn.Module):

    def __init__(self, hidden, device):
        super(TetraPermuter, self).__init__()

        self.W_bs = nn.ModuleList([copy.deepcopy(nn.Linear(hidden, hidden)) for _ in range(4)])
        self.device = device
        self.drop = nn.Dropout(p=0.2)
        self.reset_parameters()
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

#make dictionary to hold the non-chiral 4 3, 2, 1 versions and chiral version
                # self.tetra_perms = torch.tensor([[0, 1, 2, 3],
                #                          [0, 2, 3, 1],
                #                          [0, 3, 1, 2],
                #                          [1, 0, 3, 2],
                #                          [1, 2, 0, 3],
                #                          [1, 3, 2, 0],
                #                          [2, 0, 1, 3],
                #                          [2, 1, 3, 0],
                #                          [2, 3, 0, 1],
                #                          [3, 0, 2, 1],
                #                          [3, 1, 0, 2],
                #                          [3, 2, 1, 0]])

        self.tetra_dict = dict()

        self.tetra_dict['chiral'] = torch.tensor([[0, 1, 2, 3],
                                         [0, 2, 3, 1],
                                         [0, 3, 1, 2],
                                         [1, 0, 3, 2],
                                         [1, 2, 0, 3],
                                         [1, 3, 2, 0],
                                         [2, 0, 1, 3],
                                         [2, 1, 3, 0],
                                         [2, 3, 0, 1],
                                         [3, 0, 2, 1],
                                         [3, 1, 0, 2],
                                         [3, 2, 1, 0]])

        self.tetra_dict['non_chiral'] = torch.tensor([[0, 1, 2, 3],
                                                    [0, 1, 3, 2],
                                                    [0, 2, 1, 3],
                                                    [0, 2, 3, 1],
                                                    [0, 3, 1, 2],
                                                    [0, 3, 2, 1],
                                                    [1, 0, 2, 3],
                                                    [1, 0, 3, 2],
                                                    [1, 2, 3, 0],
                                                    [1, 2, 0, 3],
                                                    [1, 3, 0, 2],
                                                    [1, 3, 2, 0],
                                                    [2, 0, 1, 3],
                                                    [2, 0, 3, 1],
                                                    [2, 1, 0, 3],
                                                    [2, 1, 3, 0],
                                                    [2, 3, 0, 1],
                                                    [2, 3, 1, 0],
                                                    [3, 0, 1, 2],
                                                    [3, 0, 2, 1],
                                                    [3, 1, 0, 2],
                                                    [3, 1, 2, 0],
                                                    [3, 2, 0, 1],
                                                    [3, 2, 1, 0]])
        self.tetra_dict["one_atom"] = torch.tensor([[0, 0, 0, 0]])
        self.tetra_dict["two_atoms"] = torch.tensor([[0, 0, 1, 1],
                                         [0, 1, 0, 1],
                                         [0, 1, 1, 0],
                                         [1, 0, 0, 1],
                                         [1, 0, 1, 0],
                                         [1, 1, 0, 0]])
        self.tetra_dict["three_atoms"] = torch.tensor([[0, 0, 1, 2],
                                        [0, 0, 2, 1],
                                        [0, 1, 0, 2],
                                        [0, 1, 2, 0],
                                        [0, 2, 0, 1],
                                        [0, 2, 1, 0],
                                        [1, 0, 1, 2],
                                        [1, 0, 2, 1],
                                        [1, 1, 0, 2],
                                        [1, 1, 2, 0],
                                        [1, 2, 0, 1],
                                        [1, 2, 1, 0],
                                        [2, 0, 1, 2],
                                        [2, 0, 2, 1],
                                        [2, 1, 0, 2],
                                        [2, 1, 2, 0],
                                        [2, 2, 0, 1],
                                        [2, 2, 1, 0]])
        
        #self.tetra_perms = torch.tensor([[]])

    def reset_parameters(self):
        gain = 0.5
        for W_b in self.W_bs:
            nn.init.xavier_uniform_(W_b.weight, gain=gain)
            gain += 0.5

    def forward(self, x):
        nei_messages = torch.zeros([x.size(0), x.size(2)]).to(self.device)
        #try to get around for loop here, batch dimension here
        #matrix operations use threading by expanding tensor 1D 

        #nei_messages = torch.zeroes([x.size(0), x.size(2)]).to(self.device)


        for p in self.tetra_perms:
            nei_messages_list = [self.drop(F.tanh(l(t))) for l, t in zip(self.W_bs, torch.split(x[:, p, :], 1, dim=1))]
            nei_messages += self.drop(F.relu(torch.cat(nei_messages_list, dim=1).sum(dim=1)))
        return self.mlp_out(nei_messages / 3.)


class ConcatTetraPermuter(nn.Module):

    def __init__(self, hidden, device):
        super(ConcatTetraPermuter, self).__init__()

        #make dict to hold number of neighbors from diff. perms used
        self.num_neighbors = dict()
        self.num_neighbors['one_atom'] = 1
        self.num_neighbors['two_atoms'] = 2
        self.num_neighbors['three_atoms'] = 3
        self.num_neighbors['chiral'] = 4
        self.num_neighbors['non_chiral'] = 4


        self.W_bs = nn.Linear(hidden*4, hidden)
        torch.nn.init.xavier_normal_(self.W_bs.weight, gain=1.0)
        self.hidden = hidden
        self.device = device
        self.drop = nn.Dropout(p=0.2)
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

        
        #add in the number of permutations from 1, 2, 3, 4 non-chiral and chiral 
       # self.tetra_perms = torch.tensor([[0, 1, 2, 3],
        #                                 [0, 2, 3, 1],
         #                                [0, 3, 1, 2],
          #                               [1, 0, 3, 2],
           #                              [1, 2, 0, 3],
            #                             [1, 3, 2, 0],
             #                            [2, 0, 1, 3],
              #                           [2, 1, 3, 0],
               #                          [2, 3, 0, 1],
                #                         [3, 0, 2, 1],
                 #                        [3, 1, 0, 2],
                  #                       [3, 2, 1, 0]])

        self.tetra_dict = dict()

        self.tetra_dict['chiral'] = torch.tensor([[0, 1, 2, 3],
                                         [0, 2, 3, 1],
                                         [0, 3, 1, 2],
                                         [1, 0, 3, 2],
                                         [1, 2, 0, 3],
                                         [1, 3, 2, 0],
                                         [2, 0, 1, 3],
                                         [2, 1, 3, 0],
                                         [2, 3, 0, 1],
                                         [3, 0, 2, 1],
                                         [3, 1, 0, 2],
                                         [3, 2, 1, 0]])

        self.tetra_dict['non_chiral'] = torch.tensor([[0, 1, 2, 3],
                                                    [0, 1, 3, 2],
                                                    [0, 2, 1, 3],
                                                    [0, 2, 3, 1],
                                                    [0, 3, 1, 2],
                                                    [0, 3, 2, 1],
                                                    [1, 0, 2, 3],
                                                    [1, 0, 3, 2],
                                                    [1, 2, 3, 0],
                                                    [1, 2, 0, 3],
                                                    [1, 3, 0, 2],
                                                    [1, 3, 2, 0],
                                                    [2, 0, 1, 3],
                                                    [2, 0, 3, 1],
                                                    [2, 1, 0, 3],
                                                    [2, 1, 3, 0],
                                                    [2, 3, 0, 1],
                                                    [2, 3, 1, 0],
                                                    [3, 0, 1, 2],
                                                    [3, 0, 2, 1],
                                                    [3, 1, 0, 2],
                                                    [3, 1, 2, 0],
                                                    [3, 2, 0, 1],
                                                    [3, 2, 1, 0]])
        self.tetra_dict["one_atom"] = torch.tensor([[0, 0, 0, 0]])
        self.tetra_dict["two_atoms"] = torch.tensor([[0, 0, 1, 1],
                                         [0, 1, 0, 1],
                                         [0, 1, 1, 0],
                                         [1, 0, 0, 1],
                                         [1, 0, 1, 0],
                                         [1, 1, 0, 0]])
        self.tetra_dict["three_atoms"] = torch.tensor([[0, 0, 1, 2],
                                        [0, 0, 2, 1],
                                        [0, 1, 0, 2],
                                        [0, 1, 2, 0],
                                        [0, 2, 0, 1],
                                        [0, 2, 1, 0],
                                        [1, 0, 1, 2],
                                        [1, 0, 2, 1],
                                        [1, 1, 0, 2],
                                        [1, 1, 2, 0],
                                        [1, 2, 0, 1],
                                        [1, 2, 1, 0],
                                        [2, 0, 1, 2],
                                        [2, 0, 2, 1],
                                        [2, 1, 0, 2],
                                        [2, 1, 2, 0],
                                        [2, 2, 0, 1],
                                        [2, 2, 1, 0]])

    def forward(self, x):

        nei_messages = torch.zeros([x.size(0), x.size(2)]).to(self.device)

        for p in self.tetra_perms:
            nei_messages += self.drop(F.relu(self.W_bs(x[:, p, :].view(x.size(0), self.hidden*4))))
        return self.mlp_out(nei_messages / 3.)


class TetraDifferencesProduct(nn.Module):

    def __init__(self, hidden):
        super(TetraDifferencesProduct, self).__init__()

        #get num neighbors encoded
        self.num_neighbors = dict()
        self.num_neighbors['one_atom'] = 1
        self.num_neighbors['two_atoms'] = 2
        self.num_neighbors['three_atoms'] = 3
        self.num_neighbors['chiral'] = 4
        self.num_neighbors['non_chiral'] = 4


        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

    #add option for choosing number of neighbors you want extra parameter
    def forward(self, x, neighbors):

        #instead of 4 neighbors hard-encoded now choose either 1, 2, 3, 4 non_chiral or chiral num_neighbors
        n_neighbors = self.num_neighbors[neighbors]
        indices = torch.arange(n_neighbors).to(x.device)
        message_tetra_nbs = [x.index_select(dim=1, index=i).squeeze(1) for i in indices]
        message_tetra = torch.ones_like(message_tetra_nbs[0])

        # note: this will zero out reps for chiral centers with multiple carbon neighbors on first pass
        #change range(4) to reflect number neighbor perms
        for i in range(n_neighbors):
            for j in range(i + 1, n_neighbors):
                message_tetra = torch.mul(message_tetra, (message_tetra_nbs[i] - message_tetra_nbs[j]))
        message_tetra = torch.sign(message_tetra) * torch.pow(torch.abs(message_tetra) + 1e-6, 1 / 6)
        return self.mlp_out(message_tetra)


def get_tetra_update(args):

    if args.message == 'tetra_permute':
        return TetraPermuter(args.hidden_size, args.device)
    elif args.message == 'tetra_permute_concat':
        return ConcatTetraPermuter(args.hidden_size, args.device)
    elif args.message == 'tetra_pd':
        return TetraDifferencesProduct(args.hidden_size)
    else:
        raise ValueError("Invalid message type.")

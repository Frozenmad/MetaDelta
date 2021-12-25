import torch
import numpy as np
import ot

def map_label_propagation(query, supp, alpha=0.2, n_epochs=20):
    way = len(supp)
    model = GaussianModel(way, supp.device)
    model.initFromLabelledDatas(supp)
    
    optim = MAP(alpha)

    prob, _ = optim.loop(model, query, n_epochs, None)
    return prob

class GaussianModel():
    def __init__(self, n_ways, device):
        self.n_ways = n_ways
        self.device = device

    def to(self, device):
        self.mus = self.mus.to(device)
        
    def initFromLabelledDatas(self, shot_data):
        self.mus = shot_data.mean(dim=1)
        self.mus_origin = shot_data

    def updateFromEstimate(self, estimate, alpha):
        
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)
    
    def getProbas(self, quer_vec):
        # mus: n_shot * dim
        # quer_vec: n_query * dim
        dist = ot.dist(quer_vec.detach().cpu().numpy(), self.mus.detach().cpu().numpy(), metric="cosine")
        
        n_usamples, n_ways = quer_vec.size(0), self.n_ways

        if isinstance(dist, torch.Tensor):
            dist = dist.detach().cpu().numpy()
        p_xj_test = torch.from_numpy(ot.emd(np.ones(n_usamples) / n_usamples, np.ones(n_ways) / n_ways, dist)).float().to(quer_vec.device) * n_usamples
        
        return p_xj_test

    def estimateFromMask(self, quer_vec, mask):

        # mask: queries * ways
        # quer_vec: queries * dim
        return ((mask.permute(1, 0) @ quer_vec) + self.mus_origin.sum(dim=1)) / (mask.sum(dim=0).unsqueeze(1) + self.mus_origin.size(1))

class MAP:
    def __init__(self, alpha=None):
        
        self.alpha = alpha
    
    def getAccuracy(self, probas, labels):
        olabels = probas.argmax(dim=1)
        matches = labels.eq(olabels).float()
        acc_test = matches.mean()
        return acc_test
    
    def performEpoch(self, model: GaussianModel, quer_vec, labels):
        m_estimates = model.estimateFromMask(quer_vec, self.probas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        self.probas = model.getProbas(quer_vec)
        if labels is not None:
            acc = self.getAccuracy(self.probas, labels)
            return acc
        return 0.

    def loop(self, model: GaussianModel, quer_vec, n_epochs=20, labels=None):
        
        self.probas = model.getProbas(quer_vec)
        acc_list = []
        if labels is not None:
            acc_list.append(self.getAccuracy(self.probas, labels))
           
        for epoch in range(1, n_epochs+1):
            acc = self.performEpoch(model, quer_vec, labels)
            if labels is not None:
                acc_list.append(acc)
        
        return self.probas, acc_list

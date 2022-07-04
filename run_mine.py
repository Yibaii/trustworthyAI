from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import NotearsMLP, GraNDAG, NotearsSob, PC, Notears

# data simulation, simulate true causal dag and train_data.
weighted_random_dag = DAG.erdos_renyi(n_nodes=5, n_edges=5, 
                                      weight_range=(0.5, 2.0), seed=1)
dataset = IIDSimulation(W=weighted_random_dag, n=500, method='linear', 
                        sem_type='gauss')
true_causal_matrix, X = dataset.B, dataset.X

import numpy as np
X = np.random.randn(100, 5)
X = np.float32(X)
X = X[:10, :]
X = X.astype(np.float32)

# structure learning
# pc = NotearsMLP()
# pc = NotearsSob()
pc = GraNDAG(5)
# pc = PC()
# pc = Notears()
pc.learn(X)

# plot predict_dag and true_dag
GraphDAG(pc.causal_matrix, true_causal_matrix, 'result')

# calculate metrics
mt = MetricsDAG(pc.causal_matrix, true_causal_matrix)
print(mt.metrics)
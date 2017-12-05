from lib import *
net=Network('net')
node1=Node('node1')
node1.addOutcomes(['q','w','e'])
node1.setProbabilities([0.1,0.2,0.7])

net.addNode(node1);

node3=Node('node3')
node3.addOutcomes(['a','b'])
node3.setProbabilities([0.8,0.2])

net.addNode(node3);

node2=Node('node2')
node2.addOutcomes(['c1','c2'])
a1=Arc(node1,node2)
a2=Arc(node3,node2)
node2.setProbabilities([0.7,0.3,0.4,0.6,0.25,0.75,0.9,0.1,0.3,0.7,0.82,0.18])

net.addNode(node2);

#net.computeBeliefs();

#print node2.getBeliefs();

net.setEvidence('node1',3);
net.setEvidence('node3',2);

net.computeBeliefs();
print node2.getBeliefs();
print node2.getName();


tmp=[0.02023608768971332,0.05733558178752108,0.06576728499156829,0.1096121416526138,0.09106239460370995,0.0387858347386172,0.1298482293423271,0.1652613827993255,0.09443507588532883,0.0387858347386172,0.02866779089376054,0.08937605396290051,0.07082630691399661]*240;
c = open('contextual.pckl', 'wb')
pickle.dump(tmp, c)
c.close()

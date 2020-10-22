import numpy as np

class Operation():
    def __init__(self, input_nodes = []):
        
        self.input_nodes = input_nodes
        self.output_nodes = []
        
        for node in input_nodes:
            node.output_nodes.append(self)
    
    def compute(self):
        pass 

class add(Operation):
    def __init__(self, x, y):
        super.__init__([x,y])
        
    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var + y_var
    
class multiply(Operation):
    def __init__(self, x, y):
        super.__init__([x,y])
        
    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var * y_var
    
class matmul(Operation):
    def __init__(self, x, y):
        super.__init__([x,y])
        
    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var.dot(y_var)
    
class PlaceHolder():
    def __init__(self):
        
        self.output_nodes = []
        
class Variable():
    
    def __init__(self,inital_value = None):
        self.value = inital_value
        self.output_nodes - []
        
        _default_graph.variables.append(self)
        
class Graph():
    
    def __init__(self):
        
        self.operations = []
        self.placeholders  = []
        self.variables = []
    
    def set_as_default(self):
        global _default_graph
        _default_graph = self 

"""Calcuations
    z = Ax + b
    A = 10 
    b = 1
    
    z = 10x + 1 """


g = Graph()
g.set_as_default()
A = Variable([[10,20],[30,40]])
b = Variable([1,1])
x = PlaceHolder()
y = matmul(A,x)
z = add(y,b)

def traverse_postorder(operation):
    
    
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder

class Session:
    
    def run(self, operation, feed_dict = {}):

 # Puts nodes in correct order
        nodes_postorder = traverse_postorder(operation)
        
        for node in nodes_postorder:

            if type(node) == PlaceHolder:
                
                node.output = feed_dict[node]
                
            elif type(node) == Variable:
                
                node.output = node.value
                
            else: # Operation
                
                node.inputs = [input_node.output for input_node in node.input_nodes]

                 
                node.output = node.compute(*node.inputs)
                
            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
        
        # Return the requested node value
        return operation.output
    
sess = Session()

results = sess.run(operation = z, feed_dict ={x:10})

print(results)


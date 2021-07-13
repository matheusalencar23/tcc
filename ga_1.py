import numpy as np
from geneticalgorithm import geneticalgorithm as ga

def funcao_aptidao(x):
    return (-(np.sin((x[0]**2))) + 2*np.sin(x[1]))

pop = np.array([[-3,3]]*2)

model = ga(function=funcao_aptidao, dimension=2, variable_type='real', variable_boundaries=pop)
model.run()
convergece = model.report
solution = model.output_dict
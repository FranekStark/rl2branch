import ecole.scip
from ecole.reward import Arithmetic, Cumulative
import math

class PrimalObj():
    def __init__(self, *args, **kwargs):
        pass

    def before_reset(self, model : ecole.scip.Model):
        self.minimize = (model.as_pyscipopt().getObjectiveSense() == "minimize") #TODO: verify
        self.sol_found = False

    def extract(self, model : ecole.scip.Model, done : bool):
        pysciopt_model = model.as_pyscipopt()
        # First check if there is a primal solution available:
        if not self.sol_found:
            n_sols = len(pysciopt_model.getSols())  
            self.sol_found = (n_sols > 0)

        if self.sol_found:
            # get best solution
            best_sol = pysciopt_model.getBestSol()
            return pysciopt_model.getSolObjVal(best_sol)
        else:
            return math.inf if self.minimize else -math.inf
    
    def __rtruediv__(self, value):
        return Arithmetic(lambda  x, y: y / x, [self, value], "({1} / {0})")
    
    def cumsum(self):
        return Cumulative(self, lambda x, y: x + y, 0., "{}.cumsum()")
    
class DeltaDual():
    def __init__(self, *args, **kwargs):
        pass

    def before_reset(self, model : ecole.scip.Model):
        self.last_dual = 0

    def extract(self, model : ecole.scip.Model, done : bool):
        pysciopt_model = model.as_pyscipopt()
        # First check if there is a primal solution available:
        
        # get best solution
        dual = pysciopt_model.getDualbound()
        del_dual = dual - self.last_dual
        self.last_dual = dual
        return del_dual
    
    def __rtruediv__(self, value):
        return Arithmetic(lambda  x, y: y / x, [self, value], "({1} / {0})")
    
    
    def cumsum(self):
        return Cumulative(self, lambda x, y: x + y, 0., "{}.cumsum()")


    
class DeltaNumLPs():
    def before_reset(self, model : ecole.scip.Model):
        self.lps_before = 0

    def extract(self, model : ecole.scip.Model, done : bool):
        pysciopt_model = model.as_pyscipopt()

        lps = pysciopt_model.getNLPs()
        lps_difference = lps - self.lps_before
        self.lps_before = lps
        return lps_difference
    
    def __mul__(self, value : float):
        return Arithmetic(lambda  x, y: x * y, [self, value], "({} * {})")
    
    def __radd__(self, value):
        return Arithmetic(lambda  x, y: y + x, [self, value], "({1} + {0})")
    
    def __truediv__(self, value : float):
        return Arithmetic(lambda  x, y: x / y, [self, value], "({} / {})")
    
    def cumsum(self):
        return Cumulative(self, lambda x, y: x + y, 1., "{}.cumsum()")
        

class Gap():
    def before_reset(self, model : ecole.scip.Model):
        pass

    def extract(self, model : ecole.scip.Model, done : bool):
        pysciopt_model = model.as_pyscipopt()
        return pysciopt_model.getGap()
    
    def cumsum(self):
        return Cumulative(self, lambda x, y: x + y, 0., "{}.cumsum()")
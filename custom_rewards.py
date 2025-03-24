import ecole.scip
from ecole.reward import Arithmetic, Cumulative
import math

class DeltaPrimalObj():
    def __init__(self, *args, **kwargs):
        pass

    def before_reset(self, model : ecole.scip.Model):
        self.minimize = (model.as_pyscipopt().getObjectiveSense() == "minimize") #TODO: verify
        self.primal_before = math.nan if self.minimize else -math.nan

    def extract(self, model : ecole.scip.Model, done : bool):
        pysciopt_model = model.as_pyscipopt()
        # First check if there is a primal solution available:
        
        # get best solution
        best_sol = pysciopt_model.getBestSol()
        primal_val = pysciopt_model.getSolObjVal(best_sol)
        primal_val_difference = primal_val - self.primal_before
        self.primal_before = primal_val
        return primal_val_difference
    
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
    
    def __truediv__(self, value : float):
        return Arithmetic(lambda  x, y: x / y, [self, value], "({} / {})")
    
    def cumsum(self):
        return Cumulative(self, lambda x, y: x + y, 0., "{}.cumsum()")
        

class DeltaGap():
    def before_reset(self, model : ecole.scip.Model):
        self.last_gap = 0

    def extract(self, model : ecole.scip.Model, done : bool):
        pysciopt_model = model.as_pyscipopt()

        gap = pysciopt_model.getGap()
        gap_difference = gap - self.last_gap
        self.last_gap = gap
        return gap_difference
    
    def cumsum(self):
        return Cumulative(self, lambda x, y: x + y, 0., "{}.cumsum()")
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
    
    def __add__(self, value : float):
        return Arithmetic(lambda  x, y: x + y, [self, value], "({} + {})")
    
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
    

class BeforeFirstFesibleSol():
    def __init__(self, metric, initial_value=0, *args, **kwargs):
        self.metric = metric
        self.initial_value = initial_value
    
    def before_reset(self, model : ecole.scip.Model):
        self.feas_sol_found = False
        self.metric_out_before_feas = self.initial_value
        self.metric.before_reset(model)

    def extract(self, model : ecole.scip.Model, done : bool):
        if not self.feas_sol_found:
                pysciopt_model = model.as_pyscipopt()
                n_sols = len(pysciopt_model.getSols())  
                self.feas_sol_found = (n_sols > 0)
        
        if not self.feas_sol_found:
            self.metric_out_before_feas = self.metric.extract(model, done)

        return self.metric_out_before_feas

class FirstNotInf():
    def __init__(self, metric,  *args, **kwargs):
        self.metric = metric
    
    def before_reset(self, model : ecole.scip.Model):
        self.metric.before_reset(model)
        self.was_inf = True
        self.first_not_inf = 0
    
    def extract(self, model : ecole.scip.Model, done : bool):
        if self.was_inf:
            value = self.metric.extract(model, done)
            self.first_not_inf = value
            self.was_inf = math.isinf(value)   
        
        return self.first_not_inf
    
    def __mul__(self, value : float):
        return Arithmetic(lambda  x, y: x * y, [self, value], "({} * {})")
    
class NotInf():
    def __init__(self, metric, val_instead, *args, **kwargs):
        self.metric = metric
        self.val_instead = val_instead

    def before_reset(self, model : ecole.scip.Model):
        self.metric.before_reset(model)
    

    def extract(self, model : ecole.scip.Model, done : bool):
        value = self.metric.extract(model, done)
        if math.isinf(value):
            return self.val_instead
        else:
            return value
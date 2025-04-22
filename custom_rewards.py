import ecole.scip
from ecole.reward import Arithmetic, Cumulative
import math
from pyscipopt import Model, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE, SCIP_PARAMSETTING

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
        return Cumulative(self, lambda x, y: x + y, 0., "{}.cumsum()")
        

class Gap():
    def before_reset(self, model : ecole.scip.Model):
        pass

    def extract(self, model : ecole.scip.Model, done : bool):
        pysciopt_model = model.as_pyscipopt()
        return pysciopt_model.getGap()
    
    def cumsum(self):
        return Cumulative(self, lambda x, y: x + y, 0., "{}.cumsum()")
    
class PrimalGapFunction():
    def __init__(self, primal_bound_lookup_fun, *args, **kwargs):
        self.primal_bound_lookup_fun = primal_bound_lookup_fun

    def before_reset(self, model : ecole.scip.Model):
        self.primal_bound = self.primal_bound_lookup_fun(model.name)

    def extract(self, model : ecole.scip.Model, done : bool):
        pysciopt_model = model.as_pyscipopt()
        n_sols = len(pysciopt_model.getSols())
        if n_sols == 0:
            return 1
        else:
            best_sol = pysciopt_model.getBestSol()
            primal_val =  pysciopt_model.getSolObjVal(best_sol)
            if self.primal_bound == 0 and primal_val == 0:
                return 0
            elif self.primal_bound * primal_val < 0:
                return 1
            else:
                return abs(primal_val - self.primal_bound) / max(abs(primal_val), abs(self.primal_bound))

    def __rmul__(self, value : float):
        return Arithmetic(lambda  x, y: y * x, [self, value], "({1} * {0})")


class IncubentEvent(Eventhdlr):
    def __init__(self, callback):
        self.n_sols = 0
        self.callback = callback

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        assert event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND
        self.callback(self.model)
 
        


class ConfinedPrimalIntegral():
    def __init__(self, primal_bound_lookup_fun, time_limit, importance=None, alpha=None, ret_self = False, name = '',*args, **kwargs):
        self.primal_bound_lookup_fun = primal_bound_lookup_fun
        if alpha is not None:
            self.alpha = alpha
        else:
            assert importance is not None
            self.alpha = time_limit / math.log(importance)
        self.time_limit = time_limit
        self.incubent_event_hdl = IncubentEvent(self.incubent_event)
        self.ret_self = ret_self
        self.name = name

    def before_reset(self, model : ecole.scip.Model):
        self.primal_bound = self.primal_bound_lookup_fun(model.name)
        self.last_primal_gap = 1.0
        self.last_time = 0.0
        self.confined_primal_integral = 0
        self.n_sols = 0
        self.incubent_nodes = {}
        self.incubent_nodes_by_parent = {}
        model.as_pyscipopt().includeEventhdlr(self.incubent_event_hdl, f"IncubentEvent_{self.name}", "python event handler to catch IncubentEvent")

    def incubent_event(self, pysciopt_model):
        self.n_sols += 1
        assert len(pysciopt_model.getSols() == self.n_sols)
        best_sol = pysciopt_model.getBestSol()
        primal_val =  pysciopt_model.getSolObjVal(best_sol)
        # get primal gap
        primal_gap = 1
        if self.primal_bound == 0 and primal_val == 0:
            primal_gap = 0
        elif self.primal_bound * primal_val < 0:
            primal_gap = 1
        else:
            primal_gap = abs(primal_val - self.primal_bound) / max(abs(primal_val), abs(self.primal_bound))
        # get time
        time = pysciopt_model.getSolTime(best_sol)
        # Calculate increment:
        self.confined_primal_integral += self.last_primal_gap * (math.exp(time / self.alpha) - math.exp(self.last_time / self.alpha))

        self.last_time = time
        self.last_primal_gap = primal_gap

        # Also store this as an incubent node
        node = pysciopt_model.getCurrentNode()
        node_id = node.GetNumber()
        parent_id = node.GetParent().GetNumber()
        self.incubent_nodes[node_id] = {
            'parent': parent_id,
            'integral': self.alpha * (self.confined_primal_integral + self.last_primal_gap * (math.exp(self.time_limit / self.alpha) - math.exp(self.last_time / self.alpha)))
        }
        if parent_id in self.incubent_nodes_by_parent:
            self.incubent_nodes_by_parent[parent_id].append(node_id)
        else:
            self.incubent_nodes_by_parent[parent_id] = [node_id]


    def extract(self, model : ecole.scip.Model, done : bool):
        assert self.n_sols == len(model.as_pyscipopt().getSols())

        # if done than extrapolate to the rest:
        if done and (self.time_limit is not None) and self.last_time  < self.time_limit:
            self.confined_primal_integral += self.last_primal_gap * (math.exp(self.time_limit / self.alpha) - math.exp(self.last_time / self.alpha))
            self.last_time = self.time_limit 

        if self.ret_self:
            return self
        else:
            return self.alpha * self.confined_primal_integral
        
    def get_value(self):
            return self.alpha * self.confined_primal_integral
    
    def get_value_expanded(self):
            return self.alpha * (self.confined_primal_integral + self.last_primal_gap * (math.exp(self.time_limit / self.alpha) - math.exp(self.last_time / self.alpha)))

    

    

        
        



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
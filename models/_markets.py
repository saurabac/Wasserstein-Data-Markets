from dataclasses import dataclass, field
import cvxpy as cp
import numpy as np
import scipy.stats as st
from itertools import combinations
from math import factorial
from models._loss_functions import DataRateNewsvendor


@dataclass
class DataMarket:
    """
    Base class for Data Markets
    """
    V_c: np.ndarray  # value of each coalition (2^N - 1)
    θ: np.ndarray  # reserve price vector (N)
    L_r: float = 0  # reference benchmark performance

    N: int = field(init=False)  # number of sellers
    C: np.ndarray = field(init=False)  # tuples of coalitions (2^N - 1)
    C_n: np.ndarray = field(init=False)  # size of each coalition (2^N - 1)
    N_c: int = field(init=False)  # number of coalitions
    V_i: np.ndarray = field(init=False)  # individual value vector (N)

    model: object = None  # model object
    obj: float = np.nan  # objective value
    V_act: float = np.nan  # actual value at optimum
    V_obj: float = np.nan  # value in objective
    Ω_act: float = np.nan  # actual profit (L_r - V_act - Σq*t)
    Ω_obj: float = np.nan  # actual profit (L_r - V_obj - Σq*t)
    q: np.ndarray = field(init=False)  # allocation/selection rule (N)
    t: np.ndarray = field(init=False)  # payment (N)

    def __post_init__(self):
        self.N = np.size(self.θ)
        C = [list(combinations(range(self.N), i+1)) for i in range(self.N)]
        self.C = np.array([item for sublist in C for item in sublist], dtype=object) 
        self.C_n = np.array([len(c) for c in self.C]) 
        self.N_c = np.size(self.C) 
        self.V_i = self.V_c[:self.N] 
        self.q = np.zeros(self.N)
        self.t = np.zeros(self.N)

    def get_actual_value(self, U, q):
        q_act = self.get_idx(q)
        U_act = U[list(self.C).index(tuple(q_act))] if tuple(q_act) != () else self.L_r
        return U_act, q_act
    
    @staticmethod
    def get_idx(q):
        return np.arange(len(q))[q.astype(bool)]
    
    def shapley(self, q): 
        if not np.any(q):
            return q
        else:
            ϕ = np.zeros(self.N)
            N_m = np.sum(q, dtype=int)
            M = [list(combinations(self.get_idx(q), i+1)) for i in range(N_m)]
            M = np.array([item for sublist in M for item in sublist], dtype=object)
            V_m = self.V_c[[list(self.C).index(i) for i in M]]
            for i in range(self.N):
                idx = np.array([i in c for c in M])
                for c in M[idx]:
                    c_c = tuple(n for n in c if n != i)
                    if c_c == ():
                        V_c_c = 0
                    else:
                        V_c_c = V_m[list(M).index(c_c)]
                    N_t = len(c)
                    weight = factorial(N_t - 1)*factorial(N_m-N_t)/factorial(N_m)
                    ϕ[i] += weight*(V_m[list(M).index(c)] - V_c_c)
            return ϕ


@dataclass
class BayesianMarket(DataMarket):
    θ_params: tuple = (0, 1)  # reserve price location and scale parameters (2)
    Z: st.rv_continuous = st.uniform  # reserve price distribution (scipy.stats)
    P: np.ndarray = field(init=False)
    P_c: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.Θ = self.Z(loc=self.θ_params[0], scale=self.θ_params[1])
        self.ψ = self.θ + np.nan_to_num(
            np.divide(
                self.Θ.cdf(self.θ),
                self.Θ.pdf(self.θ),
                out=np.zeros_like(self.θ),
                where=self.Θ.pdf(self.θ) != 0),
            nan=0.0)
        self.P = np.zeros_like(self.ψ)
        self.P_c = np.zeros_like(self.V_c)


@dataclass
class Central(BayesianMarket):
    def run(self, bayesian=False, verbose=False):
        """
        Assumes access to oracle for combinatorial value function V_c
        Returns minimum Ω (value + payments)
        """
        if bayesian:
            self.P = self.ψ
        else: 
            self.P = self.θ

        self.P_c = np.array([self.P[c, ].sum() for c in self.C])  # Coalition virtual cost
        c_opt = np.nanargmin(self.V_c + self.P_c)  # Optimal coalition
        if ~np.isnan(c_opt):        
            self.q[self.C[c_opt], ] = 1  # Optimal allocation vector
            V_opt = self.V_c[c_opt]  # Optimal value function
        else:
            V_opt = self.L_r

        self.t = self.P*self.q  # Optimal payments
        self.B = self.L_r - V_opt  # optimal budget
        self.model_vars = {
            'q': self.q,
            't': self.t,
            'V_obj': V_opt,
            'V_act': self.get_actual_value(self.V_c, self.q)[0],
            'obj': V_opt,
            'c': c_opt,            
            'B': self.B            
            }

        if verbose:
            print(self.model_vars)
        return self.model_vars


@dataclass
class BudgetedMarket(DataMarket):
    B: float = 0.0  # fixed budget

    def budget_check(self):
        ΔB = self.B - sum(self.t) 
        return (ΔB >= 0), ΔB


@dataclass
class GreedyKnapsack(BudgetedMarket):
    """
    Greedy algorithm for budget constrained reverse auction
    Ren, K. (2022). Differentially Private Auction
     for Federated Learning with Non-IID Data.
     Proceedings of International Conference on Service Science,
      ICSS, 2022-May, 305–312. https://doi.org/10.1109/ICSS55994.2022.00054
    """
    k: int = 0  # index

    def run(self):
        r = 1 / self.V_i
        v = np.sort(self.θ/r)  # cost per unit value
        idx = np.argsort(self.θ/r)
        rev_idx = np.argsort(idx)
        for i in range(1, self.N+1):
            if self.B/r[idx][:i].sum() >= v[i-1]:
                self.k = i
        if self.k == 0:
            self.q = np.zeros(self.N)
            self.t = np.zeros(self.N)
        elif self.k == self.N:
            self.q = np.ones(self.N)
            self.t = (self.B/r.sum())*r
        else:
            self.q = np.zeros(self.N)
            self.q[:self.k] = 1
            self.q = self.q[rev_idx]
            self.t = min((self.B/(r*self.q).sum()), v[self.k])*r*self.q
        self.optimal = True
        self.model_vars = {
            'q': self.q,
            't': self.t,
            'V_obj': np.sum(r*self.q),
            'V_act': self.get_actual_value(self.V_c, self.q)[0],
            'obj': np.sum(r*self.q),
            'k': self.k,
            'B': self.B            
            }
        return self.model_vars   


@dataclass
class CentralBudget(BayesianMarket, BudgetedMarket):

    def run(self, bayesian=False, endogenous=False,
            joint=False, verbose=False):
        """
        Assumes access to oracle for combinatorial value function V_c
        As a result the budget constraint becomes Σq_i*ψ_i <= V_r - V_c for i in c
        """
        if bayesian:
            self.P = self.ψ
        else: 
            self.P = self.θ
        if endogenous:
            self.B = self.L_r - self.V_c

        self.P_c = np.array([self.P[c, ].sum() for c in self.C])  # Coalition virtual cost
        C_feasible = (self.P_c <= self.B)  # Feasible coalitions
        if joint:
            c_opt = np.nanargmin(np.where(
                C_feasible, self.V_c + self.P_c, np.nan)) \
                if C_feasible.any() else np.nan  # Optimal coalition
        else:
            c_opt = np.nanargmin(
                np.where(C_feasible, self.V_c, np.nan)) if C_feasible.any() else np.nan  # Optimal coalition
        if ~np.isnan(c_opt):        
            self.q[self.C[c_opt], ] = 1  # Optimal allocation vector
            V_opt = self.V_c[c_opt]  # Optimal value function
        else:
            V_opt = self.L_r

        self.t = self.P*self.q  # Optimal payments
        self.B_act = self.L_r - V_opt  # optimal budget
        self.model_vars = {
            'q': self.q,
            't': self.t,
            'V_obj': V_opt,
            'V_act': self.get_actual_value(self.V_c, self.q)[0],
            'obj': V_opt,
            'c': c_opt,            
            'B': self.B_act
            }

        if verbose:
            print(self.model_vars)
        return self.model_vars


@dataclass
class RandomBudget(BayesianMarket, BudgetedMarket):

    def run(self, bayesian=False, endogenous=False,
            joint=False, verbose=False):
        """
        Assumes access to oracle for combinatorial value function V_c
        As a result the budget constraint becomes Σq_i*ψ_i <= L_R - V_c for i in c
        """
        if bayesian:
            self.P = self.ψ
        else: 
            self.P = self.θ
        
        if endogenous:
            self.B = self.L_r - self.V_c
        
        self.P_c = np.array([self.P[c, ].sum() for c in self.C])  # Coalition virtual cost
        C_feasible = (self.P_c <= self.B)  # Feasible coalitions
        if C_feasible.sum() > 0:
            if joint:
                V_opt = (self.V_c + self.P_c)[C_feasible].mean()  # Average value of feasible coalitions
            else:
                V_opt = self.V_c[C_feasible].mean()
        else:
            V_opt = self.L_r
        # self.q[:] = np.zeros_like
        # self.t[:] = np.nan
        self.B = self.L_r - V_opt  # optimal budget
        self.model_vars = {
            'q': self.q,
            't': self.t,
            'V_obj': V_opt,
            'V_act': V_opt,
            'obj': V_opt,
            'B': self.B,
            'V': V_opt
            }

        if verbose:
            print(self.model_vars)
        return self.model_vars


@dataclass
class Random(DataMarket):

    def run(self, N, verbose=False):
        """
        Assumes access to oracle for combinatorial value function V_c
        Returns the average value of coalitions with N sellers
        """
        V_opt = self.V_c[self.C_n == N].mean()  # Average value of coalitions with N sellers
        # self.q[:] = np.nan
        # self.t[:] = np.nan
        self.model_vars = {
            'q': self.q,
            't': self.t,
            'V_obj': V_opt,
            'V_act': V_opt,
            'obj': V_opt
            }

        if verbose:
            print(self.model_vars)
        return self.model_vars


@dataclass
class SingleMindedQuery(BayesianMarket, BudgetedMarket):
    """
        Zhang, M., Beltran, F., & Liu, J. (2020).
        Selling Data at an Auction under Privacy Constraints.
        In J. Peters & D. Sontag (Eds.),
        Proceedings of the 36th Conference on Uncertainty
        in Artificial Intelligence (UAI) (Vol. 124, pp. 669–678).
        PMLR. https://proceedings.mlr.press/v124/zhang20b.html
    """
    inverse: bool = True  # invert value to maximise

    def __post_init__(self):
        super().__post_init__()
        if self.inverse:
            self.V_i = 1/self.V_i

    def run(self, verbose=False):
        if self.Z == st.uniform:
            θ_star = cp.Variable(self.N, name='theta_star')  # Optimal threshold price
            # pdf and cdf assuming uniformly distributed valuations
            # θ ~ U(0,1), F(θ) = (θ-θ_l)/(θ_u-θ_l), f(θ) = 1/(θ_u-θ_l)
            θ_l, θ_u = self.θ_params
            f = np.ones(self.N)*(1/(θ_u-θ_l))  # pdf
            F = (θ_star-θ_l)*(1/(θ_u-θ_l))  # cdf
            objective = cp.Maximize(cp.sum(cp.multiply(self.V_i, F)))  # Objective function

            constraints = [
                cp.sum((cp.power(θ_star, 2)-cp.multiply(θ_star, θ_l))*(1/(θ_u-θ_l))) <= self.B,
                θ_star >= θ_l,
                θ_star <= θ_u
            ]


            self.model = cp.Problem(objective, constraints)  # Create the problem
            self.model.solve(solver=cp.MOSEK, verbose=verbose)  # Solve the problem
            if self.model.status == 'optimal':
                self.q = np.array([1.0 if self.θ[i] <= θ_star.value[i] else 0.0 for i in range(self.N)])
                self.t = self.q*θ_star.value
                V_opt = np.sum(self.V_i*self.q)
            else:
                V_opt = self.L_r
            self.model_vars = {
                'model': self.model,
                'q': self.q,
                't': self.t,
                'V_obj': V_opt,
                'V_act': self.get_actual_value(self.V_c, self.q)[0],
                'obj': self.model.objective.value,
                'B': self.B 
                }  
        else:
            print('Only usable for uniform reserve price distribution')
            return
        return self.model_vars


@dataclass
class WassersteinMarket(DataMarket):
    K: float = 1  # Lipschitz constant
    δ: float = 0.95  # confidence level
    C_δ: float = field(init=False)  # Constant

    def hoeffding_bound(self):
        return np.array([self.calc_bound(c) for c in self.C])
    
    def calc_bound(self, c):
        return np.nan
          

@dataclass
class FixedInfiniteMarket(WassersteinMarket, BayesianMarket, BudgetedMarket):
    M: float = 100

    def __post_init__(self):
        super().__post_init__()
        self.C_δ = self.K*np.sqrt(np.log(2/(1-self.δ))/2)  # Risk dependent constant

    def run(self, budget=False, endogenous=False, verbose=False):
        """
        Data market clearing problem assuming fixed privacy parameters and infinite population
        W_r: float  # Reference Wasserstein value W(R,T)
        W_i: np.ndarray  # Individual Wasserstein vector W(X_i,T)
        """
        V_opt = np.nan
        V_act = np.nan
        if ~np.isnan(self.ψ).any():
            V = self.V_i
            q_var = cp.Variable(self.N, boolean=True, name='q')  # Allocation rule vector
            q_r = cp.Variable(boolean=True, name='q_r')  # Reference selector
            s_var = cp.Variable(name='s')  # Auxiliary variable for fractional objective
            z_var = cp.Variable(self.N, name='z')  # Auxiliary variable for binary (q) x continuous (f) linearisation
            
            # Objective function
            if budget:
                if endogenous:
                    objective = cp.Minimize(self.C_δ * s_var + q_r * self.L_r)
                else:
                    objective = cp.Minimize(s_var)
            else: 
                objective = cp.Minimize(self.C_δ*s_var + cp.sum(cp.multiply(self.ψ, q_var)) + q_r*self.L_r)

            # Budget constraint
            if budget:
                if endogenous:
                    constraints = [cp.sum(cp.multiply(self.ψ, q_var)) <= self.L_r - self.C_δ*s_var]
                else:
                    constraints = [cp.sum(cp.multiply(self.ψ, q_var)) <= self.B, cp.sum(q_var) >= 1]
            else: 
                constraints = []

            # Other constraints
            constraints += [
                cp.norm(cp.multiply(V, q_var)) <= cp.sum(z_var),
                s_var >= 0,
                cp.sum(q_var + q_r) >= 1
                ]
            
            for i in range(self.N):
                constraints += [
                    z_var[i] >= 0,
                    z_var[i] <= self.M*q_var[i],
                    s_var - self.M*(1-q_var[i]) <= z_var[i],
                    s_var >= z_var[i]
                ]

            self.model = cp.Problem(objective, constraints)  # Create the problem
            self.model.solve(solver=cp.GUROBI, verbose=verbose)  # Solve the problem

            # Calculate optimal values
            if self.model.status == 'optimal':
                self.q = np.abs(np.round(q_var.value))
                self.t = self.q*self.ψ
                if sum(self.q) > 0:
                    V_opt = self.C_δ*np.linalg.norm(self.V_i*self.q)/sum(self.q)
                    V_act = self.get_actual_value(self.V_c, self.q)[0]
                else:
                    V_opt = self.L_r
                    V_act = self.L_r
                if endogenous:
                    self.B = self.L_r - self.C_δ*s_var.value
            self.model_vars = {
                'model': self.model,
                'q': self.q,
                't': self.t,
                'V_obj': V_opt,
                'V_act': V_act,
                'obj': objective.value,
                'q_r': q_r.value,
                'z': z_var.value,
                's': s_var.value
                }
        else:
            self.model_vars = {
                'model': None,
                'q': np.zeros(self.N),
                't': np.zeros(self.N),
                'V_obj': np.nan,
                'V_act': np.nan,
                'obj': np.nan,
                'q_r': 0,
                'z': np.nan,
                's': np.nan,
                'r': np.nan
            }
        if verbose:
            print(self.model_vars)        
        return self.model_vars
    
    def calc_bound(self, c):
        return (self.C_δ/len(c))*np.sqrt(np.sum(self.W[c, ]**2))


@dataclass
class FixedFiniteMarket(WassersteinMarket, BayesianMarket, BudgetedMarket):
    M: float = 100
            
    def __post_init__(self):
        super().__post_init__()
        self.C_δ = self.K*np.sqrt(np.log(2/(1-self.δ))/(2*(self.N-1)))  # Risk dependent constant

    def run(self, budget=False, endogenous=False, verbose=False, **kwargs):
        """
        Data market clearing problem assuming fixed privacy parameters and infinite population
        W_r: float  # Reference Wasserstein value W(R,T)
        W_i: np.ndarray  # Individual Wasserstein vector W(X_i,T)
        """
        if ~np.isnan(self.ψ).any():
            V = self.V_i
            W_b = np.repeat(V, self.N)
            q_var = cp.Variable(self.N, boolean=True, name='q')  # Allocation rule vector
            q_r = cp.Variable(boolean=True, name='q_r')  # Reference selector
            r_var = cp.Variable((self.N, self.N), boolean=True, name='r')  # Aux var bin(q_i) x bin(q_j)
            z_var = cp.Variable(self.N, name='z')  # Auxiliary variable for continuous(sum(q)) x bin(q)
            s_var = cp.Variable(name='s')  # Auxiliary variables for fractional objective
            
            # Objective function
            if budget:
                if endogenous:
                    objective = cp.Minimize(self.C_δ * s_var + q_r * self.L_r)
                else:
                    objective = cp.Minimize(s_var)
            else: 
                objective = cp.Minimize(self.C_δ*s_var + q_r*self.L_r + cp.sum(cp.multiply(q_var, self.ψ)))

            # Budget constraint
            if budget:
                if endogenous:
                    constraints = [cp.sum(cp.multiply(self.ψ, q_var)) <= self.L_r - self.C_δ*s_var]
                else:
                    constraints = [cp.sum(cp.multiply(self.ψ, q_var)) <= self.B, cp.sum(q_var) >= 1]
            else: 
                constraints = []

            # Other constraints
            constraints += [  # Create the constraints
                cp.sum(q_var) <= self.N, cp.sum(q_var + q_r) >= 1, s_var >= 0,
                cp.norm(cp.multiply(W_b, r_var.T.flatten())) <= cp.sum(z_var)
                ]
            for i in range(self.N):
                constraints += [
                    z_var[i] >= 0,
                    z_var[i] <= self.M*q_var[i],
                    s_var - z_var[i] <= self.M*(1-q_var[i]),
                    s_var - z_var[i] >= 0
                ]
                for j in range(self.N):
                    constraints += [
                            r_var[i, j] <= q_var[i],
                            r_var[i, j] <= (1-q_var[j]),
                            r_var[i, j] >= q_var[i] - q_var[j]
                    ]
                    
            self.model = cp.Problem(objective, constraints)  # Create the problem
            self.model.solve(solver=cp.GUROBI, verbose=verbose, **kwargs)  # Solve the problem

            # Calculate optimal values
            
            if self.model.status == 'optimal':
                self.q = np.abs(np.round(q_var.value))
                self.t = self.q*self.ψ
                if sum(self.q) > 0:
                    V_opt = self.C_δ*np.sqrt(self.N-sum(self.q))*np.linalg.norm(self.V_i*self.q)/sum(self.q)
                    V_act = self.get_actual_value(self.V_c, self.q)[0]
                else:
                    V_opt = self.L_r
                    V_act = self.L_r
                if endogenous:
                    self.B = self.L_r - self.C_δ*s_var.value
            else:
                V_opt = np.nan
                V_act = np.nan
            self.model_vars = {
                'model': self.model,
                'q': self.q,
                't': self.t,
                'V_obj': V_opt,
                'V_act': V_act,
                'obj': objective.value,
                'q_r': q_r.value,
                'z': z_var.value,
                's': s_var.value,
                'r': r_var.value}
        else:
            self.model_vars = {
                'model': None,
                'q': np.zeros(self.N),
                't': np.zeros(self.N),
                'V_obj': np.nan,
                'V_act': np.nan,
                'obj': np.nan,
                'q_r': 0,
                'z': np.nan,
                's': np.nan,
                'r': np.nan
            }
        return self.model_vars  
    
    def calc_bound(self, c):
        return (self.C_δ/len(c))*np.sqrt((self.N - len(c))*np.sum((self.W[c, ]**2)))


@dataclass
class ModFixedFiniteMarket(FixedFiniteMarket):
    def __post_init__(self):
        from itertools import chain
        self.N = np.size(self.θ)
        self.C = chain.from_iterable([combinations(range(self.N), i + 1) for i in range(self.N)])
        self.N_c = 2 ** self.N - 1
        self.C_n = np.zeros(self.N)
        self.V_i = self.V_c[:self.N]
        self.q = np.zeros(self.N)
        self.t = np.zeros(self.N)
        self.Θ_Z = self.Z(loc=self.θ_params[0], scale=self.θ_params[1])
        self.ψ = self.θ + np.nan_to_num(
            np.divide(
                self.Θ_Z.cdf(self.θ),
                self.Θ_Z.pdf(self.θ),
                out=np.zeros_like(self.θ),
                where=self.Θ_Z.pdf(self.θ) != 0),
            nan=0.0)
        self.P = np.zeros_like(self.ψ)
        self.P_c = np.zeros_like(self.V_c)
        self.C_δ = self.K * np.sqrt(np.log(2 / (1 - self.δ)) / (2 * (self.N - 1)))  # Risk dependent constant
    def get_actual_value(self, U, q):
        return 0, np.zeros(self.N)

    def run(self, budget=False, endogenous=False, verbose=False, valid = False, **kwargs):
        """
        Data market clearing problem assuming fixed privacy parameters and infinite population
        W_r: float  # Reference Wasserstein value W(R,T)
        W_i: np.ndarray  # Individual Wasserstein vector W(X_i,T)
        """
        if ~np.isnan(self.ψ).any():
            V = self.V_i
            W_b = np.repeat(V, self.N)
            q_var = cp.Variable(self.N, boolean=True, name='q')  # Allocation rule vector
            q_r = cp.Variable(boolean=True, name='q_r')  # Reference selector
            r_var = cp.Variable((self.N, self.N), boolean=True, name='r')  # Aux var bin(q_i) x bin(q_j)
            z_var = cp.Variable(self.N, name='z')  # Auxiliary variable for continuous(sum(q)) x bin(q)
            s_var = cp.Variable(name='s')  # Auxiliary variables for fractional objective
            self.M = np.sqrt(np.sum(V))
            # Objective function
            if budget:
                if endogenous:
                    objective = cp.Minimize(s_var + q_r * self.L_r)
                else:
                    objective = cp.Minimize(s_var)
            else:
                objective = cp.Minimize(s_var + q_r * self.L_r + cp.sum(cp.multiply(q_var, self.ψ)))

            # Budget constraint
            if budget:
                if endogenous:
                    constraints = [cp.sum(cp.multiply(self.ψ, q_var)) <= self.L_r - self.C_δ * s_var]
                else:
                    constraints = [cp.sum(cp.multiply(self.ψ, q_var)) <= self.B, cp.sum(q_var) >= 1]
            else:
                constraints = []

            # Other constraints
            constraints += [  # Create the constraints
                cp.sum(q_var + q_r) <= self.N, cp.sum(q_var + q_r) >= 1, s_var >= 0,
                self.C_δ * cp.norm(cp.multiply(W_b, r_var.T.flatten())) <= cp.sum(z_var)
            ]
            for i in range(self.N):
                constraints += [
                    z_var[i] >= 0,
                    z_var[i] <= self.M * q_var[i],
                    s_var - z_var[i] <= self.M * (1 - q_var[i]),
                    s_var - z_var[i] >= 0
                ]
                for j in range(self.N):
                    constraints += [
                        r_var[i, j] <= q_var[i],
                        r_var[i, j] <= (1 - q_var[j]),
                        r_var[i, j] >= q_var[i] - q_var[j]
                    ]
                # Additional logic constraints
                constraints += [
                    cp.sum(r_var[:, i]) <= cp.sum(q_var),
                    cp.sum(r_var[:, i]) <= self.N * (1 - q_var[i]),
                    cp.sum(r_var[i, :]) <= self.N - cp.sum(q_var),
                    cp.sum(r_var[i, :]) <= self.N * q_var[i],
                    r_var[i, i] == 0,
                ]
            if valid:
                # Set Big-M at lowest W + cost individual value
                q_min = np.zeros(self.N)
                q_min[np.argmin(self.C_δ * np.sqrt(self.N - 1) * self.V_i + self.ψ)] = 1
                M_min = self.C_δ * np.sqrt(self.N - sum(q_min)) * np.linalg.norm(self.V_i * q_min) / sum(
                    q_min) + np.sum(self.ψ * q_min)
                self.M = min(M_min * 1.01, self.L_r)
                # Valid inequalities
                s_test = np.zeros((self.N, self.N))
                c_test = np.zeros((self.N, self.N))
                idx_s = np.argsort(self.V_i)
                idx_c = np.argsort(self.ψ)
                for i in range(self.N):
                    for j in range(self.N):
                        q_v = np.zeros(self.N)
                        idx = list(idx_s.copy())
                        idx.remove(i)
                        idx = np.array(list([i]) + idx)
                        q_v[idx[:j + 1]] = 1
                        s_test[i, j] = self.C_δ * np.sqrt(self.N - sum(q_v)) * np.linalg.norm(
                            self.V_i * q_v) / sum(q_v)
                        idx = list(idx_c.copy())
                        idx.remove(i)
                        idx = np.array(list([i]) + idx)
                        q_v[idx[:j + 1]] = 1
                        c_test[i, j] = np.sum(self.ψ * q_v)
                eta = s_test + c_test
                gamma = np.zeros((self.N, self.N))
                mu = np.zeros((self.N, self.N))
                gamma_opt = np.zeros(self.N)
                mu_opt = np.zeros(self.N)
                a = np.zeros(self.N)
                b = np.zeros(self.N)
                c = np.zeros(self.N)
                d = np.zeros(self.N)
                u_opt = np.zeros(self.N, dtype=int)
                v_opt = np.zeros(self.N, dtype=int)
                for i in range(self.N):
                    constraints += [
                        s_var + cp.sum(cp.multiply(q_var, self.ψ)) >= q_var[i] * np.min(eta[i, :])
                    ]
                    gamma[i] = (eta[i, :] - eta[i, -1]) / (np.arange(self.N) + 1 - self.N)
                    mu[i] = (eta[i, :] - eta[i, 0]) / (np.arange(self.N))
                    gamma_opt[i] = np.max(gamma[i][:-1])
                    mu_opt[i] = np.min(mu[i][1:])
                    u_opt[i] = np.argmax(gamma[i][:-1]) + 1
                    v_opt[i] = np.argmin(mu[i][1:]) + 1
                    a[i] = gamma_opt[i]
                    b[i] = eta[int(u_opt[i] - 1), i] - gamma_opt[i] * u_opt[i]
                    c[i] = mu_opt[i]
                    d[i] = eta[i, 0] - mu_opt[i]
                    if gamma_opt[i] > 0:
                        constraints += [
                            s_var + cp.sum(cp.multiply(q_var, self.ψ)) >= (self.N * a[i] - b[i]) * q_var[i] - a[
                                i] * cp.sum(r_var[i])
                        ]
                    if mu_opt[i] < 0:
                        constraints += [
                            s_var + cp.sum(cp.multiply(q_var, self.ψ)) >= (self.N * c[i] - d[i]) * q_var[i] - c[
                                i] * cp.sum(r_var[i])
                        ]
                s_min = np.zeros(self.N)
                cost = np.zeros(self.N)
                for i in range(self.N):
                    # Closest distance by coalition size
                    q_v = np.zeros(self.N)
                    q_v[np.argsort(self.V_i)[:i + 1]] = 1
                    s_min[i] = self.C_δ * np.sqrt(self.N - sum(q_v)) * np.linalg.norm(self.V_i * q_v) / sum(q_v)
                    # Cheapest by coalition size
                    q_p = np.zeros(self.N)
                    q_p[np.argsort(self.ψ)[:i + 1]] = 1
                    cost[i] = np.sum(self.ψ * q_p)
                x_v_min = np.argmax((s_min / (np.arange(self.N) + 1 - self.N))[:-1]) + 1
                m_v = np.max((s_min / (np.arange(self.N) + 1 - self.N))[:-1])
                c_v = s_min[x_v_min - 1] - m_v * x_v_min

                x_p_min = np.argmax((((s_min + cost) - cost[-1]) / (np.arange(self.N) + 1 - self.N))[:-1]) + 1
                m_p = np.max((((s_min + cost) - cost[-1]) / (np.arange(self.N) + 1 - self.N))[:-1])
                c_p = (s_min + cost)[x_p_min - 1] - m_p * x_p_min
                constraints += [
                    s_var >= m_v * cp.sum(q_var) + c_v * (1 - q_r),
                    s_var + cp.sum(cp.multiply(q_var, self.ψ)) >= np.min(cost + s_min) * (1 - q_r),
                    s_var + cp.sum(cp.multiply(q_var, self.ψ)) >= m_p * cp.sum(q_var) + c_p * (1 - q_r)
                ]
            self.model = cp.Problem(objective, constraints)  # Create the problem
            self.model.solve(solver=cp.GUROBI, verbose=verbose, **kwargs)  # Solve the problem

            # Calculate optimal values
            if self.model.status == 'optimal':
                self.q = np.abs(np.round(q_var.value))
                self.t = self.q * self.ψ
                if sum(self.q) > 0:
                    V_opt = self.C_δ * np.sqrt(self.N - sum(self.q)) * np.linalg.norm(self.V_i * self.q) / sum(self.q)
                    V_act = self.get_actual_value(self.V_c, self.q)[0]
                else:
                    V_opt = self.L_r
                    V_act = self.L_r
                if endogenous:
                    self.B = self.L_r - self.C_δ * s_var.value
            else:
                V_opt = np.nan
                V_act = np.nan
            self.model_vars = {
                'model': self.model,
                'q': self.q,
                't': self.t,
                'V_obj': V_opt,
                'V_act': V_act,
                'obj': objective.value,
                'q_r': q_r.value,
                'z': z_var.value,
                's': s_var.value,
                'r': r_var.value}
        else:
            self.model_vars = {
                'model': None,
                'q': np.zeros(self.N),
                't': np.zeros(self.N),
                'V_obj': np.nan,
                'V_act': np.nan,
                'obj': np.nan,
                'q_r': 0,
                'z': np.nan,
                's': np.nan,
                'r': np.nan
            }
        return self.model_vars

@dataclass
class DataRateMechanism(DataMarket):
    Z_σ: st.rv_continuous = st.norm  # assumed data distribution

    def run(self):
        c = np.sort(self.θ)
        V_σ = DataRateNewsvendor().get_data_rate(self.Z_σ)
        I = set(np.arange(self.N))  # set of data suppliers
        t_0 = np.max(np.where(c == c.min()))
        β_t = np.sum([β[j] for j in range(t_0)])

        # Supplier Selection
        active = (c[i] > (1 / (np.sqrt(2) * V_σ))
                  * np.sqrt(β_t + np.sqrt(β_t**2 + 4 * V_σ**2 * np.sum(σ**2))))
        I_0 = I[active]  # active set
        I_c = I - I_0
        # Uncertainty Reduction
        β_0 = np.sum([β[j] for j in I_0])
        V_σ_eq = np.array([V_σ * (np.sqrt(2) * β[i]) /
                           np.sqrt(β_0 + np.sqrt(β_0**2 + 4 * V_σ**2
                                                 * np.sum([σ[j]**2 for j in I_c])))
                           for i in I_0])
        Δσ = [σ[i] - β[i] / V_σ_eq[i] if i in I_0 else 0 for i in I]

        # Consumer Surplus Allocation
        CS_N = (V_σ * (σ_e - np.sqrt(β[I_0]**2 / V_σ_eq**2 + np.sum(σ[I_c]**2)))
                - β[I_0] * np.log(V_σ_eq*σ[I_0]/β[I_0]))  # Total consumer surplus

        CS = np.array([V_σ_eq[i] * σ[i] - β[i] - β[i] * np.log(V_σ_eq[i] * σ[i] / β[i])
                       if i in I_0 else 0 for i in I])  # Consumer surplus for each active
        CS_r = CS_N - np.sum(CS)  # Retailer consumer surplus allocation

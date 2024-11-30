from dataclasses import dataclass, field
import numpy as np
import math
import scipy.stats as st
from tqdm.notebook import tqdm
from models._valuators import Valuator
from models._markets import (FixedFiniteMarket, BayesianMarket,
                             WassersteinMarket, BudgetedMarket, CentralBudget)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Plotting Parameters
plt.style.use('config/publication.mplstyle')
cpal = plt.rcParams['axes.prop_cycle'].by_key()['color']
ls = plt.rcParams['axes.prop_cycle'].by_key()['linestyle']
mark = plt.rcParams['axes.prop_cycle'].by_key()['marker']
colors = [cpal[0], cpal[5], cpal[3]]
cmap = LinearSegmentedColormap.from_list('BB', colors, N=1000)
   
# Simulator Classes


@dataclass
class Simulator:
    valuator: Valuator  # valuator
    θ_bounds: np.ndarray  # min max bounds of reserve prices
    markets: list  # list of markets to test
    metrics: list  # list of metrics to test

    s_θ: np.ndarray  # max reserve price parameter sweep
    s_ρ: np.ndarray  # value-reserve price correlation parameter sweep
    s_δ: np.ndarray  # Hoeffding confidence level parameter sweep

    def __post_init__(self):
        self.N_d = self.valuator.N_d
        self.L = self.valuator.L
        self.ΔL = self.valuator.ΔL

    def run(self, L_r, seed=1345):
        pass

    def run_market(self, idx, market, arg, run_arg):
        market_instance = market(**arg)
        res_temp = market_instance.run(**run_arg)
        self.V_obj[idx] = res_temp['V_obj']
        self.V_act[idx] = res_temp['V_act']
        self.obj[idx] = res_temp['obj']
        self.q[idx] = res_temp['q']
        self.t[idx] = res_temp['t']
        self.t_Φ[idx] = market_instance.shapley(self.q[idx])

    def generate_reserve_prices(self, θ_l, θ_h,  ρ, L, seed):
        θ = self.generate_correlated_variable(L, ρ, seed)
        return θ*(θ_h-θ_l) + θ_l

    @staticmethod
    def get_actual_value(U, q, C, L_r):
        def get_idx(q):
            return np.arange(len(q))[q.astype(bool)]
        q_act = get_idx(q)
        U_act = U[list(C).index(tuple(q_act))] if tuple(q_act) != () else L_r
        return U_act, q_act


    @staticmethod
    def generate_correlated_variable(x1, ρ, seed):
        # Fernando Delgado Chaves (https://stats.stackexchange.com/users/374137/fernando-delgado-chaves),
        # Generate a random variable with a defined correlation to an existing variable(s),
        # URL (version: 2022-11-28): https://stats.stackexchange.com/q/597228
        np.random.seed(seed)
        N = np.size(x1)
        theta = math.acos(ρ)  # corresponding angle
        x2 = np.random.uniform(0, 1, N)
        X = np.vstack((x1, x2)).T
        Xctr = st.zscore(X)  # centered columns (mean 0)
        Id = np.diag(np.ones(N))  # identity matrix
        Q = np.linalg.qr(Xctr)[0][:, 0]  # QR-decomposition, just matrix Q
        P = Q.reshape(-1, 1) @ Q.reshape(1, -1)  # projection onto space defined by x1
        x2o = (Id - P) @ Xctr[:, 1]  # x2ctr made orthogonal to x1ctr
        Xc2 = np.vstack((Xctr[:, 0], x2o)).T  # bind to matrix
        Y = Xc2 @ np.diag(1 / np.sum(Xc2 ** 2, axis=0) ** 0.5)  # scale columns to length 1
        x = Y[:, 1] + (1 / math.tan(theta)) * Y[:, 0]  # final new vector
        x_range = x.max() - x.min()
        # x_scaled = x / x_range - min(0, (x / x_range).min())
        x_scaled = x/(x_range/0.8) - min(0, (x / (x_range/0.8)).min()-0.1)  # avoid 0
        return x_scaled


@dataclass
class FixedBudgetDistSimulator(Simulator):
    N_b: int = 10  # Number of budget intervals

    def __post_init__(self):
        super().__post_init__()
        self.N_r = self.valuator.N_r
        self.s_B = np.linspace(0.1, 1, self.N_b) * self.θ_bounds[1] * self.N_d
        self.V_obj = np.zeros((self.N_r, len(self.s_ρ), self.N_b,
                               len(self.metrics), len(self.markets)))
        self.V_act = np.zeros_like(self.V_obj)
        self.obj = np.zeros_like(self.V_obj)
        self.q = np.zeros((self.N_r, len(self.s_ρ), self.N_b,
                           len(self.metrics), len(self.markets), self.N_d))
        self.t = np.zeros_like(self.q)
        self.t_Φ = np.zeros_like(self.q)
        self.θ = np.zeros((self.N_r, len(self.s_ρ), self.N_d))

    def run(self, L_r, corr_metric_id=0, seed=1345):
        for i in tqdm(range(self.N_r)):
            for j, ρ in enumerate(self.s_ρ):
                self.θ[i, j] = self.generate_reserve_prices(
                    self.θ_bounds[0],
                    self.θ_bounds[1],
                    ρ, self.ΔL[i, corr_metric_id, :self.N_d], seed)  # reserve prices
                for k, B in enumerate(self.s_B):
                    for l, metric in enumerate(self.metrics):
                        for m, market in enumerate(self.markets):
                            arg = {'V_c': self.ΔL[i, l],
                                   'θ': np.squeeze(self.θ[i, j]),
                                   'L_r': L_r[i, l]}
                            run_arg = {}
                            if issubclass(market, BayesianMarket):
                                arg['θ_params'] = self.θ_bounds
                                arg['Z'] = st.uniform
                            if issubclass(market, WassersteinMarket):
                                run_arg['budget'] = True
                                if metric.K == np.inf:
                                    arg['K'] = 1  # assumed, only for RMSE
                                else:
                                    arg['K'] = metric.K
                                    arg['δ'] = 0.95
                            if issubclass(market, BudgetedMarket):
                                arg['B'] = B
                                
                            self.run_market((i, j, k, l, m), market, arg, run_arg)


@dataclass
class EndogenousBudgetDistSimulator(Simulator):
    L_r = None
    Ω_act = None
    L_act = None
    Ω_act_perc = None

    def __post_init__(self):
        super().__post_init__()
        self.N_r = self.valuator.N_r
        self.V_obj = np.zeros((self.N_r, len(self.s_θ), len(self.s_ρ),
                               len(self.metrics), len(self.markets), 3))
        self.V_act = np.zeros_like(self.V_obj)
        self.obj = np.zeros_like(self.V_obj)
        self.q = np.zeros((self.N_r, len(self.s_θ), len(self.s_ρ),
                           len(self.metrics), len(self.markets),
                           3, self.N_d))
        self.t = np.zeros_like(self.q)
        self.t_Φ = np.zeros_like(self.q)
        self.θ = np.zeros((self.N_r, len(self.s_θ), len(self.s_ρ), self.N_d))

    def run(self, L_r, δ, corr_metric_id=0, ref_metric=[0], seed=1345):
        self.L_r = L_r
        for i in tqdm(range(self.N_r)):
            for j, θ in enumerate(self.s_θ):
                for k, ρ in enumerate(self.s_ρ):
                    self.θ_bounds[1] = θ
                    self.θ[i, j, k] = self.generate_reserve_prices(
                        self.θ_bounds[0],
                        self.θ_bounds[1],
                        ρ, self.ΔL[i, corr_metric_id, :self.N_d], seed)  # reserve prices
                    for l, metric in enumerate(self.metrics):
                        B_X = L_r[i, l]
                        for m, market in enumerate(self.markets):
                            arg = {
                                'V_c': self.ΔL[i, l],
                                'θ': np.squeeze(self.θ[i, j, k]),
                                'L_r': B_X,
                            }
                            for n, mode in enumerate(['endogenous', 'joint', 'exogenous']):
                                run_arg = {}
                                if issubclass(market, BayesianMarket):
                                    arg['θ_params'] = self.θ_bounds
                                    arg['Z'] = st.uniform
                                if issubclass(market, WassersteinMarket):
                                    if mode == 'endogenous' or mode == 'exogenous':
                                        run_arg['budget'] = True
                                    else:
                                        run_arg['budget'] = False
                                    if metric.K == np.inf:
                                        arg['K'] = 1  # assumed, only for RMSE
                                    else:
                                        arg['K'] = metric.K
                                        arg['δ'] = δ
                                if issubclass(market, BudgetedMarket):
                                    if mode == 'endogenous':
                                        run_arg['endogenous'] = True
                                    if mode == 'exogenous':
                                        arg['B'] = B_X
                                if issubclass(market, BudgetedMarket) and not(issubclass(market, WassersteinMarket)):
                                    run_arg['bayesian'] = True
                                    if mode == 'joint':
                                        run_arg['joint'] = True
                                    else:
                                        run_arg['joint'] = False
                                    if mode == 'endogenous' or mode == 'joint':
                                        run_arg['endogenous'] = True

                                self.run_market((i, j, k, l, m, n), market, arg, run_arg)

    def calc_act_vals(self, val_metrics, dist_metrics):
        self.Ω_act = np.zeros(
            (len(val_metrics), len(dist_metrics),
             self.N_r, len(self.s_θ),
             len(self.s_ρ), len(self.markets), 3)
        )
        self.L_act = np.zeros_like(self.Ω_act)
        self.Ω_act_perc = np.zeros_like(self.Ω_act)
        for i, val in enumerate(val_metrics):
            U = self.ΔL[:, val, :]  # target metric
            for j, dist in enumerate(dist_metrics):
                for k in range(self.N_r):
                    for l in range(len(self.s_θ)):
                        for m in range(len(self.s_ρ)):
                            for n in range(len(self.markets)):
                                for o in range(3):
                                    self.L_act[i, j, k, l, m, n, o] = (
                                        self.get_actual_value(
                                            U[k],
                                            self.q[k, l, m, j, n, o],
                                            self.valuator.C, self.L_r[k, val])[0]
                                    )
                                    self.Ω_act[i, j, k, l, m, n, o] = \
                                        (self.L_act[i, j, k, l, m, n, o]
                                         + self.t[k, l, m, j, n, o].sum())

                                    self.Ω_act_perc[i, j, k, l, m, n, o] = (
                                            1 - (self.Ω_act[i, j, k, l, m, n, o]
                                                 / self.Ω_act[i, i, k, l, m, n, o]))

    def run_delta(self, s_δ, dist, ref):
        self.V_δ = np.zeros((self.N_r, len(self.s_θ), len(self.s_ρ),
                             len(s_δ)))
        self.q_δ = np.zeros((self.N_r, len(self.s_θ), len(self.s_ρ),
                             len(s_δ), self.N_d))
        self.t_δ = np.zeros_like(self.q_δ)
        for i in tqdm(range(self.N_r)):
            for j, θ in tqdm(enumerate(self.s_θ)):
                self.θ_bounds[1] = θ
                for k, ρ in enumerate(self.s_ρ):
                    for l, δ in enumerate(s_δ):
                        arg = {'V_c': self.ΔL[i, dist],
                               'θ': np.squeeze(self.θ[i, j, k]),
                               'L_r': self.L_r[i, ref],
                               'θ_params': self.θ_bounds,
                               'Z': st.uniform,
                               'K': 1,
                               'δ': δ
                               }
                        run_arg = {
                            'budget': False,
                            'endogenous': False,
                            'verbose': False}
                        idx = (i, j, k, l)
                        market_instance = FixedFiniteMarket(**arg)
                        res_temp = market_instance.run(**run_arg)
                        self.V_δ[idx] = res_temp['V_obj']
                        self.q_δ[idx] = res_temp['q']
                        self.t_δ[idx] = res_temp['t']
        self.Ω_act_δ = np.zeros((self.N_r, len(self.s_θ),
                                 len(self.s_ρ), len(s_δ)))
        self.Ω_act_perc_δ = np.zeros_like(self.Ω_act_δ)
        self.L_act_δ = np.zeros_like(self.Ω_act_δ)
        for i in tqdm(range(self.N_r)):
            for j, θ in tqdm(enumerate(self.s_θ)):
                self.θ_bounds[1] = θ
                for k, ρ in enumerate(self.s_ρ):
                    for l, δ in enumerate(s_δ):
                        self.L_act_δ[i, j, k, l] = (
                            self.get_actual_value(
                                self.ΔL[i, ref],
                                self.q_δ[i, j, k, l],
                                self.valuator.C,
                                self.L_r[i, ref])[0]
                        )
                        self.Ω_act_δ[i, j, k, l] = \
                            (self.L_act_δ[i, j, k, l]
                             + self.t_δ[i, j, k, l].sum())
                        self.Ω_act_perc_δ[i, j, k, l] = (
                                1 - self.Ω_act_δ[i, j, k, l]
                                / self.L_r[i, ref])

    def get_actual_value(self, U, q, C, L_r):
        q_act = self.get_idx(q)
        U_act = U[list(C).index(tuple(q_act))] if tuple(q_act) != () else L_r
        return U_act, q_act

    @staticmethod
    def get_idx(q):
        return np.arange(len(q))[q.astype(bool)]

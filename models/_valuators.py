from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.stats as st
from itertools import combinations
from math import factorial
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

from models._loss_functions import HoeffdingBound

# Plotting Parameters
plt.style.use('config/publication.mplstyle')
cpal = plt.rcParams['axes.prop_cycle'].by_key()['color']
ls = plt.rcParams['axes.prop_cycle'].by_key()['linestyle']
mark = plt.rcParams['axes.prop_cycle'].by_key()['marker']
colors = [cpal[0], cpal[5], cpal[3]]
cmap = LinearSegmentedColormap.from_list('BB', colors, N=1000)


# Valuator Classes
@dataclass
class Valuator:
    N_d: int  # number of data owners
    metrics: list  # list of performance metrics to test
    agg: str  # Aggregation scheme 'euclidean' or 'barycenter'

    C: np.ndarray = field(init=False)  # tuples of coalitions (N_c)
    C_n: np.ndarray = field(init=False)  # size of each coalition (N_c)
    N_c: int = field(init=False)  # number of coalitions

    L: np.ndarray = field(init=False)  # Loss function values (N_r, N_m, N_c)
    ΔL: np.ndarray = field(init=False)  # Loss function differential L[i] - L[-1] (N_r, N_m, N_c)
    ΔL_hat: np.ndarray = field(init=False)  # Loss function differential L[i] - L[-1] (N_m, N_c)
    Φ: np.ndarray = field(init=False)  # Shapley value (N_r, N_m, N_d)
    ρ: np.ndarray = field(init=False)  # Loss function cross-correlations
    ρ_hat: np.ndarray = field(init=False)  # Average loss function cross-correlations

    def __post_init__(self):
        C = [list(combinations(range(self.N_d), i+1)) for i in range(self.N_d)]
        self.C = np.array([item for sublist in C for item in sublist], dtype=object) 
        self.C_n = np.array([len(c) for c in self.C]) 
        self.N_c = np.size(self.C)
        self.K = np.array([metric.K for metric in self.metrics])
        self.N_m = len(self.metrics)

    def run(self):
        pass

    def data_aggregator(self, X):
        if self.agg == 'euclidean':
            return X.mean(axis=0)
        elif self.agg == 'barycenter':
            return np.nan
        else:
            print('Aggregation scheme not recognised. Select either euclidean or barycenter')
            return np.nan

    def shapley(self, V, q):
        if not np.any(q):
            return q
        else:
            ϕ = np.zeros(self.N_d)
            N_m = np.sum(q, dtype=int)
            M = [list(combinations(self.get_idx(q), i+1)) for i in range(N_m)]
            M = np.array([item for sublist in M for item in sublist], dtype=object)
            V_m = V[[list(self.C).index(i) for i in M]]
            for i in range(self.N_d):
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

    @staticmethod
    def get_idx(q):
        return np.arange(len(q))[q.astype(bool)]
    
    @staticmethod
    def calculate_correlation(L):
        return pd.DataFrame(L.T).corr().values

    def plot_correlations(self, ρ_hat, m_idx, labels, path, rotation=90, place='top', figsize=[3.5, 3]):
        fig, axes = plt.subplots(figsize=figsize)
        cax = axes.matshow(ρ_hat[:, m_idx][m_idx], cmap=cmap)
        axes.set_xticks(np.arange(len(m_idx)))
        axes.set_yticks(np.arange(len(m_idx)))
        axes.set_xticklabels(np.array(labels)[m_idx], rotation=rotation)
        axes.set_yticklabels(np.array(labels)[m_idx])
        fig.colorbar(cax)
        if place == 'bottom':
            fig.gca().xaxis.tick_bottom()
        fig.savefig(path + 'corr_' + self.gen_path() + '.pdf')
        return fig, axes

    def plot_correlation_performance(self, ρ_hat, i, m_idx, labels, path):
        # Performance comparison
        ρ_W = np.zeros((len(m_idx), len(m_idx)))
        for k, j in enumerate(m_idx):
            ρ_W[k] = ρ_hat[i, m_idx] >= ρ_hat[j, m_idx]
        fig, axes = plt.subplots()
        color_map = LinearSegmentedColormap.from_list('BB', [cpal[0], cpal[3]], N=1000)
        cax = axes.matshow(ρ_W[1:].T, cmap=color_map)
        axes.set_yticks(np.arange(len(m_idx)))
        axes.set_xticks(np.arange(len(m_idx)-1))
        axes.set_yticklabels(np.array(labels)[m_idx])
        axes.set_xticklabels(np.array(labels)[list(c for c in m_idx if c != i)], rotation=90)
        axes.set_xlabel(r'Source Metric (S)')
        axes.set_ylabel(r'Target Metric (T)')
        axes.xaxis.set_label_position('top')
        bounds = [-1, 1, 2]
        cbar = fig.colorbar(cax, cmap=color_map,
                             norm=BoundaryNorm(bounds, color_map.N),
                             boundaries=bounds)
        cbar.set_ticks([-0.2, 1.7])
        cbar.set_ticklabels(['No', 'Yes'])
        cbar.set_label(r'$\rho(S,T) \leq \rho({},T)$'.format(labels[i]),
                       rotation=270)
        fig.savefig(path + 'corr_perf_' + self.gen_path() + '.pdf')
        return fig, axes

    def gen_path(self):
        return 'N_d_{}'.format(self.N_d)

    def plot_shapley(self, r, m_idx, labels, path):
        """
        Bar plot of Shapley values for each seller for different metrics
        r: run index
        """
        fig, axes = plt.subplots()
        for i in m_idx:
            axes.bar((np.arange(self.N_d) + (1/(self.N_d+2))*i
                      - (1/(self.N_d+2))*0.5*self.N_d), self.Φ[r, i],
                     width=(1/(self.N_d+2)), label=labels[i],
                     align='center')
        axes.set_xticks(np.arange(self.N_d))
        fig.legend(ncol=2)
        fig.savefig(path + 'shapley_' + self.gen_path() + '.pdf')
        return fig, axes

    def plot_average_metrics(self, ΔL, K, m_idx, labels, path, xlabel="$n_P$"):
        """
        Line plot of average metric value for size of coalition.
        """
        fig, axes = plt.subplots()
        for i in m_idx:
            E_L = np.array([np.mean(ΔL[:, i][np.tile(self.C_n == j+1, (self.N_r, 1))])
                            for j in range(self.N_d)])
            axes.plot(np.arange(1, self.N_d+1), E_L/K[i],
                      label=labels[i])
        axes.set_xlabel(xlabel)
        axes.legend()
        fig.savefig(path + 'avg_' + self.gen_path() + '.pdf')
        return fig, axes

    def plot_scatter_metrics(self, ΔL_hat, K, x, m_idx, labels, path):
        """
        Scatter plot of metric values.
        """
        fig, axes = plt.subplots()
        for k, i in enumerate(k for k in m_idx if k != x):
            axes.scatter(ΔL_hat[x], ΔL_hat[i]/K[i],
                         marker=mark[k], label=labels[i],
                         s=2, alpha=0.5, c=cpal[k])
        max_x = np.max(ΔL_hat[x])
        axes.axline((0, 0), (max_x, max_x),  c='k',
                    ls='--', zorder=0)
        axes.set_xlabel(labels[x])
        axes.legend()
        fig.savefig(path + 'scatter_' + self.gen_path() + '.pdf')
        return fig, axes


@dataclass
class DistValuator(Valuator):
    N_s: int  # sample size of data
    N_r: int = 1  # number of simulation runs
    Z: object = st.norm  # standard data distribution (loc-scale)
    α_bounds: np.ndarray = np.array([4, 6])  # Location parameter uniformly distributed [High, Low]
    β_bounds: np.ndarray = np.array([1, 2])  # Scale parameter uniformly distributed [High, Low]
    seed: int = 1345  # random seed

    def __post_init__(self):
        super().__post_init__()
        np.random.seed(self.seed)
        self.α = np.random.uniform(low=self.α_bounds[0],
                                   high=self.α_bounds[1],
                                   size=(self.N_r, self.N_d))
        self.β = np.random.uniform(low=self.β_bounds[0],
                                   high=self.β_bounds[1],
                                   size=(self.N_r, self.N_d))
        self.L = np.zeros((self.N_r, self.N_m, self.N_c))
        self.ΔL = np.zeros_like(self.L)
        self.Φ = np.zeros((self.N_r, self.N_m, self.N_d))
        self.ρ = np.zeros((self.N_r, self.ΔL.shape[1], self.ΔL.shape[1]))
        
    def run(self, path):
        for i in tqdm(range(self.N_r)):
            if self.Z == st.norm or self.agg == 'barycenter':
                α_T, β_T = self.dist_aggregator(self.α[i], self.β[i])  # target distribution
            else:
                X = np.zeros((self.N_d, self.N_s))
                for x, (a, b) in enumerate(zip(self.α[i], self.β[i])):
                    X[x] = self.Z(loc=a, scale=b).rvs(size=self.N_s)
                X_T = self.data_aggregator(X)
            for j, metric in enumerate(self.metrics):
                for k, c in enumerate(self.C):
                    if self.Z == st.norm or self.agg == 'barycenter':
                        α_C, β_C = self.dist_aggregator(self.α[i, c], self.β[i, c])  # coalition distribution
                        self.L[i, j, k] = metric.closed_form(α_C, β_C, α_T, β_T)
                    else:
                        for x, (a, b) in enumerate(zip(self.α[i], self.β[i])):
                            X[x] = self.Z(loc=a, scale=b).rvs(size=self.N_s)
                        X_C = self.data_aggregator(X[c, ])
                        self.L[i, j, k] = metric.empirical(X_C, X_T)
                self.ΔL[i, j] = self.L[i, j, :] - self.L[i, j, -1]
                # self.Φ[i, j] = self.shapley(self.ΔL[i, j], np.ones(self.N_d))
            self.ρ[i] = self.calculate_correlation(self.ΔL[i])
        self.ρ_hat = self.ρ.mean(axis=0)
        self.ΔL_hat = self.ΔL.mean(axis=0)
        self.save_results(path)

    def dist_aggregator(self, α, β):
        if self.agg == 'euclidean':
            if self.Z == st.norm:
                return α.mean(), np.sqrt(np.sum(β**2))/np.size(β)
            else: 
                # print('Closed-form not available for chosen distribution. Returning empirical instead.')
                X = np.zeros((np.size(α), self.N_s))
                for i, (a, b) in enumerate(zip(α, β)):
                    X[i] = self.Z(loc=a, scale=b).rvs(size=self.N_s)
                    X_C = self.data_aggregator(X[i])
                return self.Z.fit_loc_scale(X_C)
        elif self.agg == 'barycenter':
            return α.mean(), β.mean()
        else:
            print('Aggregation scheme not recognised. Select either euclidean or barycenter')
            return np.nan, np.nan

    @staticmethod
    def generate_data(Z, α, β, N_d, N_s):
        return Z(loc=α, scale=β).rvs(size=(N_s, N_d))
    
    def save_results(self, path):
        with open(path + 'dist_valuation_' + self.gen_path() + '.pkl', 'wb') as f:
            pkl.dump(self, f)

    def gen_path(self):
        return 'st_{}_N_d_{}_N_r_{}'.format(self.Z.name, self.N_d, self.N_r)    

#
# @dataclass
# class ModDistValuator(DistValuator):
#
#     def __post_init__(self):
#         from itertools import chain
#         self.C = chain.from_iterable([combinations(range(self.N_d), i + 1) for i in range(self.N_d)])
#         self.N_c = 2 ** self.N_d - 1
#         self.C_n = np.zeros(self.N_c)
#         self.K = np.array([metric.K for metric in self.metrics])
#         self.N_m = len(self.metrics)
#         np.random.seed(self.seed)
#         self.α = np.random.uniform(low=self.α_bounds[0],
#                                    high=self.α_bounds[1],
#                                    size=(self.N_r, self.N_d))
#         self.β = np.random.uniform(low=self.β_bounds[0],
#                                    high=self.β_bounds[1],
#                                    size=(self.N_r, self.N_d))
#         self.L = np.zeros((self.N_r, self.N_m, self.N_c))
#         self.ΔL = np.zeros_like(self.L)
#         self.Φ = np.zeros((self.N_r, self.N_m, self.N_d))
#         self.ρ = np.zeros((self.N_r, self.ΔL.shape[1], self.ΔL.shape[1]))

@dataclass
class DataValuator(Valuator):
    X: np.ndarray = np.zeros(1)  # input data (N_d, N_s)
    data_file_name: str = 'sm_cluster'

    def __post_init__(self):
        super().__post_init__()
        self.L = np.zeros((self.N_m, self.N_c))
        self.ΔL = np.zeros_like(self.L)
        self.Φ = np.zeros((self.N_m, self.N_d))

    def run(self):
        X_T = self.data_aggregator(self.X)  # target distribution
        for i, metric in enumerate(self.metrics):
            for j, c in enumerate(self.C):
                X_C = self.data_aggregator(self.X[c, ])
                self.L[i, j] = metric.empirical(X_C, X_T)
            self.ΔL[i] = self.L[i, :] - self.L[i, -1]
            self.Φ[i] = self.shapley(self.ΔL[i], np.ones(self.N_d))
        self.ρ = self.calculate_correlation(self.ΔL)
        self.ρ_hat = self.ρ
        self.ΔL_hat = self.ΔL

    def save_results(self, path, file):
        with open(path + file + self.gen_path() + '.pkl', 'wb') as f:
            pkl.dump(self, f)

    def gen_path(self):
        return '{}'.format(self.data_file_name)

    def plot_average_metrics(self, m_idx, labels, path, xlabel="$n_c$"):
        """
        Line plot of average metric value for size of coalition.
        """
        fig, axes = plt.subplots()
        for i in m_idx:
            E_L = np.array([np.mean(self.ΔL[i][self.C_n == j+1])
                            for j in range(self.N_d)])
            axes.plot(np.arange(1, self.N_d+1), E_L/self.K[i],
                      label=labels[i])
        axes.set_xlabel(xlabel)
        axes.legend()
        fig.savefig(path + 'avg_' + self.gen_path() + '.pdf')
        return fig, axes

    def plot_shapley(self, labels, path):
        """
        Barplot of Shapley values for each seller for different metrics
        r: run index
        """
        fig, axes = plt.subplots()
        for i in range(self.N_m):
            axes.bar((np.arange(self.N_d) + (1/(self.N_d+2))*i
                      - (1/(self.N_d+2))*0.5*self.N_d), self.Φ[i],
                     width=(1/(self.N_d+2)), label=labels[i],
                     align='center')
        axes.set_xticks(np.arange(self.N_d))
        fig.legend(ncol=2)
        fig.savefig(path + 'shapley_' + self.gen_path() + '.pdf')
        return fig, axes
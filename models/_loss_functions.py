from dataclasses import dataclass
import numpy as np
import scipy.stats as st
from scipy import integrate
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_pinball_loss
from ot import emd2_1d as ws 


@dataclass
class PDF:
    bin_width: float = 1  # width of bins for empirical pdf
    est_mode: str = 'hist'  # estimation method for pdf ['hist', 'kde']

    def generate_pdf(self, X_1, X_2):
        bins = np.arange(
            min(X_1.min(), X_2.min()),
            max(X_1.max(), X_2.max()),
            self.bin_width)
        if self.est_mode == "kde":
            f_1 = st.gaussian_kde(X_1).evaluate(bins)
            f_2 = st.gaussian_kde(X_2).evaluate(bins)
        elif self.est_mode == 'hist':
            f_1 = np.histogram(X_1, bins=bins, density=True)[0]
            f_2 = np.histogram(X_1, bins=bins, density=True)[0]
        else:
            f_1 = None
            f_2 = None
        return f_1, f_2, bins


@dataclass
class PerformanceMetric:
    Z: st.rv_continuous = st.norm  # data distribution
    N_s: int = 10000  # number of samples for empirical calculation
    N_cf: int = 10  # number of runs for non-closed form
    K: float = 1  # lipschitz constant

    def closed_form(self, α_1, β_1, α_2, β_2):
        return np.nan

    def non_closed_form(self, α_1, β_1, α_2, β_2):
        return np.mean([self.empirical(
            self.Z(loc=α_1, scale=β_1).rvs(size=self.N_s),
            self.Z(loc=α_2, scale=β_2).rvs(size=self.N_s))
            for _ in range(self.N_cf)])

    def empirical(self, X_1, X_2):
        return np.nan
    
    @staticmethod
    def non_zero_divide(x, y, out=np.inf):
        x = np.array(x)
        y = np.array(y)
        with np.errstate(divide='ignore'):
            return np.divide(
                        x, y,
                        out=np.array(np.ones_like(x) * out),
                        where=(y != 0))


@dataclass
class PDFPerformanceMetric(PerformanceMetric, PDF):
    N_bins: int = 1000  # number of bin for pdf

    def non_closed_form(self, α_1, β_1, α_2, β_2):
        a = min(self.Z(loc=α_1, scale=β_1).ppf(0.0001),
                self.Z(loc=α_2, scale=β_2).ppf(0.0001))
        b = max(self.Z(loc=α_1, scale=β_1).ppf(0.9999),
                self.Z(loc=α_2, scale=β_2).ppf(0.9999))
        x = np.linspace(a, b, self.N_bins)
        f_1 = self.Z(loc=α_1, scale=β_1).pdf(x)
        f_2 = self.Z(loc=α_2, scale=β_2).pdf(x)
        return self.calculate(f_1, f_2, x)
    
    def calculate(self, f_1, f_2, x):
        pass


@dataclass
class MeanAbsoluteError(PerformanceMetric):

    def closed_form(self, α_1, β_1, α_2, β_2):
        if self.Z == st.norm:
            α = α_1 - α_2
            β = np.sqrt(β_1**2 + β_2**2)
            if β:
                return (β * np.sqrt(2 / np.pi) * np.exp(-α**2 / (2 * β**2))
                        + α * (1 - 2 * st.norm.cdf(-α / β)))
            else:
                return np.abs(α)
        else: 
            return self.non_closed_form(α_1, β_1, α_2, β_2)
    
    def empirical(self, X_1, X_2):
        return mean_absolute_error(X_2, X_1)


@dataclass
class RootMeanSquaredError(PerformanceMetric):
    a_min: float = 0  # minimum range
    b_max: float = 0  # maximum range

    def __post_init__(self):
        self.K = np.sqrt(np.max([
            np.abs(self.a_min),
            np.abs(self.b_max), 
            np.abs(self.Z.a),
            np.abs(self.Z.b)]))
        self.K = 1
        if self.K == np.inf:
            self.K = 1

    def closed_form(self, α_1, β_1, α_2, β_2):
        if self.Z == st.norm:
            μ = self.Z.mean(loc=α_1, scale=β_1) - self.Z.mean(loc=α_2, scale=β_2)
            var = self.Z.var(loc=α_1, scale=β_1) + self.Z.var(loc=α_2, scale=β_2)
            return np.sqrt(μ**2 + var)
        else: 
            return self.non_closed_form(α_1, β_1, α_2, β_2)
    
    def empirical(self, X_1, X_2):
        return np.sqrt(mean_squared_error(X_2, X_1))


@dataclass
class WassersteinMetric(PerformanceMetric):

    def closed_form(self, α_1, β_1, α_2, β_2):
        # computes theoretical wasserstein distance
        # equivalent to folded normal with μ = |μ_a-μ_b| and σ = |σ_a-σ_b|
        if self.Z == st.norm:
            α = np.abs(α_1 - α_2)
            β = np.abs(β_1 - β_2)
            return (β * np.sqrt(2 / np.pi) * np.exp(-self.non_zero_divide(α**2, 2*β**2))
                    + α * (1 - 2 * st.norm.cdf(-self.non_zero_divide(α, β))))
        elif self.Z == st.uniform:
            a = min((α_1 - α_2), (β_1 - β_2) + (α_1 - α_2))
            b = max((α_1 - α_2), (β_1 - β_2) + (α_1 - α_2))
            r = b-a
            if (a >= 0 and b >= 0) or (a <= 0 and b <= 0):
                return 0.5 * (np.abs(a) + np.abs(b))
            else:
                return 0.5 * (1 / r) * (a**2 + b**2)
        elif self.Z == st.logistic:
            α = np.abs(α_1 - α_2)
            β = np.abs(β_1 - β_2)
            return α + 2 * β * np.log(1 + np.exp(-self.non_zero_divide(α, β)))
        elif self.Z == st.rayleigh:
            α = α_1 - α_2
            β = β_1 - β_2
            if (α >= 0) and (β >= 0):
                return α + β * np.sqrt(np.pi / 2)
            else:
                return np.nan
        else: 
            return self.non_closed_form(α_1, β_1, α_2, β_2)
    
    def empirical(self, X_1, X_2):
        return ws(X_1, X_2, metric='euclidean')


@dataclass
class KolmogorovSmirnovMetric(PerformanceMetric):
    def closed_form(self, α_1, β_1, α_2, β_2):
        return self.non_closed_form(α_1, β_1, α_2, β_2)
        
    def empirical(self, X_1, X_2):
        return st.ks_2samp(X_1, X_2).statistic


@dataclass
class KullbackLieblerDivergence(PDFPerformanceMetric):
    
    def closed_form(self, α_1, β_1, α_2, β_2):
        if self.Z == st.norm:
            return np.log(β_2 / β_1) + ((β_1**2 + (α_1 - α_2)**2)/(2 * β_2**2) - 0.5)
        else:
            return self.non_closed_form(α_1, β_1, α_2, β_2)
        
    def empirical(self, X_1, X_2):
        f_1, f_2, x = self.generate_pdf(X_1, X_2)
        return self.calculate(f_1, f_2, x)

    def calculate(self, f_1, f_2, x):
        return st.entropy(f_1, f_2)


@dataclass
class JensenShannonDistance(PDFPerformanceMetric):
    def closed_form(self, α_1, β_1, α_2, β_2):
        return self.non_closed_form(α_1, β_1, α_2, β_2) 
        
    def empirical(self, X_1, X_2):
        f_1, f_2, x = self.generate_pdf(X_1, X_2)
        return self.calculate(f_1, f_2, x)

    def calculate(self, f_1, f_2, x):
        return jensenshannon(f_1, f_2)


@dataclass
class TotalVariationalDistance(PDFPerformanceMetric):
    def closed_form(self, α_1, β_1, α_2, β_2):
        return self.non_closed_form(α_1, β_1, α_2, β_2)
        
    def empirical(self, X_1, X_2):
        f_1, f_2, x = self.generate_pdf(X_1, X_2)
        return self.calculate(f_1, f_2, x)

    def calculate(self, f_1, f_2, x):
        return 0.5*sum(np.abs(f_1-f_2)*np.diff(x).mean())


@dataclass
class MeanPinballLoss(PerformanceMetric):
    τ: float = 0.95  # quantile

    def __post_init__(self):
        self.K = max(self.τ, 1-self.τ)

    def closed_form(self, α_1, β_1, α_2, β_2):
        if self.Z == st.norm:
            μ = α_2 - α_1 - β_1 * self.Z.ppf(self.τ)
            σ = β_2
            return (self.τ*self.censored_normal(μ, σ, 0)
                    + (1-self.τ)*self.censored_normal(-μ, σ, 0))
        else: 
            return self.non_closed_form(α_1, β_1, α_2, β_2)
    
    def empirical(self, X_1, X_2):
        q_2 = np.quantile(X_2, q=self.τ)
        return mean_pinball_loss(np.ones_like(X_1) * q_2, X_1, alpha=self.τ)

    @staticmethod
    def censored_normal(μ, σ, c=0):
        # returns the mean of censored normal max(X~(μ,σ^2),c)
        # https://stats.stackexchange.com/questions/360355/what-is-the-distribution-of-min0-x-when-x-follows-some-general-normal-distribu
        p = 1 - st.norm.cdf(x=c, loc=μ, scale=σ)
        return ((1 - p) * c + p *
                (μ + np.divide(
                    st.norm.pdf(-μ / σ) * σ,
                    1 - st.norm.cdf(-μ / σ),
                    out=np.zeros_like(μ),
                    where=(st.norm.cdf(-μ / σ) != 1))))


@dataclass
class Newsvendor(MeanPinballLoss):
    λ_r: float = 0.1  # Retail tariff
    λ_w: float = 0.06  # wholesale/day-ahead price
    λ_d: float = 0.03  # down regulation
    λ_u: float = 0.16  # up regulation
    obj: str = 'profit'  # objective to calculate profit or cost

    def __post_init__(self):
        self.c_o = self.λ_u - self.λ_w  # overage cost
        self.c_u = self.λ_w - self.λ_d  # underage cost
        self.τ = self.c_o / (self.c_o + self.c_u)  # critical fractile
        self.K = self.c_o + self.c_u  # Lipschitz constant

    def closed_form(self, α_1, β_1, α_2, β_2):
        if self.Z == st.norm:
            q = α_1 + β_1*self.Z.ppf(self.τ)  # order quantity at critical fractile
            if self.obj == 'profit':
                return -self.calculate_profit(α_2, β_2, q)
            else:
                return self.calculate_cost(α_2, β_2, q)
        else: 
            return self.non_closed_form(α_1, β_1, α_2, β_2)
    
    def empirical(self, X_1, X_2):
        q = np.quantile(X_1, self.τ)
        Π = (self.λ_r * X_2 - self.λ_w * q
             - self.λ_u * np.maximum(X_2 - q, 0)
             + self.λ_d * np.maximum(q - X_2, 0))
        return -Π.mean()

    def cost(self, D, q, σ_D):
        return (self.λ_w * q
                + self.λ_u * self.censored_normal(D - q, σ_D, 0)
                - self.λ_d * self.censored_normal(q - D, σ_D, 0))

    def profit(self, D, q, σ_D):
        return self.λ_r * D - self.cost(D, q, σ_D)

    def optimal_bid(self, α, β):
        return α + β * self.Z.ppf(self.τ)

    def calculate_profit(self, α, β, q=None):
        D = self.Z(loc=α, scale=β)
        if q is None:
            q = self.optimal_bid(α, β)
        return self.profit(D.mean(), q, D.std())

    def calculate_cost(self, α, β, q=None):
        D = self.Z(loc=α, scale=β)
        if q is None:
            q = self.optimal_bid(α, β)
        return self.cost(D.mean(), q, D.std())


@dataclass
class DRONewsvendor(Newsvendor):
    """
        Lee, S., Kim, H., Moon, I. (2021).
        A data-driven distributionally robust newsvendor model with a Wasserstein ambiguity set.
        Journal of the Operational Research Society, 72(8), 1879–1897.
        https://doi.org/10.1080/01605682.2020.1746203
        Theorem 3.2

        Mieth, R., Morales, J. M., Poor, H. V. (2023).
        Data Valuation from Data-Driven Optimization.
        https://arxiv.org/pdf/2305.01775.pdf
        Proposition 3
    """
    def closed_form(self, α_1, β_1, α_2, β_2):
        Π = super().closed_form(α_1, β_1, α_2, β_2)
        W = WassersteinMetric(self.Z).closed_form(α_1, β_1, α_2, β_2)
        return -self.c_o*W + Π

    def empirical(self, X_1, X_2):
        Π = super().empirical(X_1, X_2)
        W = WassersteinMetric(self.Z).empirical(X_1, X_2)
        return -self.c_o*W + Π


@dataclass
class DataRateNewsvendor(Newsvendor):
    """
        Wang, B., Guo, Q., Yang, T., Xu, L., Sun, H. (2021).
        Data valuation for decision-making with uncertainty in energy transactions:
        A case of the two-settlement market system.
        https://doi.org/10.1016/j.apenergy.2021.116643
        Table 1 pp.7
    """

    def closed_form(self, α_1, β_1, α_2, β_2):
        Δσ = (self.Z(loc=α_1, scale=β_1).std()
              - self.Z(loc=α_2, scale=β_2).std())
        Vσ = self.get_data_rate(self.Z)
        return Vσ * Δσ

    def empirical(self, X_1, X_2):
        Δσ = X_1.std() - X_2.std()
        Vσ = self.get_data_rate(self.Z)
        return Vσ * Δσ

    def get_data_rate(self, Z):
        if Z == st.norm:
            return ((np.sqrt(2 * np.pi) / 4)
                    * (self.λ_d - self.λ_u)
                    * (self.τ * np.log(1 / self.τ)
                       + (1 - self.τ) * np.log(1 / (1 - self.τ))))
        elif Z == st.uniform:
            return (np.sqrt(3)
                    * (self.λ_d - self.λ_u)
                    * self.τ * (1 - self.τ))
        elif Z == st.logistic:
            return ((np.sqrt(3) / np.pi)
                    * (self.λ_d - self.λ_u)
                    * (self.τ * np.log(1 / self.τ)
                       + (1 - self.τ) * np.log(1 / (1 - self.τ))))
        elif Z == st.laplace:
            return ((np.sqrt(2) / 4)
                    * (self.λ_d - self.λ_u)
                    * (1 - np.abs(1 - 2 * self.τ))
                    * np.log(np.e / (1 - np.abs(1 - 2 * self.τ))))
        elif Z == st.expon:
            return ((self.λ_d - self.λ_u)
                    * (1 - self.τ) * np.log(1 / (1 - self.τ)))
        elif Z == st.rayleigh:
            return (np.sqrt(4 * np.pi / (4 - np.pi)) * (self.λ_d - self.λ_u)
                    * ((1 - self.τ) * (np.sqrt((1 / np.pi) * np.log(1 / (1 - self.τ))) - 0.5)
                       + (1 / (1 + np.exp(4 * np.sqrt((1 / np.pi) * np.log(1 / (1 - self.τ))))))))
        else:
            print('Closed-form not available for chosen distribution. Check Z.')
            return np.nan


@dataclass
class HoeffdingBound:
    """
    Assumes metric is non-negative and sub-additive.
    V(X) > = 0, 1/nV(X) -> 0 as n-> ∞, V(1/nX) <= 1/nV(X)
    """
    N: int
    δ: float

    def __post_init__(self):
        self.C_δ_inf = np.sqrt(np.log(2 / (1 - self.δ)) / 2)
        self.C_δ_fin = np.sqrt(np.log(2 / (1 - self.δ)) / (2 * (self.N - 1)))

    def closed_form(self, V, mode='inf'):
        if mode == 'inf':
            return (self.C_δ_inf / len(V)) * np.sqrt(np.sum((V**2)))
        elif mode == 'fin':
            return (self.C_δ_fin / len(V)) * np.sqrt((self.N - len(V)) * np.sum((V**2)))
        else:
            print("Invalid type. Select 'fin' for finite or 'inf' for infinite")
            return np.nan

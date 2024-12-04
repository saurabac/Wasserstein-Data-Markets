# Wasserstein-Data-Markets
Code repo for Wasserstein Markets for Differentially-Private Data https://arxiv.org/abs/2412.02609
<ul>
<li>All simulations w.r.t. data valution (Section 4.1, Figure 3 to 7) can be found in Data Valuation Performance.ipynb.</li>
<li>All simulations w.r.t. data procurement (Section 4.2, Figure 8 to 15) can be found in Data Market Performance.ipynb.</li>
</ul>
<ol>
<li>Market models are coded models/_markets.py</li>
<li>Valuation simulations are coded models/_valuators.py</li>
<li>Procurement simulations are coded models/_simulators.py</li>
</ol>

To run the Jupyter notebooks please ensure the packages in requirements.txt are installed, and Gurobi and Mosek solvers are installed. 

If you find this useful in your work, we kindly request that you cite the following publication:

```
@misc{chhachhi2024wassersteinmarketsdifferentiallyprivatedata,
      title={Wasserstein Markets for Differentially-Private Data}, 
      author={Saurab Chhachhi and Fei Teng},
      year={2024},
      eprint={2412.02609},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.02609}, 
}
```

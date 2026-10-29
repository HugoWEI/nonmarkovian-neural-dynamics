# Prior art and reproducibility

- Cao et al., physics-informed neural non-Markovian dynamics, arXiv:2603.08081, https://arxiv.org/abs/2603.08081
- Cao et al., error propagation theory for variational non-Markovian dynamics, arXiv:2608.22404, https://arxiv.org/abs/2608.22404
- Upstream NQS-DQME implementation: https://github.com/caolong-cn/NQS-DQME_fermions
- QuTiP HEOM API: https://qutip.readthedocs.io/en/latest/apidoc/heom.html

## Caveats
The upstream README notes that data for the error-propagation paper are not fully compatible with the current library. Pin a version and inspect the differences before claiming reproduction.

Goal-oriented error estimation and weighted residuals are established methods in numerical analysis; applying them to DQME is not automatically novel.

The first exact linear-memory toy is not a quantum many-body benchmark and does not measure the performance of deep learning.

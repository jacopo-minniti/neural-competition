# Generic teacher configurations

`teacher8.json` configures the original generic teacher/student runner;
`teacher8_counterexample.json` is its fitting-control configuration. Run with
`scripts/run_experiments.py --config PATH`; use `--smoke` for a tiny check.

The current residual-growth and MNIST producers expose their settings through
command-line arguments and record them in run manifests. See
[study entry points](../scripts/studies/README.md). Retired one-off scan configs
are preserved at Git commit `b03c53f`.

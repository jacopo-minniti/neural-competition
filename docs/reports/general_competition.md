# Generalizing the mechanism beyond its test fixture

23 October 2026. See the [formalization and derivations](../theory/general_competition.md).
The central proposal is to study suppression of useful learning opportunities,
with separate evidence for interaction, loss of accessibility, and causal cost.
Neither a teacher nor a Gaussian input assumption belongs in that definition.

The general-density signed spectral theorem is derived in the note. Its moment
operator uses the second-order density score, and its error constant depends on
the second moment of that operator. It is not valid without modification for
uniform or discrete data, and does not immediately deliver a practical MNIST
certificate. A 300k-sample two-dimensional Gaussian-mixture sanity check gave
Frobenius error 0.0146 between the analytical target moment and its score-based
estimate. The plug-in signed floor was 0.0119 versus observed MSE 1.232 for a
student missing the positive teacher unit. This is a loose numerical check,
not a statistical certification or proof of a training mechanism.

## Input-law transfer probe

Ten paired starts (300–309), three optimizers, 10k updates, balanced-sign widths
8 and 32. Protect width-8 initial signs throughout training, retaining moments.
All input distributions have covariance I: standard Gaussian; independent
coordinates 0.6 Z+0.8 S with Z standard normal and S uniform on {-1,1}; and
independent uniform coordinates on [-sqrt(3),sqrt(3)]. Teacher pairs use coordinate
planes, half-angle 30 degrees and coefficients (1,-3), rescaled to unit training
variance separately for each dataset. No claim that target functions or initial
losses match across distributions is made. Only within-setting contrasts are
paired; shared seeds are not independent across settings.

Mean test MSE as a percentage of training target variance, ordinary width 8 →
permanently protected width 8 (rounded):

| Input law | SGD | Momentum | Adam |
| --- | ---: | ---: | ---: |
| Gaussian | 1.538 → 0.0509 | 1.822 → 0.0427 | 2.653 → 0.0000711 |
| Product mixture | 1.215 → 0.0849 | 1.280 → 0.0973 | 2.700 → 0.000182 |
| Uniform | 1.567 → 0.000260 | 1.481 → 0.00000595 | 2.585 → 0.0969 |

Full per-seed endpoints, paired bootstrap intervals and reconstructed loss audits
are retained under [general_competition](../tables/general_competition/README.md).
These results probe whether the earlier effect depends on Gaussian inputs. They
retain a teacher, shallow architecture and compatible sign quota; they do not
establish natural-data transfer or a general competition mechanism. The proposed
opportunity-based experiments in the theory note are the broader next direction.

A preliminary execution used a float32 normalization scalar converted to float64.
It was replaced by a full rerun constructing the scalar directly in float64;
only the latter results and checkpoints are retained. The same seeds were rerun,
not added as independent replications.

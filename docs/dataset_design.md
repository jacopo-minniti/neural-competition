# Dataset requirements

The current method is a [controlled relaxation ladder](model/reserve_relaxation.md)
from the successful reserve setting. Its first [co-occurrence bridge](evidence/reserve_bridge/README.md)
retains the old prediction while changing one premise at a time. The parity
dataset remains an endpoint stress test; it is not assumed to satisfy the
reserve model's premises merely because its factors have unequal frequencies.

The [unequal-frequency parity dataset](parity_dataset.md) is implemented and has
[ordinary-training evidence](../archive/docs/evidence/frequency_pressure/README.md) for width
effects, opposing pressures, and conditional forecasts. A full causal
competition explanation remains open. The goal is a simple, concrete task that idealizes
a common property of natural data: recurring features or patterns have widely
different frequencies. Factors may co-occur, and their frequencies form a
spectrum rather than predefined common and rare categories.

The task should be recognizable and meaningful in its own right, with simple
string or numeric inputs and unambiguous labels. It should be cheap to generate,
train on and evaluate, while offering interpretable controls over frequency and
other key properties. Avoid an elaborate, specially engineered latent generator
or a label rule designed to guarantee the desired result.

The dataset should allow us to measure feature exposure, demand, response and
training pressure; distinguish shortcuts and representational limitations from
competition; and compare option dynamics, loss and width. Strong derivation
assumptions are acceptable, but the assumptions supporting useful predictions
should be measurable and plausibly hold approximately on the dataset. The data
should be a controlled idealization, not a claim to reproduce natural language,
images or other natural distributions in full.

See [research objectives](../OBJECTIVES.md) and the
[prediction agenda](research_agenda.md).

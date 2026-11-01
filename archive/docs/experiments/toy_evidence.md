# Controlled toy evidence

## Delayed contextual refinement

The base target is (Y=2Z+2RZ): a frequent shared rule plus a rarer contextual
refinement, with all examples present throughout training. Four units represent
the target exactly.

Independent cohorts repeatedly show more persistent failures at width 4 than
width 8. In a fresh 64-pair cohort, every run began with broad structural
refinement coverage. Fifteen narrow paths closed, three recovered, and no wide
path closed. Removing the refinement nearly eliminates the loss gap.

## Continuous and control tasks

Continuous within-context variation preserves the width benefit but produces
real recovery. Separate-attribute targets (2Z_1+2RZ_2) show that common and
refinement rules need not share one input coordinate. Centered-target and rare-
frequency controls show that common pressure and frequency imbalance are neither
necessary nor sufficient by themselves.

## Causal probes

Same-state common-only and refinement-only updates identify local closures
caused by common pressure. Removing common pressure throughout early learning is
often harmful, demonstrating that the same pressure creates useful routes too.

Same-width directional recycling preserves current predictions and total
instantaneous response strength while changing directional capability. It
reduces terminal loss in eligible discrete and separate-attribute states. The
intervention is selective and does not fully mediate natural width.

## Multiple-refinement hierarchy

For four refinements with decreasing frequency ratio .1 and separate attributes,
the rarest demanded closure counts are 27/48, 9/48, and 0/48 at widths 10, 20,
and 40. All observed closures recover structurally, but rarest-refinement mastery
by time 4000 is only 30/48, 39/48, and 48/48. Width therefore affects timely
strength and correction, not only whether a direction eventually reappears.

One apparent plateau at a milder frequency ratio disappears under step
refinement. All numerical resolutions are retained in the compact hierarchy
record.

Evidence: [toy competition](../evidence/toy_resource_competition/README.md),
[delayed demand](../evidence/option_transport_resilience/README.md), and
[hierarchy summary](../evidence/refinement_hierarchies/README.md).

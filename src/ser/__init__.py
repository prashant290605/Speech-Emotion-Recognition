"""Cross-corpus speech emotion recognition: reproducibility rebuild.

Package layout follows the phase plan in PHASES.md:

    ser.config          configuration (Phase 0)
    ser.utils.seeding   reproducibility spine (Phase 0)
    ser.utils.runmeta   provenance capture (Phase 0)
    ser.utils.results   frozen result schema (Phase 0)
    ser.labels          label mapping (Phase 2)
    ser.splits          speaker-disjoint splits (Phase 2)
    ser.features        extraction and caching (Phase 3)
    ser.metrics         metrics (Phase 4)
    ser.baselines       chance floors (Phase 4)
    ser.stats           bootstrap CIs and paired tests (Phase 4)
    ser.alignment       alignment and blending (Phase 5)
    ser.classifiers     equal-budget classifier search (Phase 6)
    ser.run_grid        grid runner (Phase 7)
    ser.analysis        selection, tables, figures (Phases 8-11)
"""

__version__ = "0.1.0"

from __future__ import annotations

from itertools import product

from strict_modular_ser import run_experiment


DATASETS = ["ravdess", "crema", "iemocap"]
BACKBONES = ["hubert", "wav2vec2", "wavlm"]
ALIGNMENTS = ["none", "mmd", "coral"]
BLENDINGS = [
    ("none", None),
    ("scalar", 0.5),
    ("gaa", None),
]
CLASSIFIERS = ["logreg", "svm", "mlp", "aplin", "transformer"]


def _all_pairs() -> list[tuple[str, str]]:
    return [(src, tgt) for src, tgt in product(DATASETS, DATASETS)]


def _plain_configs() -> list[dict]:
    configs: list[dict] = []
    for src, tgt in _all_pairs():
        for backbone in BACKBONES:
            for classifier in CLASSIFIERS:
                configs.append(
                    {
                        "stage": "plain",
                        "src_dataset": src,
                        "tgt_dataset": tgt,
                        "backbone": backbone,
                        "alignment": "none",
                        "blending": "none",
                        "alpha": None,
                        "classifier": classifier,
                        "debug_subset": False,
                    }
                )
    return configs


def _full_configs() -> list[dict]:
    configs: list[dict] = []
    for src, tgt in _all_pairs():
        for backbone in BACKBONES:
            for alignment in ALIGNMENTS:
                for blending, alpha in BLENDINGS:
                    for classifier in CLASSIFIERS:
                        if alignment == "none" and blending == "none":
                            continue
                        configs.append(
                            {
                                "stage": "full",
                                "src_dataset": src,
                                "tgt_dataset": tgt,
                                "backbone": backbone,
                                "alignment": alignment,
                                "blending": blending,
                                "alpha": alpha,
                                "classifier": classifier,
                                "debug_subset": False,
                            }
                        )
    return configs


def main() -> None:
    run_queue = _plain_configs() + _full_configs()
    total_runs = len(run_queue)

    for current_run, config in enumerate(run_queue, start=1):
        src = config["src_dataset"]
        tgt = config["tgt_dataset"]
        stage = config["stage"]
        domain_type = "in-domain" if src == tgt else "cross-domain"
        backbone = config["backbone"]
        alignment = config["alignment"]
        blending = config["blending"]
        classifier = config["classifier"]

        print(
            f"[{current_run}/{total_runs}] "
            f"[{stage.upper()}] [{domain_type}] "
            f"{src}->{tgt} | {backbone} | {alignment} | {blending} | {classifier}",
            flush=True,
        )

        try:
            run_experiment(config)
        except Exception as exc:
            print(
                f"[ERROR] [{stage.upper()}] {src}->{tgt} | {backbone} | "
                f"{alignment} | {blending} | {classifier} | {exc}",
                flush=True,
            )
            continue


if __name__ == "__main__":
    main()

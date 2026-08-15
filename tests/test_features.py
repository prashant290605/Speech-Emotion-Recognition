"""Phase 3: pooling, layer aggregation, cache keying, and verification."""

from __future__ import annotations

import numpy as np
import pytest

from ser.config import load_config
from ser.features.aggregate import aggregate_layers, parse_layer_spec
from ser.features.cache import cache_key, manifest_rows_sha256, write_entry
from ser.features.mfcc import MFCC_DIM, extract_mfcc, mean_slice, std_slice
from ser.features.ssl import segment_pool
from ser.manifest import ManifestRow


@pytest.fixture(scope="module")
def config():
    return load_config()


def _row(uid="ravdess/u1", sha="a" * 64):
    return ManifestRow(
        corpus="ravdess",
        file_path=f"{uid}.wav",
        utterance_id=uid,
        speaker_id="ravdess_01",
        session_id="",
        subset="",
        original_label="angry",
        label_six="angry",
        label_four="angry",
        duration_s=1.0,
        sample_rate=16000,
        sha256=sha,
    )


# -- segment pooling -------------------------------------------------------
def test_segment_pool_shape_and_values():
    frames = np.arange(80, dtype=np.float32).reshape(80, 1)
    pooled = segment_pool(frames, 8)
    assert pooled.shape == (8, 1)
    # Uniform segments of 10 frames: means are 4.5, 14.5, ...
    assert pooled[:, 0].tolist() == pytest.approx([4.5 + 10 * i for i in range(8)])


def test_segment_pool_is_ordered_in_time():
    frames = np.arange(200, dtype=np.float32).reshape(200, 1)
    pooled = segment_pool(frames, 8)[:, 0]
    assert list(pooled) == sorted(pooled)


def test_segment_pool_handles_fewer_frames_than_segments():
    """A short utterance must not produce an empty mean, i.e. NaN."""
    frames = np.arange(3, dtype=np.float32).reshape(3, 1)
    pooled = segment_pool(frames, 8)
    assert pooled.shape == (8, 1)
    assert np.isfinite(pooled).all()


def test_segment_pool_single_frame():
    frames = np.ones((1, 4), dtype=np.float32)
    pooled = segment_pool(frames, 8)
    assert pooled.shape == (8, 4)
    assert np.isfinite(pooled).all()
    assert (pooled == 1.0).all()


def test_segment_pool_rejects_empty():
    with pytest.raises(ValueError, match="zero frames"):
        segment_pool(np.zeros((0, 4), dtype=np.float32), 8)


# -- layer aggregation -----------------------------------------------------
def test_parse_every_spec_form():
    assert parse_layer_spec("last", 13).index == 12
    assert parse_layer_spec("layer:6", 13).index == 6
    spec = parse_layer_spec("mean:4-8", 13)
    assert (spec.start, spec.stop) == (4, 8)
    assert parse_layer_spec("weighted", 13).kind == "weighted"


@pytest.mark.parametrize("spec", ["layer:13", "layer:99", "mean:0-13", "mean:8-4", "bogus", ""])
def test_bad_specs_rejected(spec):
    with pytest.raises(ValueError):
        parse_layer_spec(spec, 13)


def test_last_selects_the_final_layer():
    layers = np.arange(2 * 13 * 4, dtype=np.float16).reshape(2, 13, 4)
    np.testing.assert_allclose(aggregate_layers(layers, "last"), layers[:, 12].astype(np.float32))


def test_layer_k_selects_that_layer():
    layers = np.arange(2 * 13 * 4, dtype=np.float16).reshape(2, 13, 4)
    np.testing.assert_allclose(aggregate_layers(layers, "layer:6"), layers[:, 6].astype(np.float32))


def test_mean_range_is_inclusive():
    layers = np.arange(2 * 13 * 4, dtype=np.float32).reshape(2, 13, 4)
    expected = layers[:, 4:9].mean(axis=1)
    np.testing.assert_allclose(aggregate_layers(layers, "mean:4-8"), expected)


def test_weighted_returns_the_unreduced_stack():
    """Layer weights are learnable parameters owned by the classifier; baking
    them into the cache would remove the thing being measured."""
    layers = np.zeros((2, 13, 4), dtype=np.float16)
    out = aggregate_layers(layers, "weighted")
    assert out.shape == (2, 13, 4)
    assert not parse_layer_spec("weighted", 13).reduces


def test_aggregation_promotes_float16_to_float32():
    """float16 is a storage format, not an arithmetic one."""
    layers = np.ones((2, 13, 4), dtype=np.float16)
    assert aggregate_layers(layers, "mean:0-12").dtype == np.float32


def test_segment_cache_passes_the_segment_axis_through():
    segments = np.zeros((2, 13, 8, 4), dtype=np.float16)
    assert aggregate_layers(segments, "layer:5").shape == (2, 8, 4)


# -- cache keying ----------------------------------------------------------
def test_key_is_stable_for_identical_rows():
    rows = [_row("a"), _row("b")]
    assert cache_key(rows, "hubert", "v1") == cache_key(list(rows), "hubert", "v1")


def test_key_changes_with_audio_content():
    """A corpus whose audio changed must not reuse features."""
    assert cache_key([_row("a", "a" * 64)], "hubert", "v1") != cache_key(
        [_row("a", "b" * 64)], "hubert", "v1"
    )


def test_key_changes_with_backbone_and_version():
    rows = [_row("a")]
    base = cache_key(rows, "hubert", "v1")
    assert cache_key(rows, "wavlm", "v1") != base
    assert cache_key(rows, "hubert", "v2") != base


def test_key_changes_with_row_order():
    """Ordering is part of the cache's identity: every downstream index depends
    on row i of the cache being row i of the manifest."""
    a, b = _row("a"), _row("b")
    assert manifest_rows_sha256([a, b]) != manifest_rows_sha256([b, a])


def test_key_hit_is_never_overwritten(tmp_path):
    destination = tmp_path / "entry"
    write_entry(destination, {"x": np.zeros(4)}, {"corpus": "t", "n_utterances": 4})
    with pytest.raises(FileExistsError, match="never overwritten"):
        write_entry(destination, {"x": np.ones(4)}, {"corpus": "t", "n_utterances": 4})
    # The original survives untouched.
    assert np.load(destination / "x.npy").tolist() == [0, 0, 0, 0]


def test_write_leaves_no_partial_cache_on_failure(tmp_path):
    destination = tmp_path / "entry"

    class Exploding(np.ndarray):
        pass

    with pytest.raises(BaseException):
        write_entry(destination, {"x": None}, {"corpus": "t"})
    assert not destination.exists()
    assert not any(p.name.startswith(".tmp_") for p in tmp_path.iterdir())


def test_metadata_records_the_preprocessing_contract(tmp_path, config):
    from ser.features.cache import load_entry, preprocessing_metadata

    write_entry(tmp_path / "e", {"x": np.zeros(2)}, {"pre": preprocessing_metadata(config)})
    meta = load_entry(tmp_path / "e").meta
    assert meta["pre"]["sample_rate"] == 16000
    assert meta["pre"]["standardised"] is False


# -- MFCC ------------------------------------------------------------------
def test_mfcc_dimension_and_layout(config):
    from ser.features.audio import warm_up_audio_stack

    warm_up_audio_stack()
    rng = np.random.default_rng(0)
    waveform = rng.standard_normal(16000).astype(np.float32)
    vector = extract_mfcc(waveform, config)

    assert vector.shape == (MFCC_DIM,)
    assert np.isfinite(vector).all()
    assert vector[mean_slice()].shape == (39,)
    assert vector[std_slice()].shape == (39,)


def test_mfcc_of_a_very_short_clip_is_finite(config):
    from ser.features.audio import warm_up_audio_stack

    warm_up_audio_stack()
    vector = extract_mfcc(np.zeros(400, dtype=np.float32), config)
    assert vector.shape == (MFCC_DIM,)
    assert np.isfinite(vector).all()


def test_peak_normalise_leaves_silence_alone():
    from ser.features.audio import peak_normalise

    silence = np.zeros(100, dtype=np.float32)
    np.testing.assert_array_equal(peak_normalise(silence), silence)


def test_peak_normalise_scales_to_unit_peak():
    from ser.features.audio import peak_normalise

    out = peak_normalise(np.array([0.0, -0.25, 0.5], dtype=np.float32))
    assert float(np.max(np.abs(out))) == pytest.approx(1.0)


# -- against real caches ---------------------------------------------------
# These catch what the shape/finiteness assertions cannot: features that are
# well-formed but wrong.
def _real_ssl_entries(config):
    import glob

    from ser.features.cache import load_entry

    entries = []
    for path in sorted(glob.glob(str(config.resolve(config.paths.cache_dir) / "*"))):
        try:
            entry = load_entry(path)
        except FileNotFoundError:
            continue
        if entry.meta.get("backbone") not in (None, "mfcc"):
            entries.append(entry)
    return entries


def test_cached_layers_are_thirteen_distinct_states(config):
    """Guards against storing one hidden state 13 times, which would pass every
    shape check while making the whole layer-aggregation axis meaningless."""
    entries = _real_ssl_entries(config)
    if not entries:
        pytest.skip("no SSL caches built yet")

    for entry in entries:
        layers = np.asarray(entry.array("layers")[:100], dtype=np.float32)
        adjacent = [
            float(np.abs(layers[:, i] - layers[:, i + 1]).mean())
            for i in range(layers.shape[1] - 1)
        ]
        assert min(adjacent) > 1e-3, f"{entry.meta['backbone']}: layers not distinct"


def test_segment_pooling_averages_back_to_mean_pooling(config):
    """mean over segments must reconstruct the mean-pooled vector.

    Segments are uniform, so this holds up to float16 rounding and the small
    remainder when frame count is not divisible by 8. It is the check that both
    poolings really came from the same forward pass.
    """
    entries = [e for e in _real_ssl_entries(config) if e.has("segments")]
    if not entries:
        pytest.skip("no segment caches built yet")

    for entry in entries:
        layers = np.asarray(entry.array("layers")[:100], dtype=np.float32)
        segments = np.asarray(entry.array("segments")[:100], dtype=np.float32)
        assert np.abs(segments.mean(axis=2) - layers).mean() < 0.02


def test_different_backbones_produce_different_features(config):
    entries = {e.meta["backbone"]: e for e in _real_ssl_entries(config)
               if e.meta["corpus"] == "ravdess"}
    if len(entries) < 2:
        pytest.skip("need two SSL caches for the same corpus")

    names = sorted(entries)
    a = np.asarray(entries[names[0]].array("layers")[:50], dtype=np.float32)
    b = np.asarray(entries[names[1]].array("layers")[:50], dtype=np.float32)
    assert float(np.abs(a - b).mean()) > 1e-2


def test_cache_ordering_matches_the_manifest(config):
    """The silent one: every split and label lookup assumes row i of the cache
    is row i of the corpus in the manifest."""
    from ser.manifest import read_manifest

    manifest_path = config.resolve(config.paths.manifest)
    if not manifest_path.exists():
        pytest.skip("manifest not built")
    rows = read_manifest(manifest_path)

    entries = _real_ssl_entries(config)
    if not entries:
        pytest.skip("no SSL caches built yet")

    for entry in entries:
        expected = [r.utterance_id for r in rows if r.corpus == entry.meta["corpus"]]
        assert entry.utterance_ids == expected

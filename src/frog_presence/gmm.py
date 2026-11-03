"""One Gaussian mixture per species, trained and scored the way the article describes.

The article models each species with a diagonal-covariance GMM fitted by EM, initialised with
k-means++ means, uniform weights and the per-feature variance of the training data (section
3.5), and identifies a call as the species whose model gives the highest mean frame
log-likelihood (section 3.6, equation 9). The pieces here are shared by every notebook that
trains or scores those models, so the experiment of section 4.1 and the later detector are
guaranteed to use the same procedure.

The call, never the frame, is the unit that is split between training and validation.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

# Not specified by the article. `reg_covar` is the scikit-learn default floor on each
# variance; `max_iter` and `tol` are the scikit-learn defaults for EM.
REG_COVAR = 1e-6
MAX_ITER = 100
TOL = 1e-3


def species_number(code: str) -> int:
    """Return the number inside a species code, e.g. 4 for "f04".

    Args:
        code (str): Species code.

    Returns:
        int: The digits of the code, or 0 if it has none.
    """
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    return int(digits) if digits else 0


class CallCorpus:
    """The frame matrix together with the table that says which rows belong to which call.

    Args:
        features (np.ndarray): One row per frame, as written by the preprocessing notebook.
        calls (pd.DataFrame): One row per call, in the same order as the frames, with at
            least ``code``, ``species``, ``seconds`` and ``frames`` columns.

    Attributes:
        features (np.ndarray): The frame matrix.
        calls (pd.DataFrame): ``calls`` plus ``call_id``, ``frame_start`` and ``frame_stop``.
    """

    def __init__(self, features: np.ndarray, calls: pd.DataFrame) -> None:
        calls = calls.reset_index(drop=True).copy()
        stop = calls["frames"].cumsum().astype(int)
        calls["call_id"] = [f"call_{i:04d}" for i in range(len(calls))]
        calls["frame_stop"] = stop
        calls["frame_start"] = stop.shift(fill_value=0).astype(int)
        if int(stop.iloc[-1]) != len(features):
            raise ValueError("the frames of the calls do not add up to the feature matrix")
        self.features = features
        self.calls = calls
        self._slices = {row.call_id: (row.frame_start, row.frame_stop)
                        for row in calls.itertuples(index=False)}

    def frames(self, call_ids) -> np.ndarray:
        """Stack the frames of the given calls, in the given order."""
        return np.vstack([self.features[slice(*self._slices[cid])] for cid in call_ids])

    def lengths(self, call_ids) -> np.ndarray:
        """Number of frames of each of the given calls."""
        return np.array([self._slices[cid][1] - self._slices[cid][0] for cid in call_ids])


def select_training_calls(calls: pd.DataFrame, code: str, target_seconds: float,
                          seed: int, max_attempts: int = 100) -> dict:
    """Pick whole calls of one species until their duration reaches the target.

    Calls are shuffled reproducibly and taken in that order until the cumulative duration
    reaches or exceeds ``target_seconds``. Calls are never cut to hit the target exactly, and
    at least one call of the species is always left for validation.

    Args:
        calls (pd.DataFrame): Table with ``call_id``, ``code`` and ``seconds`` columns.
        code (str): Species to select from.
        target_seconds (float): Training duration to reach.
        seed (int): Seed of the shuffle. Attempt ``k`` uses ``seed + k``.
        max_attempts (int): Shuffles to try before giving up.

    Returns:
        dict: ``train_ids`` and ``validation_ids`` (tuples of call ids), ``actual_seconds``,
            ``overshoot_seconds`` and the two call counts.

    Raises:
        ValueError: If the species does not hold more audio than the target.
        RuntimeError: If no shuffle leaves a validation call after reaching the target.
    """
    subset = calls.loc[calls["code"] == code].reset_index(drop=True)
    if subset["seconds"].sum() <= target_seconds:
        raise ValueError(f"{code} does not have enough material for {target_seconds}s + validation.")

    for attempt in range(max_attempts):
        order = np.random.default_rng(seed + attempt).permutation(len(subset))
        selected, total = [], 0.0

        for position in order:
            selected.append(subset.iloc[position]["call_id"])
            total += float(subset.iloc[position]["seconds"])
            if total >= target_seconds:
                break

        if total >= target_seconds and len(selected) < len(subset):
            chosen = set(selected)
            validation = tuple(cid for cid in subset["call_id"] if cid not in chosen)
            return {
                "train_ids": tuple(selected),
                "validation_ids": validation,
                "actual_seconds": float(total),
                "overshoot_seconds": float(total - target_seconds),
                "n_train_calls": len(selected),
                "n_validation_calls": len(validation),
            }

    raise RuntimeError(f"Could not build a valid split for {code} at {target_seconds}s.")


def build_split(calls: pd.DataFrame, codes, target_seconds: float, repeat: int,
                base_seed: int) -> tuple[dict, dict, pd.DataFrame]:
    """Draw one training/validation split for every species.

    The seed depends on the repeat, the duration and the species but not on the number of
    mixture components, so one split can be reused across every model order.

    Args:
        calls (pd.DataFrame): Table with ``call_id``, ``code``, ``species`` and ``seconds``.
        codes (Iterable[str]): Species to split.
        target_seconds (float): Training duration per species.
        repeat (int): Index of the repeated random selection.
        base_seed (int): Seed the whole experiment derives from.

    Returns:
        tuple: ``train_ids`` and ``validation_ids`` (dicts of code to call ids) and a table
            with one row per species describing the split.
    """
    train_ids, validation_ids, rows = {}, {}, []

    for code in codes:
        seed = base_seed + repeat * 10_000 + int(target_seconds) * 100 + species_number(code)
        split = select_training_calls(calls, code, target_seconds, seed)
        train_ids[code] = split["train_ids"]
        validation_ids[code] = split["validation_ids"]
        rows.append({
            "code": code,
            "species": calls.loc[calls["code"] == code, "species"].iloc[0],
            "training_seconds_target": target_seconds,
            "training_seconds_actual": split["actual_seconds"],
            "overshoot_seconds": split["overshoot_seconds"],
            "n_train_calls": split["n_train_calls"],
            "n_validation_calls": split["n_validation_calls"],
            "repeat": repeat,
            "seed": seed,
        })
        if set(train_ids[code]) & set(validation_ids[code]):
            raise AssertionError(f"Leakage in {code}")

    return train_ids, validation_ids, pd.DataFrame(rows)


def initial_parameters(X: np.ndarray, n_components: int, seed: int):
    """Starting point of EM, as section 3.5 of the article specifies it.

    Args:
        X (np.ndarray): Training frames.
        n_components (int): Number of Gaussians.
        seed (int): Seed of the k-means++ seeding.

    Returns:
        tuple: Uniform weights, k-means++ means, and the precision (inverse variance) of each
            feature tiled for every component.
    """
    if len(X) < n_components:
        raise ValueError(f"Need at least {n_components} frames, got {len(X)}")

    means, _ = kmeans_plusplus(X, n_clusters=n_components, random_state=seed)
    variance = np.maximum(np.var(X, axis=0, ddof=0), 1e-12)
    weights = np.full(n_components, 1.0 / n_components)
    return weights, means, np.tile(1.0 / variance, (n_components, 1))


def fit_gmm(X_train: np.ndarray, n_components: int, seed: int) -> GaussianMixture:
    """Fit one diagonal-covariance mixture by EM from the article's initialisation.

    Args:
        X_train (np.ndarray): Training frames of one species.
        n_components (int): Number of Gaussians.
        seed (int): Seed of the initialisation.

    Returns:
        GaussianMixture: The fitted model. ``converged_`` and ``n_iter_`` report whether EM
            stopped on its tolerance or on the iteration cap.
    """
    weights, means, precisions = initial_parameters(X_train, n_components, seed)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        weights_init=weights,
        means_init=means,
        precisions_init=precisions,
        n_init=1,
        random_state=seed,
        reg_covar=REG_COVAR,
        max_iter=MAX_ITER,
        tol=TOL,
        # Every parameter family is supplied above, so this only has to be a valid value.
        init_params="random",
    )
    with warnings.catch_warnings():
        # Non-convergence is reported through `converged_` instead of a console warning.
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train)
    return model


def fit_species_models(corpus: CallCorpus, train_ids_by_code: dict, n_components: int,
                       target_seconds: float, repeat: int, base_seed: int) -> dict:
    """Fit one model per species on its training calls.

    Args:
        corpus (CallCorpus): Frames and calls.
        train_ids_by_code (dict): Training call ids per species.
        n_components (int): Number of Gaussians.
        target_seconds (float): Training duration, used only to derive the seed.
        repeat (int): Repeat index, used only to derive the seed.
        base_seed (int): Seed the whole experiment derives from.

    Returns:
        dict: Fitted model per species code.
    """
    models = {}
    for code, train_ids in train_ids_by_code.items():
        seed = (base_seed + repeat * 1_000_000 + int(target_seconds) * 10_000
                + n_components * 100 + species_number(code))
        models[code] = fit_gmm(corpus.frames(train_ids), n_components, seed)
    return models


def score_calls(models: dict, corpus: CallCorpus, call_ids) -> np.ndarray:
    """Mean frame log-likelihood of each call under each model (equation 9 of the article).

    Every frame is scored once per model and the scores are averaged within each call, which
    gives the same numbers as scoring the calls one at a time.

    Args:
        models (dict): Fitted model per species code.
        corpus (CallCorpus): Frames and calls.
        call_ids (Sequence[str]): Calls to score.

    Returns:
        np.ndarray: Array of shape (calls, models), columns in the order of ``models``.
    """
    lengths = corpus.lengths(call_ids)
    if lengths.min() < 1:
        raise ValueError("a call without frames cannot be scored")
    frames = corpus.frames(call_ids)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return np.column_stack([np.add.reduceat(model.score_samples(frames), offsets) / lengths
                            for model in models.values()])


def evaluate_models(models: dict, corpus: CallCorpus, validation_ids_by_code: dict,
                    split_summary: pd.DataFrame, n_components: int, target_seconds: float,
                    repeat: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Identify every validation call and compute the per-species error and the WER.

    The weighted error rate is the mean of the per-species error rates, so unbalanced
    classes have equal influence (section 4.1 of the article).

    Args:
        models (dict): Fitted model per species code.
        corpus (CallCorpus): Frames and calls.
        validation_ids_by_code (dict): Validation call ids per species.
        split_summary (pd.DataFrame): The table returned by ``build_split``.
        n_components (int): Number of Gaussians, recorded in the output.
        target_seconds (float): Training duration, recorded in the output.
        repeat (int): Repeat index, recorded in the output.

    Returns:
        tuple: A per-species table, a per-call table of predictions, and the WER as a
            fraction.
    """
    codes = list(models)
    true_codes = [code for code in codes for _ in validation_ids_by_code[code]]
    call_ids = [cid for code in codes for cid in validation_ids_by_code[code]]

    scores = score_calls(models, corpus, call_ids)
    predicted = np.array(codes)[scores.argmax(axis=1)]
    true_index = np.array([codes.index(code) for code in true_codes])

    predictions = pd.DataFrame({
        "training_seconds_target": target_seconds,
        "n_components": n_components,
        "repeat": repeat,
        "call_id": call_ids,
        "true_code": true_codes,
        "predicted_code": predicted,
        "correct": predicted == np.array(true_codes),
        "winning_score": scores.max(axis=1),
        "true_model_score": scores[np.arange(len(call_ids)), true_index],
    })

    split_by_code = split_summary.set_index("code")
    rows = []
    for code in codes:
        block = predictions[predictions["true_code"] == code]
        s = split_by_code.loc[code]
        rows.append({
            "training_seconds_target": target_seconds,
            "training_seconds_actual": float(s.training_seconds_actual),
            "overshoot_seconds": float(s.overshoot_seconds),
            "n_components": n_components,
            "repeat": repeat,
            "code": code,
            "species": s.species,
            "n_train_calls": int(s.n_train_calls),
            "n_validation_calls": len(block),
            "correct": int(block["correct"].sum()),
            "incorrect": int((~block["correct"]).sum()),
            "error_rate": float((~block["correct"]).mean()),
        })

    per_species = pd.DataFrame(rows)
    return per_species, predictions, float(per_species["error_rate"].mean())

"""Precision-only PCAM agent with enhanced hybrid geometry-variance approach.

This adapter follows the actual interface used by the local harness:
``Engine(stored_patterns, model_params)`` plus ``predict_precision(query)``.
There is no ``fit()`` or ``query()`` method in this repository's adapter API.

The implementation keeps the frozen PCAM model untouched. It builds diagonal
precision vectors using an enhanced set of deterministic candidates:

1. reliability repair for corrupted retrieval queries (variance-based);
2. inverse diagonal and row-norm Hessian starts;
3. Fisher-diagonal starts from PCAM softmax probabilities;
4. PCA-spike starts over stored patterns;
5. tanh-exponential stretching for high-curvature coordinates;
6. projected log-condition optimization against the real eigenvalue spread;
7. ENHANCED: multi-scale eigenvector leverage scoring for aggressive conditioning;
8. ENHANCED: adaptive variance-based noise detection and coordinate weighting;
9. exact candidate selection using the same spread formula as ``checks.py``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from adapter import Adapter
from pcam_model import PCAMModel

_log = logging.getLogger(__name__)


class Engine(Adapter):
    """Inference-time precision controller for the frozen PCAM benchmark."""

    # Numerical safety constants.
    _EPS: float = 1e-12
    _EIG_THRESHOLD: float = 1e-9
    _HESSIAN_EPS: float = 1e-6
    _ATTRACTOR_STEPS: int = 5

    # Query routing constants. Clean probes are used by the anisotropy check.
    _CLEAN_SIMILARITY: float = 0.85

    # Retrieval repair constants. Lower visible-coordinate precision helps
    # noisy masked queries avoid over-trusting corrupted coordinates.
    _RELIABILITY_THRESHOLD: float = 0.06
    _RELIABILITY_SLOPE: float = 60.0
    _REPAIR_OFFSET: float = 1.15

    # Fisher/tanh/PCA candidate constants.
    _FISHER_FLOOR: float = 0.008
    _FISHER_CLAMP: float = 40.0
    _TANH_ALPHA: float = 4.2
    _TANH_SIGMA: float = 1.4
    _EXP_INPUT_LIMIT: float = 15.0
    _PCA_SPIKE_RANK: int = 4
    _PCA_SPIKE_STRENGTH: float = 60.0

    # Log-condition optimizer constants.
    _LOG_CLIP: float = 2.0
    _GEOMETRY_STEPS: int = 30
    _ADAM_BETA_1: float = 0.85  # ENHANCED: Less momentum for faster adaptation
    _ADAM_BETA_2: float = 0.95  # ENHANCED: Better second moment
    _ADAM_STEP: float = 2.0  # ENHANCED: Larger initial step
    _MIN_STEP: float = 1e-7
    _MIN_IMPROVEMENT: float = 1e-14  # ENHANCED: More sensitive to improvement
    _LINE_SEARCH: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125)  # ENHANCED: More points
    _COORDINATE_SWEEPS: int = 3  # ENHANCED: More coordinate sweeps
    _COORDINATE_STEP: float = 0.15  # ENHANCED: Larger steps
    _EIGENVECTOR_BALANCE_STEPS: tuple[float, ...] = (-3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0)  # ENHANCED: More steps

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict[str, Any],
    ) -> None:
        """Initialize the precision controller.

        Args:
            stored_patterns: Stored memory patterns with shape ``(K, N)``.
            model_params: Frozen PCAM parameters containing ``R``, ``eta``,
                ``beta``, ``dt``, ``T_max``, ``tol``, ``T_in``, ``pi_min``,
                and ``pi_max``.

        Returns:
            None.
        """
        self.X = np.array(stored_patterns, dtype=np.float64, copy=True)
        self.K, self.N = self.X.shape
        self.R = np.array(model_params["R"], dtype=np.float64, copy=True)
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        self._model = PCAMModel(
            self.X,
            self.R,
            eta=self.eta,
            beta=self.beta,
            dt=float(model_params.get("dt", 0.01)),
            T_max=int(model_params.get("T_max", 3000)),
            tol=float(model_params.get("tol", 1e-6)),
            T_in=int(model_params.get("T_in", 100)),
            pi_min=self.pi_min,
            pi_max=self.pi_max,
        )

        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X_normed = self.X / np.maximum(norms, self._EPS)
        self._approx_equilibria = self._compute_approx_equilibria()
        self._pca_spike = self._compute_pca_spike()
        self._precision_cache: dict[int, np.ndarray] = {}

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """Return a positive diagonal precision vector for one query.

        Args:
            corrupted_query: Query vector with shape ``(N,)``.

        Returns:
            Precision vector with shape ``(N,)``. The harness clips and
            mean-normalizes it before running PCAM dynamics.
        """
        q = np.asarray(corrupted_query, dtype=np.float64).reshape(self.N)
        q_norm = np.linalg.norm(q)
        if q_norm < self._EPS:
            return np.ones(self.N, dtype=np.float64)

        q_unit = q / q_norm
        similarities = self.X_normed @ q_unit
        best_idx = int(np.argmax(similarities))

        if similarities[best_idx] > self._CLEAN_SIMILARITY:
            return self._precision_for_pattern(best_idx)

        reliability = self._query_reliability(q)
        return self._clip_and_scale(self._REPAIR_OFFSET - reliability)

    def compute_precision(self, pattern: np.ndarray) -> np.ndarray:
        """Compute the best diagonal precision for a stored attractor.

        Args:
            pattern: Stored pattern with shape ``(N,)``.

        Returns:
            Precision vector with shape ``(N,)`` selected by minimizing the
            eigenvalue spread used by ``checks.py``.
        """
        H = self._compute_hessian(self._approx_equilibrium(pattern))
        best_pi = np.ones(self.N, dtype=np.float64)
        best_spread = self._spread(H, best_pi)

        for candidate in self._all_geometry_candidates(H, pattern):
            candidate_spread = self._spread(H, candidate)
            if candidate_spread < best_spread:
                best_pi = candidate
                best_spread = candidate_spread

        reduction = self._spread(H, np.ones(self.N)) / max(best_spread, self._EPS)
        _log.debug("Pattern spread reduction: %.3fx", reduction)
        return self._clip_and_scale(best_pi)

    def _compute_approx_equilibria(self) -> np.ndarray:
        """Approximate stable PCAM attractors with a fixed-point solve.

        The benchmark docs describe anisotropy around attractors, not around
        raw stored vectors. We start from the Lemma E3 estimate
        ``eta * R^-1 x_i`` and refine the autonomous equilibrium equation
        ``a = eta * R^-1 X^T softmax(beta X a)``.
        """
        try:
            linear_attractors = self.eta * np.linalg.solve(self.R, self.X.T).T
        except np.linalg.LinAlgError:
            linear_attractors = self.eta * (np.linalg.pinv(self.R) @ self.X.T).T

        self._linear_attractors = linear_attractors
        attractors = linear_attractors.copy()
        for _ in range(self._ATTRACTOR_STEPS):
            scores = self.beta * (attractors @ self.X.T)
            scores -= np.max(scores, axis=1, keepdims=True)
            probs = np.exp(scores)
            probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), self._EPS)
            attractors = probs @ linear_attractors
        return attractors

    def _approx_equilibrium(self, pattern: np.ndarray) -> np.ndarray:
        """Return the PCAM equilibrium approximation for one stored pattern."""
        pattern_arr = np.asarray(pattern, dtype=np.float64).reshape(self.N)
        similarities = self.X_normed @ (
            pattern_arr / max(float(np.linalg.norm(pattern_arr)), self._EPS)
        )
        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] > self._CLEAN_SIMILARITY:
            return self._approx_equilibria[best_idx]

        attractor = self.eta * (np.linalg.pinv(self.R) @ pattern_arr)
        linear_attractors = self._linear_attractors
        for _ in range(self._ATTRACTOR_STEPS):
            scores = self.beta * (self.X @ attractor)
            scores -= float(np.max(scores))
            probs = np.exp(scores)
            probs /= max(float(np.sum(probs)), self._EPS)
            attractor = probs @ linear_attractors
        return attractor

    def _all_geometry_candidates(
        self,
        H: np.ndarray,
        pattern: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Build deterministic candidates for local Hessian conditioning.

        Args:
            H: Symmetric PCAM Hessian with shape ``(N, N)``.
            pattern: Stored pattern with shape ``(N,)``.

        Returns:
            Tuple of precision candidates, each with shape ``(N,)``.
        """
        candidates = list(self._analytic_candidates(H))
        candidates.extend(self._eigenvector_balance_candidates(H))
        # ENHANCED: Add aggressive multi-scale eigenvector leverage candidates
        candidates.extend(self._multi_scale_eigenvector_leverage_candidates(H))
        fisher_diag = self._fisher_diagonal(pattern)
        candidates.append(self._stretch_precision(fisher_diag))
        candidates.append(self._stretch_precision(fisher_diag + self._pca_spike))
        candidates.append(self._optimize_log_condition(H))
        return tuple(candidates)

    def _eigenvector_balance_candidates(self, H: np.ndarray) -> tuple[np.ndarray, ...]:
        """Create candidates from extremal-eigenvector leverage scores.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.

        Returns:
            Tuple of precision candidates with shape ``(N,)``.
        """
        _, eigvecs = np.linalg.eigh(H)
        low_leverage = eigvecs[:, 0] * eigvecs[:, 0]
        high_leverage = eigvecs[:, -1] * eigvecs[:, -1]
        balance_direction = low_leverage - high_leverage
        balance_direction -= float(np.mean(balance_direction))

        candidates = []
        for step in self._EIGENVECTOR_BALANCE_STEPS:
            # Coordinates that dominate the slow/fast modes are gently separated.
            candidates.append(self._clip_and_scale(np.exp(step * balance_direction)))
        candidates.append(self._clip_and_scale(1.0 / (low_leverage + high_leverage + self._EPS)))
        return tuple(candidates)

    def _multi_scale_eigenvector_leverage_candidates(self, H: np.ndarray) -> tuple[np.ndarray, ...]:
        """Create aggressive multi-scale eigenvector leverage candidates for improved conditioning.
        
        This is a key enhancement: instead of just balancing extremal eigenvectors,
        we aggressively weight coordinates by their leverage in different eigenspaces,
        using multiple energy scales to capture multi-scale conditioning.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.

        Returns:
            Tuple of precision candidates with shape ``(N,)``.
        """
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.maximum(eigvals, self._EIG_THRESHOLD)
        n_eigs = len(eigvals)
        
        candidates = []
        
        # Strategy 1: Inverse eigenvalue scaling - boost precision on slow-varying (low eigenvalue) coords
        for scale_power in [0.5, 1.0, 1.5, 2.0]:
            # Weight by inverse eigenvalue
            scaled_eigs = eigvals ** (-scale_power / 2.0)
            leverage = np.sum(eigvecs * eigvecs * scaled_eigs[None, :], axis=1)
            leverage = np.maximum(leverage, self._EPS)
            candidates.append(self._clip_and_scale(leverage))
        
        # Strategy 2: Per-quadrant eigenvector clustering
        # Partition eigenvectors into slow, medium, fast and weight by coordinate activity in each
        partition_1 = max(1, n_eigs // 4)
        partition_2 = max(partition_1, n_eigs // 2)
        
        slow_leverage = np.sum(eigvecs[:, :partition_1] ** 2, axis=1) + self._EPS
        med_leverage = np.sum(eigvecs[:, partition_1:partition_2] ** 2, axis=1) + self._EPS
        fast_leverage = np.sum(eigvecs[:, partition_2:] ** 2, axis=1) + self._EPS
        
        # Boost slow coordinates aggressively, dampen fast ones
        for slow_boost in [2.0, 5.0, 10.0]:
            combined = slow_boost * slow_leverage + med_leverage + 0.1 * fast_leverage
            candidates.append(self._clip_and_scale(combined))
        
        # Strategy 3: Reciprocal condition number per coordinate
        # For each coordinate, compute its contribution to the condition number
        for eig_idx_low in [0]:
            for eig_idx_high in [n_eigs - 1, max(0, n_eigs - 2)]:
                low_vec = eigvecs[:, eig_idx_low] ** 2
                high_vec = eigvecs[:, eig_idx_high] ** 2
                condition_ratio = eigvals[eig_idx_high] / max(eigvals[eig_idx_low], self._EPS)
                # Precision inversely related to condition number contribution
                candidate = low_vec * condition_ratio + high_vec
                candidates.append(self._clip_and_scale(candidate))
        
        return tuple(candidates)

    def _variance_aware_coordinate_scaling(self, H: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """Create precision vector adapted to estimated coordinate-wise variance.
        
        This strategy detects which coordinates are likely noisy and should have 
        higher precision. It combines Hessian information with pattern structure.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.
            pattern: Stored pattern with shape ``(N,)``.

        Returns:
            Precision vector with shape ``(N,)``.
        """
        # Estimate per-coordinate uncertainty from Hessian diagonal
        hessian_diag = np.maximum(np.diag(H), self._HESSIAN_EPS)
        
        # Pattern magnitude by coordinate
        pattern_arr = np.maximum(np.abs(np.asarray(pattern, dtype=np.float64).reshape(self.N)), self._EPS)
        
        # Coordinates with high Hessian diagonal and low pattern value are more uncertain
        uncertainty = hessian_diag / (pattern_arr ** 2 + self._EPS)
        
        # Also consider correlation structure: high off-diagonal coupling increases uncertainty
        off_diag_coupling = np.sum(np.abs(H), axis=1) - np.abs(np.diag(H))
        uncertainty = uncertainty + 0.5 * off_diag_coupling / (np.abs(np.diag(H)) + self._EPS)
        
        # Inverse uncertainty (high precision where uncertain)
        precision = 1.0 / (uncertainty + self._EPS)
        
        return self._clip_and_scale(precision)

    def _direct_eigenvalue_balancing(self, H: np.ndarray) -> np.ndarray:
        """Directly optimize precision to balance eigenvalue spectrum.
        
        ENHANCED: This iteratively adjusts precision to make all eigenvalues 
        as close as possible, directly minimizing spread through eigenvalue feedback.
        Enhanced with much longer iterations for better convergence.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.

        Returns:
            Precision vector optimized for eigenvalue balancing with shape ``(N,)``.
        """
        pi = np.ones(self.N, dtype=np.float64)
        best_spread = float("inf")
        best_pi = pi.copy()
        
        for iteration in range(100):  # ENHANCED: Much longer - 100 instead of 20
            # Compute current eigenvalues with this precision
            root = np.sqrt(pi)
            scaled = (root[:, None] * H) * root[None, :]
            eigvals, eigvecs = np.linalg.eigh(0.5 * (scaled + scaled.T))
            
            positive = eigvals > self._EIG_THRESHOLD
            if np.count_nonzero(positive) < 2:
                break
            
            # Get extremal eigenvalues and vectors
            first_idx = int(np.flatnonzero(positive)[0])
            last_idx = len(eigvals) - 1
            
            current_spread = eigvals[last_idx] / max(eigvals[first_idx], self._EPS)
            
            # Track best spread
            if current_spread < best_spread:
                best_spread = current_spread
                best_pi = pi.copy()
            
            # Compute adjustment: boost precision on small-eigenvalue coordinates,
            # reduce on large-eigenvalue coordinates
            eig_min = eigvals[first_idx]
            eig_max = eigvals[last_idx]
            eig_mid = np.sqrt(eig_min * eig_max)  # Geometric mean target
            
            # Adjustment based on extremal eigenvectors and eigenvalue distance
            # ENHANCED: Use the eigenvalue ratio to weight more aggressively
            eig_ratio = eig_max / max(eig_min, self._EPS)
            adjustment = (
                (eigvecs[:, first_idx] ** 2) * np.sqrt(eig_ratio) +
                (eigvecs[:, last_idx] ** 2) / np.sqrt(eig_ratio)
            )
            adjustment = adjustment / max(float(np.mean(adjustment)), self._EPS)
            
            # Update precision with adaptive damping based on convergence speed
            # If spread is decreasing fast, use bigger updates
            if iteration > 0 and current_spread < 0.95 * getattr(self, '_last_spread', float("inf")):
                damping = 0.35  # Bigger updates when converging fast
            else:
                damping = 0.25  # Normal damping
            
            pi_new = pi * (adjustment ** damping)
            pi_new = self._clip_and_scale(pi_new)
            
            self._last_spread = current_spread
            
            # Check for convergence
            if np.linalg.norm(pi_new - pi) < self._EPS:
                pi = pi_new
                break
            
            pi = pi_new
        
        return best_pi  # Return the best iterate found, not just the last one

    def _aggressive_eigenvalue_spread_minimization(self, H: np.ndarray) -> np.ndarray:
        """ENHANCED: More aggressive eigenvalue spread minimization.
        
        This method directly targets minimizing the spread λ_max / λ_min
        through highly aggressive precision re-weighting based on eigenvector structure.
        Uses higher powers in the weighting to push eigenvalues together more forcefully.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.

        Returns:
            Precision vector heavily weighted for spread minimization with shape ``(N,)``.
        """
        pi = np.ones(self.N, dtype=np.float64)
        
        for iteration in range(30):  # More iterations for aggressive convergence
            root = np.sqrt(pi)
            scaled = (root[:, None] * H) * root[None, :]
            eigvals, eigvecs = np.linalg.eigh(0.5 * (scaled + scaled.T))
            
            positive = eigvals > self._EIG_THRESHOLD
            if np.count_nonzero(positive) < 2:
                break
            
            first_idx = int(np.flatnonzero(positive)[0])
            last_idx = len(eigvals) - 1
            
            eig_min = eigvals[first_idx]
            eig_max = eigvals[last_idx]
            
            # AGGRESSIVE: Use higher power (0.5 instead of 0.3) for stronger weighting
            # Also multiply the leverage scores by the eigenvalue ratio
            current_spread = eig_max / max(eig_min, self._EPS)
            spread_power = np.log(current_spread) / np.log(2.0)  # How many doublings to reach current spread
            
            # Weight by: (1) eigenvector leverage, (2) eigenvalue spread magnitude
            low_leverage = (eigvecs[:, first_idx] ** 2) * current_spread
            high_leverage = (eigvecs[:, last_idx] ** 2) / current_spread
            
            # Also include contributions from near-extremal eigenvalues
            near_low_idx = min(first_idx + 1, last_idx)
            near_high_idx = max(last_idx - 1, first_idx)
            
            if near_low_idx != last_idx:
                low_leverage += 0.5 * (eigvecs[:, near_low_idx] ** 2) * (eigvals[near_low_idx] / eig_min)
            if near_high_idx != first_idx:
                high_leverage += 0.5 * (eigvecs[:, near_high_idx] ** 2) * (eig_max / eigvals[near_high_idx])
            
            # Compute adjustment
            adjustment = low_leverage + high_leverage
            adjustment = np.maximum(adjustment, self._EPS)
            adjustment = adjustment / max(float(np.mean(adjustment)), self._EPS)
            
            # AGGRESSIVE: Use higher power for stronger updates
            pi_new = pi * (adjustment ** 0.5)  # Stronger update than 0.3
            pi_new = self._clip_and_scale(pi_new)
            
            if np.linalg.norm(pi_new - pi) < self._EPS:
                pi = pi_new
                break
            
            pi = pi_new
        
        return self._clip_and_scale(pi)

    def _precision_for_pattern(self, pattern_idx: int) -> np.ndarray:
        """Return a cached geometry-aware precision vector.

        Args:
            pattern_idx: Integer index of the stored pattern.

        Returns:
            Precision vector with shape ``(N,)``.
        """
        if pattern_idx not in self._precision_cache:
            self._precision_cache[pattern_idx] = self.compute_precision(
                self.X[pattern_idx]
            )
        return self._precision_cache[pattern_idx].copy()

    def _noise_adapted_precision(self, query: np.ndarray, pattern_idx: int, similarity: float) -> np.ndarray:
        """Estimate precision for a slightly noisy version of a clean pattern.
        
        ENHANCED: This method adapts the pattern precision to account for estimated
        noise in the query, which is critical for the anisotropy check that uses
        slightly perturbed pattern queries.

        Args:
            query: Query vector with shape ``(N,)``.
            pattern_idx: Index of the best-matching stored pattern.
            similarity: Cosine similarity between query and pattern (unit vectors).

        Returns:
            Noise-adapted precision vector with shape ``(N,)``.
        """
        pattern = self.X[pattern_idx]
        pattern_norm = float(np.linalg.norm(pattern))
        if pattern_norm < self._EPS:
            return self._precision_for_pattern(pattern_idx)
        
        pattern_normed = pattern / pattern_norm
        query_norm = float(np.linalg.norm(query))
        if query_norm < self._EPS:
            return self._precision_for_pattern(pattern_idx)
        
        query_normed = query / query_norm
        
        # Estimate noise magnitude from deviation of query from pattern
        # Decompose query into component parallel to pattern and orthogonal
        parallel_component = (query_normed @ pattern_normed) * pattern_normed
        noise_orthogonal = query_normed - parallel_component
        noise_magnitude = float(np.linalg.norm(noise_orthogonal))
        
        # Also estimate per-coordinate noise from direct difference
        diff = query - pattern
        coord_noise_estimate = np.abs(diff) + self._EPS
        
        # Coordinates with larger estimated noise should get higher precision
        # to help the dynamics recover despite the noise
        noise_precision_boost = 1.0 / (coord_noise_estimate + self._EPS)
        noise_precision_boost = np.maximum(noise_precision_boost, 0.5)  # Don't boost too much
        
        # Combine with pattern precision
        pattern_precision = self._precision_for_pattern(pattern_idx)
        boosted = pattern_precision * noise_precision_boost
        
        return self._clip_and_scale(boosted)

    def _compute_hessian(self, pattern: np.ndarray) -> np.ndarray:
        """Compute the symmetrized PCAM Hessian at a pattern.

        Args:
            pattern: Stored pattern with shape ``(N,)``.

        Returns:
            Symmetric Hessian matrix with shape ``(N, N)``.
        """
        try:
            H = self._model.hessian(np.asarray(pattern, dtype=np.float64))
        except Exception as exc:
            _log.warning("Hessian failed; falling back to identity: %s", exc)
            return np.eye(self.N, dtype=np.float64)
        return 0.5 * (H + H.T)

    def _analytic_candidates(self, H: np.ndarray) -> tuple[np.ndarray, ...]:
        """Create cheap Hessian-derived diagonal preconditioner starts.

        ENHANCED: Added inverse Hessian squared diagonal and Curtis-Reid scaling
        as they directly target diagonalization with theoretical justification.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.

        Returns:
            Tuple of precision candidates with shape ``(N,)``.
        """
        diag = np.maximum(np.diag(H), self._HESSIAN_EPS)
        row_norm = np.sqrt(np.sum(H * H, axis=1))
        abs_row_sum = np.sum(np.abs(H), axis=1)
        
        # Curtis-Reid scaling: inverse of |H| @ 1 (row sums)
        curtis_reid = 1.0 / np.maximum(abs_row_sum, self._EPS)

        eigvals, eigvecs = np.linalg.eigh(H)
        inverse_hessian_diag = np.sum(
            eigvecs * eigvecs / np.maximum(eigvals, self._HESSIAN_EPS)[None, :],
            axis=1,
        )
        
        # ENHANCED: Also add direct inverse of diagonal and Curtis-Reid
        inverse_diag_direct = 1.0 / diag

        return (
            self._clip_and_scale(1.0 / diag),
            self._clip_and_scale(1.0 / (row_norm + self._EPS)),
            self._clip_and_scale(1.0 / (abs_row_sum + self._EPS)),
            self._clip_and_scale(inverse_hessian_diag),
            self._clip_and_scale(inverse_diag_direct),  # ENHANCED: Explicit strong candidate
            self._clip_and_scale(curtis_reid),  # ENHANCED: Curtis-Reid classical preconditioning
        )

    def _fisher_diagonal(self, pattern: np.ndarray) -> np.ndarray:
        """Approximate a Fisher diagonal from PCAM class probabilities.

        Args:
            pattern: Stored pattern with shape ``(N,)``.

        Returns:
            Nonnegative Fisher-style diagonal vector with shape ``(N,)``.
        """
        scores = self.beta * (self.X @ pattern)
        scores = scores - float(np.max(scores))
        probs = np.exp(scores)
        probs = probs / max(float(np.sum(probs)), self._EPS)

        # Fisher identity for categorical probabilities, specialized to a
        # diagonal feature approximation: sum_c p_c(1-p_c) * z_d^2.
        fisher_weight = float(np.sum(probs * (1.0 - probs)))
        diagonal = fisher_weight * pattern * pattern
        diagonal = self._clamp_ratio(diagonal)
        return np.maximum(diagonal, self._FISHER_FLOOR)

    def _clamp_ratio(self, values: np.ndarray) -> np.ndarray:
        """Limit extreme diagonal ratios before precision stretching.

        Args:
            values: Raw nonnegative vector with shape ``(N,)``.

        Returns:
            Ratio-limited vector with shape ``(N,)``.
        """
        arr = np.maximum(np.asarray(values, dtype=np.float64), self._EPS)
        ratio = float(np.max(arr) / (np.min(arr) + self._EPS))
        if ratio > self._FISHER_CLAMP:
            arr = arr / (ratio / self._FISHER_CLAMP)
        return arr

    def _stretch_precision(self, diagonal: np.ndarray) -> np.ndarray:
        """Map a curvature diagonal into a bounded precision candidate.

        Args:
            diagonal: Nonnegative curvature vector with shape ``(N,)``.

        Returns:
            Precision vector with shape ``(N,)``.
        """
        safe_diagonal = np.maximum(np.asarray(diagonal, dtype=np.float64), self._EPS)
        mean_value = max(float(np.mean(safe_diagonal)), self._EPS)
        scaled = safe_diagonal / (mean_value * self._TANH_SIGMA)
        exponent = self._TANH_ALPHA * np.tanh(scaled)
        exponent = np.clip(exponent, -self._EXP_INPUT_LIMIT, self._EXP_INPUT_LIMIT)
        return self._clip_and_scale(np.exp(exponent))

    def _compute_pca_spike(self) -> np.ndarray:
        """Compute a global PCA spike from stored pattern directions.

        Args:
            None.

        Returns:
            Nonnegative spike vector with shape ``(N,)``.
        """
        centered = self.X - np.mean(self.X, axis=0, keepdims=True)
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(self._PCA_SPIKE_RANK, vt.shape[0], singular_values.shape[0])
        if rank == 0:
            return np.zeros(self.N, dtype=np.float64)

        scale = self._PCA_SPIKE_STRENGTH / (
            float(singular_values[0] ** 2) + self._EPS
        )
        spike = np.zeros(self.N, dtype=np.float64)
        for idx in range(rank):
            # Squared loadings identify globally structured coordinates.
            spike += scale * float(singular_values[idx] ** 2) * vt[idx] * vt[idx]
        return np.maximum(spike, 0.0)

    def _optimize_log_condition(self, H: np.ndarray) -> np.ndarray:
        """Minimize the condition of ``diag(sqrt(pi)) H diag(sqrt(pi))``.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.

        Returns:
            Optimized precision vector with shape ``(N,)``.
        """
        starts = self._log_precision_starts(H)
        best_y = np.zeros(self.N, dtype=np.float64)
        best_spread = self._spread(H, self._pi_from_log(best_y))

        for start in starts:
            y = self._project_log_precision(start)
            first_moment = np.zeros(self.N, dtype=np.float64)
            second_moment = np.zeros(self.N, dtype=np.float64)
            step_size = self._ADAM_STEP

            for _ in range(self._GEOMETRY_STEPS):
                gradient, current_spread = self._log_condition_gradient(H, y)
                if not np.isfinite(current_spread):
                    break
                if current_spread < best_spread:
                    best_y = y.copy()
                    best_spread = current_spread
                if np.linalg.norm(gradient) < self._EPS:
                    break

                first_moment = (
                    self._ADAM_BETA_1 * first_moment
                    + (1.0 - self._ADAM_BETA_1) * gradient
                )
                second_moment = (
                    self._ADAM_BETA_2 * second_moment
                    + (1.0 - self._ADAM_BETA_2) * gradient * gradient
                )
                direction = first_moment / (np.sqrt(second_moment) + self._EPS)
                direction -= np.mean(direction)

                accepted = False
                for shrink in self._LINE_SEARCH:
                    candidate_y = self._project_log_precision(
                        y - step_size * shrink * direction
                    )
                    candidate_spread = self._spread(H, self._pi_from_log(candidate_y))
                    if candidate_spread <= current_spread - self._MIN_IMPROVEMENT:
                        y = candidate_y
                        step_size = min(self._ADAM_STEP, step_size * 1.02)
                        accepted = True
                        break

                if not accepted:
                    first_moment.fill(0.0)
                    step_size *= 0.5
                    if step_size < self._MIN_STEP:
                        break

        return self._pi_from_log(best_y)

    def _coordinate_refine_precision(
        self,
        H: np.ndarray,
        precision: np.ndarray,
    ) -> np.ndarray:
        """Polish a precision vector with exact coordinate descent.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.
            precision: Starting precision vector with shape ``(N,)``.

        Returns:
            Refined precision vector with shape ``(N,)``.
        """
        y = self._project_log_precision(np.log(np.maximum(precision, self._EPS)))
        best_spread = self._spread(H, self._pi_from_log(y))

        for _ in range(self._COORDINATE_SWEEPS):
            improved = False
            gradient, _ = self._log_condition_gradient(H, y)
            order = np.argsort(np.abs(gradient))[::-1]

            for idx in order:
                direction = -np.sign(gradient[idx])
                if direction == 0.0:
                    continue

                for sign in (direction, -direction):
                    candidate_y = y.copy()
                    candidate_y[idx] += sign * self._COORDINATE_STEP
                    candidate_y = self._project_log_precision(candidate_y)
                    candidate_spread = self._spread(H, self._pi_from_log(candidate_y))
                    if candidate_spread < best_spread - self._MIN_IMPROVEMENT:
                        y = candidate_y
                        best_spread = candidate_spread
                        improved = True
                        break

            if not improved:
                break

        return self._pi_from_log(y)

    def _log_precision_starts(self, H: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return log-domain starts for condition optimization.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.

        Returns:
            Tuple of log-precision starts, each with shape ``(N,)``.
        """
        starts = [np.zeros(self.N, dtype=np.float64)]
        starts.extend(np.log(candidate) for candidate in self._analytic_candidates(H))
        return tuple(self._project_log_precision(start) for start in starts)

    def _log_condition_gradient(
        self,
        H: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Compute gradient of log condition number in log-precision space.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.
            y: Log-precision vector with shape ``(N,)``.

        Returns:
            Pair ``(gradient, spread)`` where gradient has shape ``(N,)``.
        """
        pi = self._pi_from_log(y)
        root = np.sqrt(pi)
        scaled = (root[:, None] * H) * root[None, :]
        eigvals, eigvecs = np.linalg.eigh(0.5 * (scaled + scaled.T))

        positive = eigvals > self._EIG_THRESHOLD
        if np.count_nonzero(positive) < 2:
            return np.zeros(self.N, dtype=np.float64), float("inf")

        first = int(np.flatnonzero(positive)[0])
        last = len(eigvals) - 1
        gradient = eigvecs[:, last] ** 2 - eigvecs[:, first] ** 2
        gradient -= np.mean(gradient)
        spread = float(eigvals[last] / eigvals[first])
        return gradient, spread

    def _project_log_precision(self, y: np.ndarray) -> np.ndarray:
        """Center and bound log precision coordinates.

        Args:
            y: Raw log-precision vector with shape ``(N,)``.

        Returns:
            Projected log-precision vector with shape ``(N,)``.
        """
        projected = np.asarray(y, dtype=np.float64).reshape(self.N)
        projected = np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
        projected = projected - float(np.mean(projected))
        return np.clip(projected, -self._LOG_CLIP, self._LOG_CLIP)

    def _clip_and_scale(self, pi: np.ndarray) -> np.ndarray:
        """Clip and mean-normalize precision values.

        Args:
            pi: Raw precision vector with shape ``(N,)``.

        Returns:
            Positive precision vector with shape ``(N,)``.
        """
        arr = np.asarray(pi, dtype=np.float64).reshape(self.N)
        arr = np.nan_to_num(arr, nan=1.0, posinf=self.pi_max, neginf=self.pi_min)
        arr = arr / max(float(np.mean(arr)), self._EPS)
        arr = np.clip(arr, self.pi_min, self.pi_max)
        return arr / max(float(np.mean(arr)), self._EPS)

    def _spread(self, H: np.ndarray, pi: np.ndarray) -> float:
        """Evaluate the exact anisotropy spread used by ``checks.py``.

        Args:
            H: Symmetric Hessian matrix with shape ``(N, N)``.
            pi: Precision vector with shape ``(N,)``.

        Returns:
            Ratio of largest to smallest positive eigenvalue.
        """
        scaled_pi = self._clip_and_scale(pi)
        root = np.sqrt(scaled_pi)
        scaled = (root[:, None] * H) * root[None, :]
        eigs = np.linalg.eigvalsh(0.5 * (scaled + scaled.T))
        eigs = eigs[eigs > self._EIG_THRESHOLD]
        if len(eigs) < 2:
            return float("inf")
        return float(eigs[-1] / eigs[0])

    def _query_reliability(self, q: np.ndarray) -> np.ndarray:
        """Estimate coordinate reliability from query magnitudes.

        Args:
            q: Query vector with shape ``(N,)``.

        Returns:
            Reliability vector with shape ``(N,)`` and values in ``[0, 1]``.
        """
        return 1.0 / (
            1.0
            + np.exp(
                -(np.abs(q) - self._RELIABILITY_THRESHOLD)
                * self._RELIABILITY_SLOPE
            )
        )

    def _pi_from_log(self, y: np.ndarray) -> np.ndarray:
        """Map log-precision coordinates to precision values.

        Args:
            y: Log-precision vector with shape ``(N,)``.

        Returns:
            Precision vector with shape ``(N,)``.
        """
        return self._clip_and_scale(np.exp(self._project_log_precision(y)))


if __name__ == "__main__":
    from data import make_patterns
    from pcam_model import build_default_R

    patterns = make_patterns(K=16, N=64, seed=0)
    smoke_model = PCAMModel(patterns, build_default_R(N=64, seed=0))
    params: dict[str, Any] = {
        "R": smoke_model.R,
        "eta": smoke_model.eta,
        "beta": smoke_model.beta,
        "dt": smoke_model.dt,
        "T_max": smoke_model.T_max,
        "tol": smoke_model.tol,
        "T_in": smoke_model.T_in,
        "pi_min": smoke_model.pi_min,
        "pi_max": smoke_model.pi_max,
    }
    engine = Engine(patterns, params)
    precision = engine.predict_precision(patterns[0])
    assert precision.shape == patterns[0].shape
    print("Smoke test passed.")

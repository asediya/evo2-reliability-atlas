"""A self-contained tour of glmtrust on synthetic data (no external files needed).

    python examples/quickstart.py

It builds a score correlated with a binary label, then shows each layer in turn: calibration,
Mondrian conformal sets, selective prediction, and the one-object TrustLayer report.
"""
import numpy as np

from glmtrust import (MondrianConformal, SelectivePredictor, TrustLayer,
                      cross_conformal_calibrate, metrics)


def main():
    rng = np.random.default_rng(0)
    n = 6000
    y = (rng.random(n) < 0.25).astype(int)              # 25% deleterious
    score = rng.normal(y * 1.5, 1.0)                    # a genomic-LM-like score

    print("raw score AUROC: %.3f" % metrics.auroc(y, score))

    # 1. calibration ----------------------------------------------------------
    prob = cross_conformal_calibrate(score, y, method="isotonic")
    print("calibrated ECE : %.4f (raw sigmoid ECE %.4f)"
          % (metrics.ece(y, prob), metrics.ece(y, 1 / (1 + np.exp(-0.3 * score)))))

    # 2. Mondrian conformal ---------------------------------------------------
    cal, te = slice(0, n // 2), slice(n // 2, n)
    mc = MondrianConformal(alpha=0.1).fit(prob[cal], y[cal])
    rep = mc.evaluate(prob[te], y[te])
    print("conformal coverage (negative %.3f, positive %.3f) at target 0.90; abstain %.3f"
          % (rep["coverage_class_0"], rep["coverage_class_1"], rep["abstention_rate"]))

    # 3. selective prediction -------------------------------------------------
    sp = SelectivePredictor(coverage=0.85).fit(prob)
    srep = sp.evaluate(prob, y)
    print("selective: keep 85%% -> selective-error %.4f (full %.4f), capture %.3f, lift %.2f"
          % (srep["selective_error"], srep["full_error"], srep["capture"], srep["lift"]))
    print("  class-asymmetry: capture over-calls %.2f vs missed positives %.2f"
          % (srep["capture_false_positive"], srep["capture_false_negative"]))

    # 4. the whole layer as one object ---------------------------------------
    print("\n" + TrustLayer(alpha=0.1, coverage=0.85).summary(score, y))


if __name__ == "__main__":
    main()

"""
pp_tgnn — Detection-Bias-Corrected Temporal Event-Stream Pipeline
for Dynamic Plant-Pollinator Link Prediction.

Reference implementation accompanying:
  Ghosh, M. & Mukhopadhyay, A. "A Detection-Bias-Corrected Temporal
  Event-Stream Dataset for Dynamic Plant-Pollinator Link Prediction."

Public API:
    pp_tgnn.pipeline.run_pipeline   — Algorithms 1-8 end-to-end
    pp_tgnn.model.TGNNLite          — memory-based temporal GNN
    pp_tgnn.train.run_epoch         — train/eval loop for TGNNLite
    pp_tgnn.figures.make_figure1    — reproduce paper Figure 1
"""

__version__ = "0.1.0"

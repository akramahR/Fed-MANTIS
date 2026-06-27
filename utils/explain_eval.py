# utils/explain_eval.py
import json
import numpy as np
import torch
from utils.explain import evidence_packet

@torch.no_grad()
def explain_model(
    model,
    data_loader,
    device,
    explainer,
    idx_to_label=None,
    out_jsonl_path=None,
    topk=3,
):
    model.eval()
    model.to(device)

    results = []
    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).long()

        logits = model(X_batch)
        if isinstance(logits, tuple):  # just in case
            logits = logits[0]

        logits_np = logits.detach().cpu().numpy()
        y_np = y_batch.detach().cpu().numpy()

        packets = evidence_packet(logits_np, y_true=y_np, topk=topk)
        explanations = explainer.explain_batch(packets)

        # store
        pred = np.argmax(logits_np, axis=1)
        for i in range(len(pred)):
            row = {
                "y_true": int(y_np[i]),
                "y_pred": int(pred[i]),
                "explanation": explanations[i],
                "topk_idx": packets[i]["topk_idx"],
                "topk_prob": packets[i]["topk_prob"],
                "entropy": packets[i]["entropy"],
                "margin": packets[i]["margin"],
            }
            results.append(row)

    if out_jsonl_path:
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    return results

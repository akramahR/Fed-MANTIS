# utils/explain.py
import json
import numpy as np
import torch

def softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def evidence_packet(
    logits: np.ndarray,
    y_true: np.ndarray | None = None,
    topk: int = 3,
    x_stats: dict | None = None,
):
    """
    Build a compact 'evidence packet' you can feed to an LLM.
    logits: [B, K]
    y_true: [B] or None
    """
    probs = softmax_np(logits)
    B, K = probs.shape

    packets = []
    for i in range(B):
        p = probs[i]
        top_idx = np.argsort(-p)[:topk]
        pkt = {
            "topk_idx": top_idx.tolist(),
            "topk_prob": [float(p[j]) for j in top_idx],
            "entropy": float(-np.sum(p * np.log(p + 1e-12))),
            "margin": float(p[top_idx[0]] - p[top_idx[1]]) if topk >= 2 else float(p[top_idx[0]]),
        }
        if y_true is not None:
            pkt["y_true"] = int(y_true[i])
        if x_stats is not None:
            # optional per-sample stats you compute externally
            pkt["x_stats"] = x_stats[i] if isinstance(x_stats, list) else x_stats
        packets.append(pkt)

    return packets


class TemplateExplainer:
    """
    Always-available explainer. No LLM required.
    Produces consistent, paper-friendly explanations from evidence packet.
    """
    def __init__(self, idx_to_label=None):
        self.idx_to_label = idx_to_label or {}

    def explain_one(self, pkt: dict):
        topk = list(zip(pkt["topk_idx"], pkt["topk_prob"]))
        pred_idx = topk[0][0]
        pred_name = self.idx_to_label.get(pred_idx, f"class_{pred_idx}")

        entropy = pkt["entropy"]
        margin = pkt["margin"]

        confidence = "high" if (margin > 0.40 and entropy < 1.0) else ("medium" if margin > 0.20 else "low")
        msg = (
            f"Prediction: {pred_name}. "
            f"Confidence: {confidence} (margin={margin:.3f}, entropy={entropy:.3f}). "
            f"Top-{len(topk)}: " +
            ", ".join([f"{self.idx_to_label.get(i, f'class_{i}')}={p:.2f}" for i, p in topk])
        )

        # Optional: if you pass x_stats (sensor quality etc.) you can add a line:
        if "x_stats" in pkt:
            msg += f". Signal stats: {pkt['x_stats']}"

        return msg

    def explain_batch(self, packets: list[dict]):
        return [self.explain_one(pkt) for pkt in packets]


class OpenAIExplainer:
    """
    Uses OpenAI API (optional). Requires:
      pip install openai
      export OPENAI_API_KEY=...
    """
    def __init__(self, model="gpt-4o-mini", idx_to_label=None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.idx_to_label = idx_to_label or {}

    def _prompt(self, pkt: dict):
        # convert indices to names for readability
        topk_named = []
        for i, p in zip(pkt["topk_idx"], pkt["topk_prob"]):
            topk_named.append([self.idx_to_label.get(i, f"class_{i}"), p])

        return (
            "You are explaining a Human Activity Recognition (HAR) model prediction.\n"
            "Given this evidence, write 1-2 sentences explaining the prediction and uncertainty.\n"
            "Be concise and avoid speculation.\n\n"
            f"EVIDENCE:\n{json.dumps({**pkt, 'topk_named': topk_named}, indent=2)}\n\n"
            "OUTPUT FORMAT:\n"
            "Prediction: <label>\n"
            "Explanation: <short explanation>\n"
        )

    def explain_batch(self, packets: list[dict]):
        explanations = []
        for pkt in packets:
            prompt = self._prompt(pkt)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You write short, technical model explanations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            explanations.append(resp.choices[0].message.content.strip())
        return explanations


class LocalHFExplainer:
    def __init__(self, hf_model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 idx_to_label=None, device=None,
                 max_new_tokens=80):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        self.idx_to_label = idx_to_label or {}
        self.max_new_tokens = max_new_tokens

    def _messages(self, pkt: dict):
        topk_named = []
        for i, p in zip(pkt["topk_idx"], pkt["topk_prob"]):
            topk_named.append([self.idx_to_label.get(i, f"class_{i}"), float(p)])

        evidence = {**pkt, "topk_named": topk_named}

        system = (
            "You are a concise technical assistant. "
            "Explain HAR predictions using only the provided evidence. "
            "If uncertainty is high (low margin / high entropy), say so."
        )
        user = (
            "Given this evidence from a HAR classifier, produce:\n"
            "Prediction: <label>\n"
            "Explanation: <1-2 sentences>\n\n"
            f"EVIDENCE:\n{json.dumps(evidence, indent=2)}"
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    @torch.no_grad()
    def explain_batch(self, packets):
        outs = []
        for pkt in packets:
            messages = self._messages(pkt)

            # Use chat template if available (recommended for instruct models)
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # fallback
                prompt = messages[-1]["content"]

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            text = self.tokenizer.decode(gen[0], skip_special_tokens=True)

            # simple heuristic: return tail after the prompt
            outs.append(text[len(prompt):].strip() if text.startswith(prompt) else text.strip())

        return outs


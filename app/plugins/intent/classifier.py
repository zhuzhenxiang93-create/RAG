"""Lazy LoRA classifier with a deterministic Lite fallback."""

from importlib.util import find_spec
import threading
from typing import List, Sequence, Tuple

from app.core.config import Settings
from app.plugins.intent.assets import safe_label_count, validate_assets
from app.schemas.intent import (
    IntentClassificationResponse,
    IntentPluginStatus,
    IntentPrediction,
    KnowledgeRoute,
)


# The Lite taxonomy is an engineering fallback, not a trained benchmark model.
_LITE_RULES: Sequence[Tuple[str, str, Sequence[str]]] = (
    ("refund_request", "commerce", ("退款", "退钱", "refund", "chargeback")),
    ("order_status", "commerce", ("订单", "物流", "配送", "外卖", "快递", "order", "delivery")),
    ("product_support", "support", ("故障", "报错", "不能用", "连不上", "维修", "error", "broken")),
    ("account_management", "account", ("账户", "账号", "登录", "密码", "注册", "account", "password")),
    ("payment_query", "finance", ("支付", "账单", "扣款", "发票", "付款", "payment", "invoice")),
    ("policy_query", "hr", ("报销", "请假", "入职", "薪资", "差旅", "制度", "expense", "leave")),
    ("calendar_query", "productivity", ("日历", "会议", "提醒", "闹钟", "calendar", "meeting", "alarm")),
    ("weather_query", "weather", ("天气", "温度", "下雨", "weather", "temperature")),
    ("transport_query", "transport", ("公交", "地铁", "打车", "路线", "火车", "航班", "traffic", "train")),
    ("communication", "communication", ("邮件", "短信", "电话", "联系人", "email", "message", "call")),
    ("media_control", "media", ("音乐", "播放", "暂停", "音量", "music", "play", "volume")),
    ("document_search", "general_knowledge", ("文档", "资料", "规定", "手册", "document", "policy")),
)


def route_for(domain: str, intent: str, probability: float, threshold: float) -> KnowledgeRoute:
    knowledge_base = {
        "commerce": "commerce_support",
        "support": "product_support",
        "account": "account_help",
        "finance": "finance_policy",
        "hr": "employee_handbook",
        "productivity": "productivity_help",
        "weather": "external_service",
        "transport": "external_service",
        "communication": "productivity_help",
        "media": "productivity_help",
        "general_knowledge": "general_documents",
    }.get(domain, domain or "general_documents")
    exact_markers = ("status", "account", "payment", "order", "calendar")
    profile = "exact" if any(marker in intent for marker in exact_markers) else "balanced"
    uncertain = probability < threshold
    if uncertain:
        knowledge_base = "general_documents"
        profile = "broad"
    return KnowledgeRoute(
        knowledge_base=knowledge_base,
        retrieval_profile=profile,
        allow_workflow=not uncertain and domain in {"commerce", "account", "productivity"},
        abstain=uncertain,
        reason=(
            "Confidence is below the routing threshold; search all indexed documents."
            if uncertain
            else "The predicted domain selects a knowledge base and retrieval profile."
        ),
    )


class IntentClassifier:
    """Classify a query using Lite rules or a MASSIVE-trained LoRA adapter."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._assets = None
        self._device = None

    def status(self) -> IntentPluginStatus:
        dependencies = {
            name: find_spec(name) is not None
            for name in ("torch", "transformers", "peft")
        }
        configured = {
            "base_model": bool(self.settings.intent_base_model),
            "adapter": bool(self.settings.intent_adapter_path),
            "taxonomy": bool(self.settings.intent_labels_path),
        }
        if not self.settings.intent_enabled:
            state = "disabled"
            message = "Intent routing is disabled."
        elif self.settings.intent_backend == "lite":
            state = "lite_ready"
            message = "Deterministic Lite intent routing is ready."
        elif not all(configured.values()):
            state = "not_configured"
            message = "LoRA intent configuration is incomplete."
        elif not all(dependencies.values()):
            state = "dependencies_missing"
            message = "Install torch, transformers and peft for LoRA inference."
        elif self._model is not None:
            state = "loaded"
            message = "LoRA intent classifier is loaded."
        else:
            try:
                validate_assets(
                    base_model=self.settings.intent_base_model,
                    adapter_path=self.settings.intent_adapter_path,
                    labels_path=self.settings.intent_labels_path,
                )
                state = "ready"
                message = "Intent artifacts are valid and will load on first inference."
            except ValueError as exc:
                state = "not_configured"
                message = str(exc)
        return IntentPluginStatus(
            enabled=self.settings.intent_enabled,
            backend=self.settings.intent_backend,
            state=state,
            dependencies=dependencies,
            configured_assets=configured,
            label_count=(
                len(_LITE_RULES) + 1
                if self.settings.intent_backend == "lite"
                else safe_label_count(self.settings.intent_labels_path)
            ),
            device=self._device or ("cpu" if self.settings.intent_backend == "lite" else None),
            message=message,
        )

    def predict(self, text: str, top_k: int = 3) -> IntentClassificationResponse:
        status = self.status()
        if status.state == "disabled":
            raise RuntimeError(status.message)
        if self.settings.intent_backend == "lite":
            return self._predict_lite(text, top_k)
        self._ensure_loaded()
        return self._predict_lora(text, top_k)

    def _predict_lite(self, text: str, top_k: int) -> IntentClassificationResponse:
        lowered = text.lower()
        matches = []
        for label_id, (intent, domain, keywords) in enumerate(_LITE_RULES):
            hits = sum(keyword.lower() in lowered for keyword in keywords)
            if hits:
                matches.append((hits, label_id, intent, domain))
        matches.sort(key=lambda item: (-item[0], item[1]))
        predictions: List[IntentPrediction] = []
        if matches:
            selected = matches[:top_k]
            secondary_total = sum(item[0] for item in selected[1:])
            for rank, (hits, label_id, intent, domain) in enumerate(selected):
                if rank == 0:
                    probability = min(0.94, 0.68 + 0.1 * (hits - 1))
                else:
                    probability = 0.24 * hits / max(secondary_total, 1)
                predictions.append(
                    IntentPrediction(
                        intent=intent,
                        domain=domain,
                        label_id=label_id,
                        probability=round(probability, 6),
                    )
                )
        else:
            predictions.append(
                IntentPrediction(
                    intent="general_question",
                    domain="general_knowledge",
                    label_id=len(_LITE_RULES),
                    probability=0.36,
                )
            )
        top = predictions[0]
        return IntentClassificationResponse(
            predictions=predictions,
            route=route_for(
                top.domain,
                top.intent,
                top.probability,
                self.settings.intent_confidence_threshold,
            ),
            backend="lite",
            model="keyword-baseline-v1",
            device="cpu",
            calibrated=False,
        )

    def _predict_lora(self, text: str, top_k: int) -> IntentClassificationResponse:
        import torch

        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=self.settings.intent_max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(self._device) for name, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(**encoded).logits[0].float(), dim=-1)
            count = min(top_k, len(self._assets.intents))
            values, indices = torch.topk(probabilities, k=count)
        predictions = []
        for value, index in zip(
            values.detach().cpu().tolist(), indices.detach().cpu().tolist()
        ):
            label_id = int(index)
            intent = self._assets.intents_by_id[label_id]
            predictions.append(
                IntentPrediction(
                    intent=intent,
                    domain=self._assets.intent_to_domain[intent],
                    label_id=label_id,
                    probability=round(float(value), 6),
                )
            )
        top = predictions[0]
        return IntentClassificationResponse(
            predictions=predictions,
            route=route_for(
                top.domain,
                top.intent,
                top.probability,
                self.settings.intent_confidence_threshold,
            ),
            backend="lora",
            model=self._assets.base_model,
            adapter=self._assets.adapter_path.name,
            device=str(self._device),
            calibrated=False,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            status = self.status()
            if status.state not in {"ready", "loaded"}:
                raise RuntimeError(status.message)
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            assets = validate_assets(
                base_model=self.settings.intent_base_model,
                adapter_path=self.settings.intent_adapter_path,
                labels_path=self.settings.intent_labels_path,
            )
            device = self._select_device(torch)
            tokenizer = AutoTokenizer.from_pretrained(
                assets.base_model, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForSequenceClassification.from_pretrained(
                assets.base_model,
                num_labels=len(assets.intents),
                trust_remote_code=True,
            )
            base_model.config.pad_token_id = tokenizer.pad_token_id
            model = PeftModel.from_pretrained(base_model, str(assets.adapter_path))
            self._assets = assets
            self._tokenizer = tokenizer
            self._model = model.to(device).eval()
            self._device = device

    def _select_device(self, torch):
        requested = self.settings.intent_device.lower()
        if requested == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(requested)

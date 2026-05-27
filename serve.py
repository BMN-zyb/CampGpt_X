# serve.py

"""
CampGPT 推理服务
支持：命令行交互 / Flask API / 单次问答
"""

import os
import json
import time
from typing import List, Dict

import torch

os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
import tiktoken

from config import GPTConfig
from model import GPT, KVCache


# =============================================================================
# ========================= 模型服务 ==========================================
# =============================================================================

class CampGPTServer:
    """CampGPT 推理服务"""

    def __init__(
        self,
        model_dir: str = "campgpt-student-handbook",
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.dtype = dtype
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

        self._load_model(model_dir)
        self._load_chat_template(model_dir)

        self.enc = tiktoken.get_encoding("gpt2")
        self.conversation_history: List[Dict] = []

        print(f"\n[Server] CampGPT ready on {self.device}")


    def _load_model(self, model_dir: str):
        """加载模型配置和权重"""
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        model_config = GPTConfig(
            vocab_size=config_dict["vocab_size"],
            n_embd=config_dict["n_embd"],
            n_head=config_dict["n_head"],
            n_kv_head=config_dict["n_kv_head"],
            n_layer=config_dict["n_layer"],
            block_size=config_dict["block_size"],
            norm_eps=config_dict.get("norm_eps", 1e-6),
            multiple_of=config_dict.get("multiple_of", 64),
            use_moe=config_dict.get("use_moe", False),
            n_experts=config_dict.get("n_experts", 0),
            n_experts_per_tok=config_dict.get("n_experts_per_tok", 0),
            n_shared_experts=config_dict.get("n_shared_experts", 0),
        )
        self.model_config = model_config

        # 加载权重
        sf_path = os.path.join(model_dir, "model.safetensors")
        weight_path = os.path.join(model_dir, "pytorch_model.bin")
        shared_path = os.path.join(model_dir, "shared_weights.json")

        if os.path.exists(sf_path):
            from safetensors.torch import load_file
            state_dict = load_file(sf_path)

            # 恢复共享权重
            if os.path.exists(shared_path):
                with open(shared_path) as f:
                    shared_keys = json.load(f)
                for missing_key, source_key in shared_keys.items():
                    state_dict[missing_key] = state_dict[source_key]
                print(f"[Server] Restored shared weights: {list(shared_keys.keys())}")

        elif os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No weights found in {model_dir}")

        self.model = GPT(model_config)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.model.set_gradient_checkpointing(False)

        total = sum(p.numel() for p in self.model.parameters())
        print(f"[Server] Model loaded: {total:,} params ({total/1e6:.1f}M)")

    
    # -------------------------------------------------------------------------
    def _load_chat_template(self, model_dir: str):
        """加载对话模板"""
        template_path = os.path.join(model_dir, "chat_template.json")
        if os.path.exists(template_path):
            with open(template_path) as f:
                self.chat_template = json.load(f)
        else:
            self.chat_template = {
                "system_prompt": "You are a helpful university assistant.",
                "user_prefix": "\n\n### User:\n",
                "assistant_prefix": "\n\n### Assistant:\n",
                "turn_end": "\n\n",
            }

        self.system_prompt = self.chat_template.get("system_prompt", "")

    # -------------------------------------------------------------------------
    def _build_prompt(self, messages: List[Dict]) -> str:
        """构建完整 prompt 文本"""
        text = ""
        if self.system_prompt:
            text += f"### System:\n{self.system_prompt}\n\n"

        for msg in messages:
            if msg["role"] == "user":
                text += f"### User:\n{msg['content']}\n\n"
            elif msg["role"] == "assistant":
                text += f"### Assistant:\n{msg['content']}\n\n"

        text += "### Assistant:\n"
        return text

    # -------------------------------------------------------------------------
    def chat(
        self,
        user_message: str,
        temperature: float = 0.1,
        top_k: int = 50,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        use_history: bool = True,
    ) -> str:
        """单轮/多轮对话"""

        # 添加到历史
        if use_history:
            self.conversation_history.append(
                {"role": "user", "content": user_message}
            )
            messages = self.conversation_history
        else:
            messages = [{"role": "user", "content": user_message}]

        # 构建 prompt
        prompt_text = self._build_prompt(messages)
        prompt_ids = self.enc.encode(prompt_text)

        # 检查长度，必要时截断历史
        max_total = self.model_config.block_size
        if len(prompt_ids) + max_new_tokens > max_total:
            max_new_tokens = max_total - len(prompt_ids) - 10
            if max_new_tokens <= 0:
                if use_history and len(self.conversation_history) > 2:
                    self.conversation_history = self.conversation_history[-2:]
                    return self.chat(
                        user_message, temperature, top_k, top_p, 256, use_history
                    )
                return "[Error] Context too long."

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # 生成
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=self.dtype):
                generated = self.model.generate(
                    prompt_t,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

        full_output = self.enc.decode(generated[0].tolist())
        response = self._extract_response(full_output, prompt_text)

        if use_history:
            self.conversation_history.append(
                {"role": "assistant", "content": response}
            )

        return response

    # -------------------------------------------------------------------------
    def _extract_response(self, full_text: str, prompt_text: str) -> str:
        """从生成文本中提取 assistant 回复"""
        if full_text.startswith(prompt_text):
            response = full_text[len(prompt_text):]
        else:
            parts = full_text.split("### Assistant:\n")
            response = parts[-1] if len(parts) > 1 else full_text

        # 截断到停止标记
        for marker in ["### User:", "### System:", "### Assistant:"]:
            if marker in response:
                response = response[: response.index(marker)]

        return response.strip()

    # -------------------------------------------------------------------------
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("[Server] Conversation history cleared.")

    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_history.copy()


# =============================================================================
# ========================= 命令行交互 =========================================
# =============================================================================

def interactive_cli(server: CampGPTServer):
    """命令行交互式对话"""
    print(f"\n{'='*60}")
    print(f"  CampGPT Student Handbook Assistant")
    print(f"  Commands:")
    print(f"    /clear   - Clear conversation history")
    print(f"    /history - Show conversation history")
    print(f"    /quit    - Exit")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            server.clear_history()
            continue
        elif user_input.lower() == "/history":
            history = server.get_history()
            if not history:
                print("  (empty)")
            for msg in history:
                role = msg["role"].capitalize()
                print(f"  [{role}]: {msg['content'][:100]}...")
            continue

        t0 = time.time()
        response = server.chat(user_input)
        t1 = time.time()

        print(f"\nAssistant: {response}")
        print(f"  ({t1-t0:.2f}s)\n")


# =============================================================================
# ========================= Flask API ==========================================
# =============================================================================

def create_api(server: CampGPTServer):
    """创建 Flask API 服务"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("[Error] Flask not installed. Run: pip install flask")
        return None

    app = Flask(__name__)

    @app.route("/v1/chat", methods=["POST"])
    def chat_endpoint():
        data = request.json
        message = data.get("message", "")
        temperature = data.get("temperature", 0.1)
        max_tokens = data.get("max_tokens", 256)
        use_history = data.get("use_history", False)

        if not message:
            return jsonify({"error": "message is required"}), 400

        response = server.chat(
            message,
            temperature=temperature,
            max_new_tokens=max_tokens,
            use_history=use_history,
        )

        return jsonify({
            "response": response,
            "model": "campgpt-student-handbook",
        })

    @app.route("/v1/clear", methods=["POST"])
    def clear_endpoint():
        server.clear_history()
        return jsonify({"status": "ok"})

    @app.route("/health", methods=["GET"])
    def health_endpoint():
        return jsonify({"status": "healthy", "device": server.device})

    return app


# =============================================================================
# ========================= 入口 ===============================================
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CampGPT Server")
    parser.add_argument(
        "--model_dir", type=str, default="campgpt-student-handbook",
        help="Model directory path",
    )
    parser.add_argument(
        "--mode", type=str, default="cli",
        choices=["cli", "api", "single"],
        help="cli=interactive, api=Flask server, single=one query",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query (for mode=single)",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="API port (for mode=api)",
    )
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    server = CampGPTServer(model_dir=args.model_dir, device=args.device)

    if args.mode == "cli":
        interactive_cli(server)

    elif args.mode == "single":
        query = args.query or "What is a 'Referral Notice' in the context of student discipline?"
        print(f"\nQ: {query}")
        response = server.chat(query, use_history=False)
        print(f"A: {response}")

    elif args.mode == "api":
        app = create_api(server)
        if app:
            print(f"\n[API] Starting on port {args.port}")
            print(f"[API] POST http://localhost:{args.port}/v1/chat")
            app.run(host="0.0.0.0", port=args.port, debug=False)
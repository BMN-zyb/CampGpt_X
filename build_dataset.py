

# 基于英文学生手册PDF构建SFT和DPO数据集

## 完整代码

# build_dataset.py

import os
import json
import time
import random
import re
from typing import List, Dict
from pathlib import Path

import pdfplumber
from openai import OpenAI


# =============================================================================
# ========================= 配置 ===============================================
# =============================================================================

class DatasetConfig:
    """数据集构建配置"""
    # PDF路径
    pdf_path = "student-code-of-conduct.pdf"
    
    # 输出路径
    sft_output_path = "sft_dataset.json"      # 最终SFT数据集
    dpo_output_path = "dpo_dataset.json"      # 最终DPO数据集
    chunks_debug_path = "chunks_debug.json"   # 调试用，保存分块结果
    
    # 文本分块
    chunk_size = 5000          # 英文文本每个chunk的最大字符数（英文比中文字符多）
    chunk_overlap = 500        # chunk之间的重叠字符数
    
    # 生成数量控制
    sft_qa_per_chunk = 5       # 每个chunk生成的单轮QA数量
    sft_multi_per_chunk = 2    # 每个chunk生成的多轮对话数量
    sft_scene_per_chunk = 2    # 每个chunk生成的场景化问答数量
    dpo_candidates = 2         # DPO阶段每个问题生成的候选答案数
    
    # API配置 (Qwen3-Max)
    api_key = "sk-"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name = "qwen3-max"
    
    # 请求控制
    max_retries = 3
    retry_delay = 3
    request_delay = 1.0        # qwen-max 限流更严，间隔大一些
    temperature = 0.8
    
    # 语言设置
    language = "en"            # 输出语言: "en" 英文, "zh" 中文


# =============================================================================
# ========================= Step 1: PDF解析 ====================================
# =============================================================================

class PDFParser:
    """从英文PDF中提取文本并分块"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def extract_text(self, pdf_path: str) -> str:
        """提取PDF全部文本"""
        print(f"[PDF] Parsing: {pdf_path}")
        full_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
                if (i + 1) % 10 == 0:
                    print(f"  Parsed {i+1}/{total_pages} pages")
        
        print(f"[PDF] Done: {total_pages} pages, {len(full_text)} characters")
        return full_text
    
    def clean_text(self, text: str) -> str:
        """清洗英文文本"""
        # 去除多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 去除页码模式
        text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'- \d+ -', '', text)
        # 去除多余空格（保留换行）
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # 修复断行连词（英文PDF常见的行末断词）
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        return text.strip()
    
    def split_into_chunks(self, text: str) -> List[Dict]:
        """
        智能分块：先按章节标题分割，再对过长章节进一步分块
        """
        sections = self._split_by_sections(text)
        
        chunks = []
        chunk_id = 0
        
        for section in sections:
            title = section.get("title", "")
            content = section["content"]
            
            if len(content.strip()) < 30:
                continue
            
            if len(content) <= self.config.chunk_size:
                chunks.append({
                    "chunk_id": chunk_id,
                    "title": title,
                    "content": content
                })
                chunk_id += 1
            else:
                sub_chunks = self._split_long_content(content, title)
                for sc in sub_chunks:
                    sc["chunk_id"] = chunk_id
                    chunks.append(sc)
                    chunk_id += 1
        
        print(f"[PDF] Chunking done: {len(chunks)} chunks")
        return chunks
    
    def _split_by_sections(self, text: str) -> List[Dict]:
        """按英文章节标题分割"""
        # 匹配英文学生手册常见标题格式
        patterns = [
            # "Chapter 1: Academic Policies" / "CHAPTER 1 ACADEMIC POLICIES"
            r'^((?:CHAPTER|Chapter)\s+\d+[:\s].*)$',
            # "Section 1.1: ..." / "SECTION 1.1 ..."
            r'^((?:SECTION|Section)\s+\d+[\.\d]*[:\s].*)$',
            # "ARTICLE I - ..." / "Article 1: ..."
            r'^((?:ARTICLE|Article)\s+[IVX\d]+[:\s\-].*)$',
            # "1. Academic Standards" / "1.1 Grading System" (数字开头，后跟大写)
            r'^(\d+(?:\.\d+)*[.\s]+[A-Z][A-Za-z\s]{3,})$',
            # "ACADEMIC POLICIES" (全大写标题, 长度限制避免误匹配)
            r'^([A-Z][A-Z\s]{4,40})$',
            # "Part I: ..." / "Part 1: ..."
            r'^((?:PART|Part)\s+[IVX\d]+[:\s\-].*)$',
        ]
        
        sections = []
        current_title = "Introduction"
        current_content = []
        
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                current_content.append("")
                continue
            
            is_title = False
            for pattern in patterns:
                if re.match(pattern, stripped) and len(stripped) < 80:
                    is_title = True
                    break
            
            if is_title:
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append({
                            "title": current_title,
                            "content": content
                        })
                current_title = stripped
                current_content = []
            else:
                current_content.append(stripped)
        
        # 最后一个章节
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append({
                    "title": current_title,
                    "content": content
                })
        
        if not sections:
            sections = [{"title": "Student Handbook", "content": text}]
        
        print(f"  Found {len(sections)} sections")
        for s in sections[:]:
            print(f"    - {s['title'][:60]} ({len(s['content'])} chars)")
        # if len(sections) > 10:
        #     print(f"    ... and {len(sections)-10} more")
        
        return sections
    
    def _split_long_content(self, content: str, title: str) -> List[Dict]:
        """将过长内容按段落分块"""
        paragraphs = re.split(r'\n\s*\n', content)  # 按空行分段
        chunks = []
        current_parts = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_len = len(para)
            
            if current_len + para_len > self.config.chunk_size and current_parts:
                chunk_text = '\n\n'.join(current_parts)
                chunks.append({"title": title, "content": chunk_text})
                
                # 保留最后一段作为重叠
                overlap = current_parts[-1] if current_parts else ""
                current_parts = [overlap] if len(overlap) <= self.config.chunk_overlap else []
                current_len = len(overlap) if current_parts else 0
            
            current_parts.append(para)
            current_len += para_len
        
        if current_parts:
            chunk_text = '\n\n'.join(current_parts)
            if chunk_text.strip():
                chunks.append({"title": title, "content": chunk_text})
        
        return chunks


# =============================================================================
# ========================= Step 2: Qwen API 封装 ==============================
# =============================================================================

class QwenClient:
    """Qwen3-Max API 调用封装"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.total_calls = 0
        self.failed_calls = 0
    
    def call(self, system_prompt: str, user_prompt: str,
             temperature: float = None) -> str:
        """调用Qwen3-Max API"""
        temp = temperature or self.config.temperature
        
        for attempt in range(self.config.max_retries):
            try:
                self.total_calls += 1
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                    max_tokens=4096,
                    # qwen3 支持 enable_thinking，这里关闭以获取纯输出
                    extra_body={"enable_thinking": False},
                )
                result = response.choices[0].message.content.strip()
                time.sleep(self.config.request_delay)
                return result
                
            except Exception as e:
                self.failed_calls += 1
                print(f"  [API] Attempt {attempt+1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    wait = self.config.retry_delay * (attempt + 1)
                    print(f"  [API] Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [API] Max retries reached, skipping")
                    return ""
    
    def parse_json_response(self, response: str) -> list:
        """从API返回中解析JSON（鲁棒处理各种格式）"""
        if not response:
            return []
        
        # 1. 直接解析
        try:
            data = json.loads(response)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            pass
        
        # 2. 提取 ```json ... ```
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                pass
        
        # 3. 找 [ ... ] 数组
        array_match = re.search(r'\[[\s\S]*\]', response)
        if array_match:
            try:
                data = json.loads(array_match.group())
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                pass
        
        # 4. 逐个找 { ... }
        results = []
        for m in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response):
            try:
                obj = json.loads(m.group())
                results.append(obj)
            except json.JSONDecodeError:
                continue
        
        if not results:
            print(f"  [WARN] Failed to parse JSON, response preview: {response[:200]}...")
        
        return results
    
    def print_stats(self):
        print(f"[API] Total calls: {self.total_calls}, Failed: {self.failed_calls}")


# =============================================================================
# ========================= Step 3: SFT 数据生成 ===============================
# =============================================================================

class SFTDataGenerator:
    """SFT数据集生成器（英文学生手册）"""
    
    def __init__(self, config: DatasetConfig, client: QwenClient):
        self.config = config
        self.client = client
    
    def generate_all(self, chunks: List[Dict]) -> List[Dict]:
        """对所有chunks生成SFT数据"""
        all_data = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            print(f"\n[SFT] Processing chunk {i+1}/{total}: "
                  f"{chunk['title'][:50]}... ({len(chunk['content'])} chars)")
            
            content = chunk["content"]
            if len(content.strip()) < 50:
                print(f"  Skipped (too short)")
                continue
            
            # 1. 单轮问答
            qa_data = self._generate_single_qa(content, chunk["title"])
            all_data.extend(qa_data)
            print(f"  Single-turn QA: {len(qa_data)} items")
            
            # 2. 多轮对话
            multi_data = self._generate_multi_turn(content, chunk["title"])
            all_data.extend(multi_data)
            print(f"  Multi-turn dialogue: {len(multi_data)} items")
            
            # 3. 场景化问答
            scene_data = self._generate_scene_qa(content, chunk["title"])
            all_data.extend(scene_data)
            print(f"  Scenario QA: {len(scene_data)} items")
            
            # 定期保存（防止中途失败丢失数据）
            if (i + 1) % 5 == 0:
                self._save_checkpoint(all_data, "sft_checkpoint.json")
        
        print(f"\n[SFT] Total generated: {len(all_data)} items")
        return all_data
    
    def _save_checkpoint(self, data: list, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [Checkpoint] Saved {len(data)} items to {path}")
    
    # -------------------------------------------------------------------------
    def _generate_single_qa(self, context: str, section_title: str) -> List[Dict]:
        """生成单轮英文问答"""
        system_prompt = """You are an expert at creating high-quality training data for educational AI assistants.
You will be given content from a university student handbook. Generate question-answer pairs that:

1. Questions should be natural - the way real students would actually ask
2. Include diverse question styles: direct, indirect, colloquial
3. Answers must be accurate, complete, and based ONLY on the provided content
4. Answers should be well-structured (use bullet points or numbered lists when appropriate)
5. Cover different question types: factual, procedural, conditional, comparative
6. Do NOT fabricate information not present in the source text"""

        user_prompt = f"""Based on the following student handbook content (Section: {section_title}), 
generate exactly {self.config.sft_qa_per_chunk} high-quality single-turn Q&A pairs.

【Student Handbook Content】
{context}

Output STRICTLY as a JSON array, no extra text:
[
  {{
    "messages": [
      {{"role": "user", "content": "Student's question in English"}},
      {{"role": "assistant", "content": "Accurate answer in English"}}
    ]
  }}
]"""
        
        response = self.client.call(system_prompt, user_prompt)
        results = self.client.parse_json_response(response)
        
        valid = []
        for item in results:
            if self._validate_sft_format(item):
                item["type"] = "single_qa"
                item["source_section"] = section_title
                valid.append(item)
        
        return valid
    
    # -------------------------------------------------------------------------
    def _generate_multi_turn(self, context: str, section_title: str) -> List[Dict]:
        """生成多轮英文对话"""
        system_prompt = """You are an expert at creating multi-turn dialogue training data.
Generate realistic conversations between a student and an AI assistant about university policies.

Requirements:
1. Conversations should flow naturally - students ask follow-up questions, seek clarification
2. The assistant should be helpful, accurate, and proactive in sharing relevant info
3. Each conversation should have 2-4 exchanges (4-8 messages total)
4. Show contextual awareness - later messages reference earlier parts of the conversation
5. All information must come from the provided handbook content"""

        user_prompt = f"""Based on the following student handbook content (Section: {section_title}), 
generate {self.config.sft_multi_per_chunk} multi-turn conversations.

【Student Handbook Content】
{context}

Output STRICTLY as a JSON array:
[
  {{
    "messages": [
      {{"role": "user", "content": "Student's initial question"}},
      {{"role": "assistant", "content": "Assistant's response"}},
      {{"role": "user", "content": "Student's follow-up"}},
      {{"role": "assistant", "content": "Assistant's further response"}}
    ]
  }}
]"""
        
        response = self.client.call(system_prompt, user_prompt)
        results = self.client.parse_json_response(response)
        
        valid = []
        for item in results:
            if self._validate_sft_format(item) and len(item.get("messages", [])) >= 4:
                item["type"] = "multi_turn"
                item["source_section"] = section_title
                valid.append(item)
        
        return valid
    
    # -------------------------------------------------------------------------
    def _generate_scene_qa(self, context: str, section_title: str) -> List[Dict]:
        """生成场景化英文问答"""
        system_prompt = """You are an expert at creating scenario-based training data.
Generate Q&A pairs where the student provides context about their specific situation.

Requirements:
1. Each question should include a realistic scenario/background
2. Scenarios should vary: freshman orientation, academic trouble, graduation prep, etc.
3. Answers should be tailored to the specific scenario
4. Natural, conversational tone"""

        user_prompt = f"""Based on the following student handbook content (Section: {section_title}), 
generate {self.config.sft_scene_per_chunk} scenario-based Q&A pairs.

Example scenarios:
- "I'm a freshman and just started this semester..."
- "I failed a course last semester and I'm worried about..."
- "I'm planning to study abroad next year and need to know..."
- "I'm in my final year and want to make sure I meet all graduation requirements..."
- "I got a disciplinary notice and I'm not sure what to do..."

【Student Handbook Content】
{context}

Output STRICTLY as a JSON array:
[
  {{
    "messages": [
      {{"role": "user", "content": "(Question with scenario context)"}},
      {{"role": "assistant", "content": "(Tailored response)"}}
    ]
  }}
]"""
        
        response = self.client.call(system_prompt, user_prompt)
        results = self.client.parse_json_response(response)
        
        valid = []
        for item in results:
            if self._validate_sft_format(item):
                item["type"] = "scene_qa"
                item["source_section"] = section_title
                valid.append(item)
        
        return valid
    
    # -------------------------------------------------------------------------
    def _validate_sft_format(self, item: dict) -> bool:
        """验证SFT数据格式"""
        if "messages" not in item:
            return False
        msgs = item["messages"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            return False
        
        # 检查角色交替
        for i, msg in enumerate(msgs):
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ("user", "assistant", "system"):
                return False
            if not msg["content"].strip():
                return False
        
        # 第一条必须是user
        if msgs[0]["role"] != "user":
            return False
        
        return True


# =============================================================================
# ========================= Step 4: DPO 数据生成 ===============================
# =============================================================================

class DPODataGenerator:
    """DPO偏好数据集生成器"""
    
    def __init__(self, config: DatasetConfig, client: QwenClient):
        self.config = config
        self.client = client
    
    def generate_from_sft(self, sft_data: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """基于SFT数据的问题构建DPO偏好对"""
        
        # 提取问题并关联原始section（用于找相关参考资料）
        question_items = []
        seen_questions = set()
        
        for item in sft_data:
            msgs = item["messages"]
            section = item.get("source_section", "")
            
            for msg in msgs:
                if msg["role"] == "user":
                    q = msg["content"].strip()
                    # 简单去重（基于前50字符）
                    q_key = q[:50].lower()
                    if q_key not in seen_questions:
                        seen_questions.add(q_key)
                        question_items.append({
                            "question": q,
                            "section": section
                        })
                    break
        
        random.shuffle(question_items)
        
        # 构建参考文档索引（按section分组）
        section_map = self._build_section_map(chunks)
        
        print(f"\n[DPO] Unique questions extracted: {len(question_items)}")
        
        all_dpo = []
        for i, qi in enumerate(question_items):
            question = qi["question"]
            section = qi["section"]
            
            print(f"[DPO] {i+1}/{len(question_items)}: {question[:60]}...")
            
            # 获取相关参考内容
            reference = self._get_relevant_reference(section, section_map, chunks)
            
            dpo_item = self._generate_dpo_pair(question, reference)
            if dpo_item:
                all_dpo.append(dpo_item)
                print(f"  ✓ Generated")
            else:
                print(f"  ✗ Failed, skipping")
            
            # 定期保存
            if (i + 1) % 10 == 0:
                self._save_checkpoint(all_dpo, "dpo_checkpoint.json")
        
        print(f"\n[DPO] Total generated: {len(all_dpo)} items")
        return all_dpo
    
    def _save_checkpoint(self, data: list, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [Checkpoint] Saved {len(data)} items to {path}")
    
    # -------------------------------------------------------------------------
    def _build_section_map(self, chunks: List[Dict]) -> Dict[str, List[str]]:
        """按section title分组chunks"""
        section_map = {}
        for chunk in chunks:
            title = chunk.get("title", "General")
            if title not in section_map:
                section_map[title] = []
            section_map[title].append(chunk["content"])
        return section_map
    
    def _get_relevant_reference(self, section: str, section_map: Dict, 
                                 chunks: List[Dict], max_len: int = 3000) -> str:
        """获取与问题相关的参考内容"""
        # 优先使用同section的内容
        if section in section_map:
            relevant = '\n\n'.join(section_map[section])
            if len(relevant) <= max_len:
                return relevant
            return relevant[:max_len]
        
        # fallback: 拼接所有chunks（截断）
        all_text = '\n\n'.join(c["content"] for c in chunks)
        return all_text[:max_len]
    
    # -------------------------------------------------------------------------
    def _generate_dpo_pair(self, question: str, reference: str) -> Dict:
        """为单个问题生成chosen + rejected偏好对"""
        
        system_prompt = """You are a data quality expert for training AI assistants.
Given a question about university policies and reference material, generate TWO answers:

【CHOSEN (high-quality) requirements】
1. Completely accurate, based on reference material
2. Well-structured with clear formatting
3. Complete - covers all relevant points
4. Professional yet friendly tone
5. Includes specific details (dates, percentages, procedures)

【REJECTED (lower-quality) requirements - pick ONE OR MORE flaws】
1. Incomplete - misses important conditions, steps, or details
2. Partially incorrect - wrong dates, percentages, or requirements  
3. Vague and generic - lacks specific information
4. Poor structure - disorganized or hard to follow
5. Inappropriate tone - too casual, dismissive, or unhelpful
6. Mixes up related but different policies

IMPORTANT: The rejected answer should be plausible but flawed - "close but not quite right" 
is more valuable for training than obviously wrong answers."""

        user_prompt = f"""【Reference Material】
{reference}

【Question】
{question}

Generate one high-quality answer (chosen) and one flawed answer (rejected).

Output STRICTLY as JSON, no extra text:
{{
  "prompt": "{question}",
  "chosen": "High-quality accurate answer",
  "rejected": "Plausible but flawed answer",
  "reject_reason": "Brief description of what's wrong with the rejected answer"
}}"""
        
        response = self.client.call(system_prompt, user_prompt, temperature=0.9)
        results = self.client.parse_json_response(response)
        
        if results:
            item = results[0]
            if self._validate_dpo_format(item):
                return {
                    "prompt": item.get("prompt", question),
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "reject_reason": item.get("reject_reason", ""),
                }
        
        return None
    
    # -------------------------------------------------------------------------
    def _validate_dpo_format(self, item: dict) -> bool:
        """验证DPO数据格式"""
        for key in ["chosen", "rejected"]:
            if key not in item:
                return False
            if not isinstance(item[key], str) or len(item[key].strip()) < 10:
                return False
        
        # chosen和rejected不能几乎相同
        if item["chosen"].strip() == item["rejected"].strip():
            return False
        
        # 简单检查：两者相似度不能太高
        chosen_words = set(item["chosen"].lower().split())
        rejected_words = set(item["rejected"].lower().split())
        if chosen_words and rejected_words:
            overlap = len(chosen_words & rejected_words) / min(len(chosen_words), len(rejected_words))
            if overlap > 0.95:
                return False
        
        return True


# =============================================================================
# ========================= Step 5: 数据质量检查 ================================
# =============================================================================

class DataQualityChecker:
    """数据质量检查与统计"""
    
    @staticmethod
    def check_sft(data: List[Dict]) -> Dict:
        stats = {
            "total": len(data),
            "single_qa": sum(1 for d in data if d.get("type") == "single_qa"),
            "multi_turn": sum(1 for d in data if d.get("type") == "multi_turn"),
            "scene_qa": sum(1 for d in data if d.get("type") == "scene_qa"),
            "avg_turns": 0,
            "avg_q_len": 0,
            "avg_a_len": 0,
            "unique_sections": len(set(d.get("source_section", "") for d in data)),
        }
        
        total_turns = 0
        total_q_len = 0
        total_a_len = 0
        q_count = 0
        a_count = 0
        
        for item in data:
            msgs = item["messages"]
            total_turns += len(msgs)
            for msg in msgs:
                if msg["role"] == "user":
                    total_q_len += len(msg["content"].split())  # word count
                    q_count += 1
                elif msg["role"] == "assistant":
                    total_a_len += len(msg["content"].split())
                    a_count += 1
        
        if data:
            stats["avg_turns"] = round(total_turns / len(data), 1)
        if q_count:
            stats["avg_q_len_words"] = round(total_q_len / q_count, 1)
        if a_count:
            stats["avg_a_len_words"] = round(total_a_len / a_count, 1)
        
        return stats
    
    @staticmethod
    def check_dpo(data: List[Dict]) -> Dict:
        stats = {
            "total": len(data),
            "avg_prompt_len": 0,
            "avg_chosen_len": 0,
            "avg_rejected_len": 0,
            "chosen_longer_ratio": 0,
        }
        
        if not data:
            return stats
        
        prompt_lens = [len(d["prompt"].split()) for d in data]
        chosen_lens = [len(d["chosen"].split()) for d in data]
        rejected_lens = [len(d["rejected"].split()) for d in data]
        
        stats["avg_prompt_len_words"] = round(sum(prompt_lens) / len(data), 1)
        stats["avg_chosen_len_words"] = round(sum(chosen_lens) / len(data), 1)
        stats["avg_rejected_len_words"] = round(sum(rejected_lens) / len(data), 1)
        stats["chosen_longer_ratio"] = round(
            sum(1 for c, r in zip(chosen_lens, rejected_lens) if c > r) / len(data), 2
        )
        
        # 统计reject原因分布
        reasons = [d.get("reject_reason", "unknown") for d in data]
        stats["reject_reasons_sample"] = reasons[:5]
        
        return stats
    
    @staticmethod
    def print_report(sft_stats: Dict, dpo_stats: Dict):
        print(f"\n{'='*60}")
        print(f"  Dataset Quality Report")
        print(f"{'='*60}")
        
        print(f"\n📋 SFT Dataset:")
        print(f"  Total samples: {sft_stats['total']}")
        print(f"  Single-turn QA: {sft_stats['single_qa']}")
        print(f"  Multi-turn dialogue: {sft_stats['multi_turn']}")
        print(f"  Scenario QA: {sft_stats['scene_qa']}")
        print(f"  Unique sections covered: {sft_stats.get('unique_sections', 'N/A')}")
        print(f"  Avg turns/sample: {sft_stats['avg_turns']}")
        print(f"  Avg question length: {sft_stats.get('avg_q_len_words', 'N/A')} words")
        print(f"  Avg answer length: {sft_stats.get('avg_a_len_words', 'N/A')} words")
        
        print(f"\n📋 DPO Dataset:")
        print(f"  Total samples: {dpo_stats['total']}")
        print(f"  Avg prompt length: {dpo_stats.get('avg_prompt_len_words', 'N/A')} words")
        print(f"  Avg chosen length: {dpo_stats.get('avg_chosen_len_words', 'N/A')} words")
        print(f"  Avg rejected length: {dpo_stats.get('avg_rejected_len_words', 'N/A')} words")
        print(f"  Chosen is longer: {dpo_stats.get('chosen_longer_ratio', 0)*100:.0f}%")
        print(f"{'='*60}\n")




# =============================================================================
# ========================= Step 6: 主流程 =====================================
# =============================================================================

def save_json(data: list, path: str, remove_internal_fields: bool = True):
    """保存JSON文件"""
    if remove_internal_fields:
        clean_data = []
        for item in data:
            clean = {k: v for k, v in item.items() 
                     if k not in ("type", "source_section")}
            clean_data.append(clean)
    else:
        clean_data = data
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    print(f"[Save] Saved {len(clean_data)} items to {path}")


def load_checkpoint(path: str) -> list:
    """加载检查点"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Load] Loaded checkpoint: {len(data)} items from {path}")
        return data
    return []


def main():
    config = DatasetConfig()
    
    # ==================== 前置检查 ====================
    if not os.path.exists(config.pdf_path):
        print(f"[Error] PDF not found: {config.pdf_path}")
        print(f"Place your student handbook PDF in the current directory")
        print(f"or update DatasetConfig.pdf_path")
        return
    
    if config.api_key == "your-api-key-here":
        print("[Error] Please configure your API key!")
        print("Update DatasetConfig.api_key with your Qwen API key")
        print("Get one at: https://dashscope.console.aliyun.com/")
        return
    
    print(f"\n{'='*60}")
    print(f"  Student Handbook Dataset Builder")
    print(f"{'='*60}")
    print(f"  PDF:        {config.pdf_path}")
    print(f"  Model:      {config.model_name}")
    print(f"  SFT output: {config.sft_output_path}")
    print(f"  DPO output: {config.dpo_output_path}")
    print(f"  Chunk size: {config.chunk_size} chars")
    print(f"  Per chunk:  {config.sft_qa_per_chunk} QA + "
          f"{config.sft_multi_per_chunk} multi + "
          f"{config.sft_scene_per_chunk} scene")
    print(f"{'='*60}\n")
    
    # ==================== Step 1: PDF解析与分块 ====================
    print("=" * 30 + " Step 1: PDF Parsing " + "=" * 30)
    parser = PDFParser(config)
    raw_text = parser.extract_text(config.pdf_path)
    clean_text = parser.clean_text(raw_text)
    chunks = parser.split_into_chunks(clean_text)
    
    # 保存chunks（调试用）
    with open(config.chunks_debug_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[Debug] Chunks saved to {config.chunks_debug_path}")
    
    # 预估API调用次数
    calls_per_chunk = 3  # single_qa + multi_turn + scene_qa
    est_sft_calls = len(chunks) * calls_per_chunk
    est_questions = len(chunks) * (config.sft_qa_per_chunk + 
                                    config.sft_multi_per_chunk + 
                                    config.sft_scene_per_chunk)
    est_total = est_sft_calls + est_questions
    print(f"\n[Estimate] ~{est_sft_calls} SFT API calls + ~{est_questions} DPO calls")
    print(f"[Estimate] Total: ~{est_total} API calls")
    print(f"[Estimate] Time: ~{est_total * config.request_delay / 60:.0f} minutes\n")
    
    proceed = input("Proceed? [y/N]: ").strip().lower()
    if proceed != 'y':
        print("Aborted.")
        return
    
    # ==================== Step 2: 初始化API ====================
    print("\n" + "=" * 30 + " Step 2: API Init " + "=" * 30)
    client = QwenClient(config)
    
    test_resp = client.call(
        "You are a helpful assistant.",
        "Reply with exactly: API_OK",
        temperature=0.1
    )
    if test_resp and "OK" in test_resp.upper():
        print(f"[API] Connection test passed: {test_resp[:30]}")
    else:
        print(f"[API] Connection test result: {test_resp[:50]}")
        cont = input("API response unexpected. Continue anyway? [y/N]: ").strip().lower()
        if cont != 'y':
            return
    


    # ==================== Step 3: 生成SFT数据 ====================
    print("\n" + "=" * 30 + " Step 3: SFT Generation " + "=" * 30)
    
    # 检查是否有checkpoint可以恢复
    sft_data = load_checkpoint("sft_checkpoint.json")
    if sft_data:
        use_ckpt = input(f"Found SFT checkpoint ({len(sft_data)} items). Use it? [Y/n]: ") # 默认使用checkpoint
        if use_ckpt.strip().lower() == 'n':  # 如果用户选择不使用checkpoint，则清空数据重新生成
            sft_data = []
    
    if not sft_data:
        sft_generator = SFTDataGenerator(config, client)
        sft_data = sft_generator.generate_all(chunks)
    
    save_json(sft_data, config.sft_output_path, remove_internal_fields=True) # 保存最终版本（去掉type/source_section等调试字段）
    # 也保存带元信息的版本（调试用）
    save_json(sft_data, "sft_dataset_full.json", remove_internal_fields=False) # 包含type/source_section等字段，便于后续分析和DPO生成使用
    


    # ==================== Step 4: 生成DPO数据 ====================
    print("\n" + "=" * 30 + " Step 4: DPO Generation " + "=" * 30)
    
    dpo_data = load_checkpoint("dpo_checkpoint.json")
    if dpo_data:
        use_ckpt = input(f"Found DPO checkpoint ({len(dpo_data)} items). Use it? [Y/n]: ")
        if use_ckpt.strip().lower() == 'n':
            dpo_data = []
    
    if not dpo_data:
        dpo_generator = DPODataGenerator(config, client)
        dpo_data = dpo_generator.generate_from_sft(sft_data, chunks)
    
    save_json(dpo_data, config.dpo_output_path)
    


    # ==================== Step 5: 质量报告 ====================
    print("\n" + "=" * 30 + " Step 5: Quality Report " + "=" * 30)
    checker = DataQualityChecker()
    sft_stats = checker.check_sft(sft_data)
    dpo_stats = checker.check_dpo(dpo_data)
    checker.print_report(sft_stats, dpo_stats)
    
    client.print_stats()
    
    print("\n🎉 Pipeline complete!")
    print(f"  SFT: {config.sft_output_path} ({len(sft_data)} samples)")
    print(f"  DPO: {config.dpo_output_path} ({len(dpo_data)} samples)")


if __name__ == "__main__":
    main()




'''

## 安装依赖

```bash
pip install pdfplumber openai
```

## 使用方式

```bash
# 1. 把学生手册PDF放到当前目录，命名为 student_handbook.pdf
#    或修改 DatasetConfig.pdf_path

# 2. 修改 DatasetConfig.api_key 为你的通义千问API Key
#    获取地址: https://dashscope.console.aliyun.com/

# 3. 运行
python build_dataset.py
```

## 关键设计说明

```
┌────────────────────┐
│ student_handbook.pdf│
└─────────┬──────────┘
          ▼
┌─────────────────────┐
│  PDF提取 + 清洗       │  处理英文断词、页码等
│  智能分块(按章节)      │  Chapter/Section/Article 识别
└─────────┬───────────┘
          ▼
┌─────────────────────────────────────────┐
│        Qwen3-Max 生成 SFT 数据           │
│  ┌───────────┬────────────┬───────────┐ │
│  │ Single QA │ Multi-turn │ Scenario  │ │
│  │  5/chunk  │  2/chunk   │  2/chunk  │ │
│  └───────────┴────────────┴───────────┘ │
│  • enable_thinking=False 纯输出          │
│  • 每5个chunk自动checkpoint              │
└─────────────────┬───────────────────────┘
                  ▼
           sft_dataset.json
                  │
                  ▼  提取问题 + 去重
┌─────────────────────────────────────────┐
│        Qwen3-Max 生成 DPO 数据           │
│                                         │
│  每个问题 → chosen(优质) + rejected(有缺陷)│
│  • 关联原始section作为参考资料             │
│  • rejected要求"似是而非"                 │
│  • 每10条自动checkpoint                  │
└─────────────────┬───────────────────────┘
                  ▼
           dpo_dataset.json
```

**与中文版的核心区别：**

| 改动点 | 说明 |
|--------|------|
| PDF清洗 | 处理英文断词(`hyphen-\nated`)、英文页码格式 |
| 章节识别 | Chapter/Section/Article/Part + 全大写标题 |
| chunk_size | 2000字符（英文比中文字符多） |
| Prompt语言 | 全英文system/user prompt |
| 模型配置 | `qwen-max` + `enable_thinking=False` |
| 词级统计 | 质量报告按word count而非字符数 |
| DPO验证 | 增加词级相似度检查，防止chosen≈rejected |
| 断点恢复 | SFT/DPO都支持checkpoint加载 |



## 输出文件

### SFT：2个JSON

| 文件 | 说明 |
|------|------|
| `sft_dataset.json` | **正式数据集**，只含 `messages` 字段，直接用于训练 |
| `sft_dataset_full.json` | **调试版**，额外保留 `type`（单轮/多轮/场景）和 `source_section`（来源章节），方便排查问题 |

### DPO：1个JSON

| 文件 | 说明 |
|------|------|
| `dpo_dataset.json` | **正式数据集**，含 `prompt`、`chosen`、`rejected`、`reject_reason` |

### 另外还有2个辅助文件

| 文件 | 说明 |
|------|------|
| `chunks_debug.json` | PDF分块结果，调试用 |
| `sft/dpo_checkpoint.json` | 中途自动保存的断点，防止生成过程中断丢数据 |

**总结：训练实际只需要 `sft_dataset.json` + `dpo_dataset.json` 两个文件。**

'''
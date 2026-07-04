"""
Causal Relation Extractor

Uses LLM to extract causal triples from paper abstracts.
Supports both strict (causal-only) and relaxed (includes associations) modes.
"""

import os
import re
import json
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from .pubmed_client import Paper


@dataclass
class CausalTriple:
    """Extracted causal triple"""
    head_entity: str
    relation_type: str  # Cause, Treat, Prevent, Worsen, Associated, NoEffect
    tail_entity: str
    confidence: float
    evidence_text: str
    pmid: str
    is_causal: bool = True  # True for strict causal, False for associations
    direction: str = "unclear"  # beneficial, harmful, neutral, unclear


# ============ RELAXED EXTRACTION PROMPT (Stage 5 Online) ============
RELAXED_EXTRACTION_PROMPT = """你是生物医学文献分析专家。任务：从论文摘要中提取与查询实体对相关的医学关系。

=== 输入信息 ===

查询实体对:
- 头实体: {head_entity}
- 尾实体: {tail_entity}

论文信息:
- PMID: {pmid}
- 标题: {title}
- 摘要: {abstract}

=== 实体匹配规则（宽松匹配） ===

允许以下情况视为匹配：

1. 精确匹配: "aspirin" = "aspirin"
2. 子类/变体:
   - "vitamin" 匹配 "Vitamin D", "Vitamin B12", "vitamin deficiency"
   - "heart disease" 匹配 "cardiovascular disease", "coronary artery disease"
3. 药物-成分: "fish oil" 匹配 "omega-3 fatty acids"
4. 疾病-症状: "diabetes" 匹配 "hyperglycemia", "insulin resistance"

提取时记录实际出现的实体名称。

=== 关系类型（6类） ===

| 类型 | 含义 | 示例表述 |
|------|------|----------|
| Treat | 明确结果显示治疗有效/显著改善/优于对照 | "significantly improved", "superior to placebo", "effective in reducing" |
| Prevent | 预防/降低风险 | "prevents", "reduces risk of", "protective against" |
| Cause | 导致/引起 | "causes", "leads to", "induces", "results in" |
| Worsen | 加重/恶化 | "worsens", "exacerbates", "increases risk of" |
| NoEffect | 明确结果显示无效/无差异/未达到终点 | "no significant difference", "failed to demonstrate", "not statistically significant", "comparable to placebo", "no superiority", "did not improve", "no benefit observed" |
| Associated | 存在关联（非因果） | "associated with", "correlated with", "linked to" |

注意：Associated 类型也要提取，这对搜索决策有参考价值。
仅当摘要中出现"结果/结论"层面的表述时，才可使用 Treat/Prevent/Cause/Worsen/NoEffect。仅有研究目的、试验设计或"正在研究/用于治疗"的表述，不得判为任何方向性关系类型。

=== 置信度评估（与方向无关） ===

| 分数 | 证据强度 |
|------|----------|
| 0.85-0.95 | RCT/Meta-analysis，样本量大，结论明确（无论是有效还是无效） |
| 0.70-0.85 | RCT/队列研究，结论清晰 |
| 0.55-0.70 | 观察性研究，有统计数据 |
| 0.40-0.55 | 有明确结论陈述但无统计数据 |
| 0.25-0.40 | 仅相关描述或结论不明确 |

所有 >= 0.25 的关系都输出，由下游系统决定使用。
注意：高质量 RCT 若结论为"无效"，应输出 NoEffect 且置信度可高。

=== 结论优先原则 ===

关系类型必须基于"研究结果/结论"，而非"研究目的/设计/标题"。若摘要只描述了试验设计或研究目的（例如 "X was investigated for Y"），则不能判为 Treat/Prevent/Cause/Worsen/NoEffect。

=== 判断指南 ===

提取关系时问自己：
1. 摘要是否同时提到了头实体与尾实体（或其变体）？
2. 论文"结果/结论"对两者的关系是什么？（正向/负向/无效/仅相关/不明确）
3. 证据强度如何？（与方向无关）

若只有研究设计或研究目的、缺少结果/结论信息：只能输出 Associated（direction=unclear，低置信度），或标记为 no_direct_relation。

=== 否定结果优先级（强制覆盖） ===

若结果/结论中出现以下任一否定/无差异表述，关系类型**必须**为 NoEffect、direction=neutral，**严禁**输出 Treat/Prevent：
"no significant difference", "not statistically significant", "did not improve", "failed to demonstrate",
"no benefit", "no superiority", "comparable to placebo", "not better than", "did not differ",
"no difference", "no effect", "p>=0.05", "trend but not significant", "no advantage",
"no meaningful improvement", "not effective", "no reduction"

Treat/Prevent 仅当 evidence_text 明确包含显著性或优效表述时才可使用，例如：
"significantly improved/reduced", "p<0.05", "superior to", "effective in reducing", "statistically significant"

⚠ 背景/目的性表述（如"X is used to treat Y"、"investigated for"）不得作为 Treat/Prevent 的依据。

=== 两步抽取（必须遵守） ===
Step 1: 识别研究对象与研究问题（谁对谁，研究了什么）。
Step 2: 仅依据"结果/结论"判定关系方向：
- 结论含显著性措辞且有效 → Treat/Prevent
- 结论含否定/无差异措辞 → NoEffect（即使摘要背景提到"用于治疗"）
- 明确有害 → Worsen/Cause
- 仅相关/结论不清 → Associated（direction=unclear）

=== 输出格式 ===

仅输出JSON：

{{
    "triples": [
        {{
            "head_entity": "论文中实际出现的头实体名称",
            "relation_type": "Treat/Prevent/Cause/Worsen/NoEffect/Associated",
            "tail_entity": "论文中实际出现的尾实体名称",
            "confidence": 0.XX,
            "evidence_text": "支持该关系的原文片段（限100字内）",
            "is_causal": true/false,
            "direction": "beneficial/harmful/neutral/unclear"
        }}
    ],
    "paper_relevance": {{
        "discusses_head": true/false,
        "discusses_tail": true/false,
        "relevance_score": 0.0-1.0
    }},
    "extract_status": "found_relation/no_direct_relation/off_topic"
}}

字段说明：
- is_causal: true 表示严格因果关系（Treat/Prevent/Cause/Worsen），false 表示关联或无效
- direction:
  - beneficial = Treat 或 Prevent（对疾病有益）
  - harmful = Cause 或 Worsen（对疾病有害）
  - neutral = NoEffect
  - unclear = 无法判断方向（Associated 可带方向，也可为 unclear）
- paper_relevance: 即使没提取到关系，也要评估论文相关性

=== 示例 ===

输入：head="vitamin D", tail="depression"
摘要："This study found that vitamin D supplementation was associated with reduced depressive symptoms in elderly patients (p=0.03, n=200)."

输出：
{{
    "triples": [
        {{
            "head_entity": "vitamin D supplementation",
            "relation_type": "Associated",
            "tail_entity": "depressive symptoms",
            "confidence": 0.65,
            "evidence_text": "vitamin D supplementation was associated with reduced depressive symptoms (p=0.03, n=200)",
            "is_causal": false,
            "direction": "beneficial"
        }}
    ],
    "paper_relevance": {{
        "discusses_head": true,
        "discusses_tail": true,
        "relevance_score": 0.9
    }},
    "extract_status": "found_relation"
}}

注意：虽然是"associated with"而非严格因果，但仍然提取，并标记 is_causal=false。

示例（无效结果）：

输入：head="pregabalin", tail="chronic prostatitis"
摘要："In a randomized, double-blind trial, pregabalin did not significantly improve symptoms compared with placebo (p=0.62, n=180)."

输出：
{{
    "triples": [
        {{
            "head_entity": "pregabalin",
            "relation_type": "NoEffect",
            "tail_entity": "chronic prostatitis",
            "confidence": 0.90,
            "evidence_text": "pregabalin did not significantly improve symptoms compared with placebo (p=0.62, n=180)",
            "is_causal": false,
            "direction": "neutral"
        }}
    ],
    "paper_relevance": {{
        "discusses_head": true,
        "discusses_tail": true,
        "relevance_score": 0.9
    }},
    "extract_status": "found_relation"
}}

注意：高质量 RCT 结论为无效时，关系类型应为 NoEffect 而非 Treat，置信度仍然可高（反映证据质量而非效果方向）。
"""


# ============ STRICT EXTRACTION PROMPT (Causal-only) ============
STRICT_EXTRACTION_PROMPT = """你是生物医学因果关系提取专家。任务：从论文中提取严格的因果三元组。

核心原则：因果关系(Causation)与相关性(Correlation)不同。只有明确的因果证据才能提取，相关性描述必须排除。

=== 输入信息 ===

目标实体对:
- 头实体 (治疗/干预): {head_entity}
- 尾实体 (疾病/结果): {tail_entity}

论文信息:
- PMID: {pmid}
- 标题: {title}
- 摘要: {abstract}

=== 关系类型定义（仅限5类） ===

1. Cause (方向: lead_to)
   定义: A 直接导致 B 发生
   正例: "Smoking causes lung cancer" -> Smoking -Cause-> Lung Cancer
   反例: "COVID-19 patients present with fever" -> 这是症状描述，不是因果

2. Inhibit (方向: decrease)
   定义: A 降低/抑制 B 的发生或表达
   正例: "Tocilizumab inhibits IL-6 signaling" -> Tocilizumab -Inhibit-> IL-6
   反例: "Lower A levels were observed in B patients" -> 仅观察性相关

3. Stimulate (方向: increase)
   定义: A 增加/促进 B 的发生或表达
   正例: "IL-6 promotes inflammation through..." -> IL-6 -Stimulate-> Inflammation
   反例: "Higher A correlates with B" -> 仅统计相关

4. Treat (方向: cure)
   定义: 药物/干预措施缓解或治愈疾病
   正例: "Aspirin effectively treats headache" -> Aspirin -Treat-> Headache
   反例: "A is being investigated for B" -> 研究中，未确认疗效

5. NoEffect (方向: no_impact)
   定义: 明确证据表明 A 对 B 无作用
   正例: "Vitamin C showed no effect on cold prevention (p=0.89)" -> Vitamin C -NoEffect-> Cold
   反例: 没有提及的关系不能标为 NoEffect

=== 排除规则（必须遵守） ===

以下表述不构成因果关系，禁止提取：

1. 相关性描述
   - "A is associated with B"
   - "A correlates with B"
   - "A is linked to B"
   - "A is related to B"

2. 症状/特征描述
   - "A is characterized by B"
   - "A presents with B"
   - "B is a symptom of A"
   - "Patients with A have B"

3. 未确认的研究状态
   - "A is under investigation for B"
   - "A is a potential treatment for B"
   - "A may affect B"

4. 仅时间先后
   - "A occurred before B"
   - "A was followed by B"

=== 置信度规则 ===

0.85-0.95: 有统计显著性 (p<0.05) 且样本量>=100
0.75-0.85: 有统计显著性，样本量<100或未知
0.65-0.75: 有统计趋势但 p>=0.05
0.60-0.70: 明确因果陈述但无统计数据
<0.60: 不输出

=== 输出格式 ===

仅输出JSON，无额外文字：

{{
    "triples": [
        {{
            "head_entity": "{head_entity}",
            "relation_type": "Cause/Inhibit/Stimulate/Treat/NoEffect",
            "tail_entity": "{tail_entity}",
            "confidence": 0.XX,
            "evidence_text": "支持因果关系的原文片段",
            "is_causal": true,
            "direction": "beneficial/harmful/neutral"
        }}
    ],
    "paper_relevance": {{
        "discusses_head": true/false,
        "discusses_tail": true/false,
        "relevance_score": 0.0-1.0
    }},
    "extract_status": "found_relation/no_valid_causal_relation/off_topic"
}}

注意：
- direction: beneficial = Treat 或 Inhibit（对疾病有益）; harmful = Cause 或 Stimulate; neutral = NoEffect
- 若无有效因果关系，返回空triples列表
"""


_NOEFFECT_CUE_RE = re.compile(
    r"(no significant difference|not statistically significant|not significant"
    r"|did not improve|did not differ|failed to demonstrate|no benefit"
    r"|no superiority|comparable to placebo|not better than|no difference"
    r"|no effect|no advantage|not effective|no reduction"
    r"|no meaningful improvement|trend .{0,20}not significant"
    r"|p\s*[>=]\s*0\.05)",
    re.IGNORECASE,
)


class CausalExtractor:
    """LLM-based causal relation extractor"""

    DEFAULT_API_BASE = "https://www.packyapi.com/v1"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
        timeout: int = 60,
        mode: str = "relaxed"  # "strict" or "relaxed"
    ):
        """
        Initialize extractor.

        Args:
            api_key: API key for LLM service
            api_base: API base URL
            model: Model name
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            mode: "strict" for causal-only, "relaxed" for including associations
        """
        self.api_key = api_key or os.getenv("PACKY_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found. Set PACKY_API_KEY or LLM_API_KEY environment variable.")
        self.api_base = api_base or self.DEFAULT_API_BASE
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.mode = mode

        # Select prompt based on mode
        self.prompt_template = (
            RELAXED_EXTRACTION_PROMPT if mode == "relaxed"
            else STRICT_EXTRACTION_PROMPT
        )

        # Build API endpoint URL
        self.api_url = f"{self.api_base.rstrip('/')}/chat/completions"

        print(f"[Extractor] Initialized: mode={self.mode}, model={self.model}")

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling markdown code blocks"""
        if not response:
            return None

        response = response.strip()

        # Handle ```json ... ``` format
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            response = json_match.group(1).strip()

        # Handle bare ``` wrapper
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]

        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None

    def _determine_direction(self, relation_type: str, is_causal: bool) -> str:
        """Determine direction (beneficial/harmful/neutral) from relation type"""
        beneficial_types = {"Treat", "Prevent", "Inhibit"}
        harmful_types = {"Cause", "Worsen", "Stimulate"}
        neutral_types = {"NoEffect"}

        if relation_type in beneficial_types:
            return "beneficial"
        elif relation_type in harmful_types:
            return "harmful"
        elif relation_type in neutral_types:
            return "neutral"
        else:
            return "unclear"

    def _extract_with_prompt(
        self,
        paper: Paper,
        head_entity: str,
        tail_entity: Optional[str],
        prompt_template: str,
    ) -> Dict[str, Any]:
        """Shared extraction logic for any prompt template."""
        prompt = prompt_template.format(
            head_entity=head_entity,
            tail_entity=tail_entity or "",
            pmid=paper.pmid,
            title=paper.title[:500],
            abstract=paper.abstract[:2500],
        )

        for attempt in range(self.max_retries):
            try:
                print(f"[Extractor] Calling API for PMID:{paper.pmid} (attempt {attempt+1}/{self.max_retries})")

                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                }
                payload = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': self.temperature,
                    'max_tokens': 1500
                }

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code != 200:
                    error_msg = f"API returned {response.status_code}: {response.text[:200]}"
                    print(f"[Extractor] {error_msg}")
                    if attempt < self.max_retries - 1:
                        import time
                        time.sleep(1)
                        continue
                    return self._error_result(paper.pmid, error_msg)

                response_data = response.json()
                content = response_data['choices'][0]['message']['content'].strip()
                # Strip <think>...</think> block from reasoning models
                import re
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
                print(f"[Extractor] Response received ({len(content)} chars)")
                result = self._parse_json_response(content)

                if result is None:
                    print(f"[Extractor] JSON parse failed. Raw: {content[:200]}...")
                    if attempt < self.max_retries - 1:
                        continue
                    return self._error_result(paper.pmid, f"JSON parse failed: {content[:100]}...")

                triples = []
                raw_triples = result.get("triples", [])
                default_tail = tail_entity or ""
                print(f"[Extractor] Found {len(raw_triples)} triples for PMID:{paper.pmid}")

                for t in raw_triples:
                    evidence_text = t.get("evidence_text", "")
                    relation_type = t.get("relation_type", "Unknown")
                    is_causal = t.get("is_causal", relation_type not in ["Associated", "Unknown"])
                    direction = t.get("direction") or self._determine_direction(relation_type, is_causal)

                    # Guardrail: NoEffect must be neutral
                    if relation_type == "NoEffect":
                        direction = "neutral"
                        is_causal = False

                    # Guardrail: override Treat/Prevent when evidence contains NoEffect cues
                    if relation_type in ("Treat", "Prevent") and _NOEFFECT_CUE_RE.search(evidence_text):
                        print(f"[Extractor]   ! Guardrail: {relation_type}→NoEffect (evidence contains negation cue)")
                        relation_type = "NoEffect"
                        direction = "neutral"
                        is_causal = False

                    triple = CausalTriple(
                        head_entity=head_entity,
                        relation_type=relation_type,
                        tail_entity=t.get("tail_entity") or default_tail,
                        confidence=float(t.get("confidence", 0.5)),
                        evidence_text=evidence_text[:500],
                        pmid=paper.pmid,
                        is_causal=is_causal,
                        direction=direction,
                    )
                    triples.append(triple)
                    print(f"[Extractor]   - {relation_type}: {triple.confidence:.2f} ({direction}, causal={is_causal})")

                paper_relevance = result.get("paper_relevance", {})
                relevance_score = paper_relevance.get("relevance_score", 0.5 if triples else 0.0)

                return {
                    "pmid": paper.pmid,
                    "pub_year": paper.pub_year,
                    "triples": triples,
                    "paper_relevance": paper_relevance,
                    "relevance_score": relevance_score,
                    "extract_status": result.get("extract_status", "found_relation" if triples else "no_relation"),
                    "is_relevant": relevance_score > 0.3 or len(triples) > 0
                }

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[Extractor] API error (attempt {attempt+1}): {error_msg}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                return self._error_result(paper.pmid, error_msg)

        return self._error_result(paper.pmid, "Max retries exceeded")

    def extract_from_paper(
        self,
        paper: Paper,
        head_entity: str,
        tail_entity: str,
    ) -> Dict[str, Any]:
        """Extract causal relations from a single paper."""
        return self._extract_with_prompt(paper, head_entity, tail_entity, self.prompt_template)

    def _error_result(self, pmid: str, error_msg: str) -> Dict[str, Any]:
        """Return error result structure"""
        return {
            "pmid": pmid,
            "triples": [],
            "paper_relevance": {},
            "relevance_score": 0.0,
            "extract_status": "error",
            "is_relevant": False,
            "error": error_msg
        }

    def extract_batch(
        self,
        papers: List[Paper],
        head_entity: str,
        tail_entity: str
    ) -> List[Dict[str, Any]]:
        """Extract causal relations from a batch of papers."""
        results = []
        for paper in papers:
            result = self.extract_from_paper(paper, head_entity, tail_entity)
            results.append(result)
        return results


# Test function
if __name__ == "__main__":
    from .pubmed_client import Paper

    test_paper = Paper(
        pmid="12345678",
        title="Vitamin D supplementation and depression in elderly patients",
        abstract="This study found that vitamin D supplementation was associated with reduced depressive symptoms in elderly patients (p=0.03, n=200). Higher vitamin D levels correlated with better mood outcomes.",
        authors=["Smith J", "Jones A"],
        journal="J Psychiatry",
        pub_date="2023",
        pub_year=2023,
        mesh_terms=["Vitamin D", "Depression"]
    )

    print("=== Testing RELAXED mode ===")
    extractor = CausalExtractor(mode="relaxed")
    result = extractor.extract_from_paper(
        test_paper,
        head_entity="vitamin D",
        tail_entity="depression"
    )
    print(json.dumps(result, indent=2, default=str))

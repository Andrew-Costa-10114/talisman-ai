"""
Deterministic X Post Classification for BitTensor Subnet Relevance

Uses atomic tool calls for each classification dimension to achieve deterministic
LLM evaluation. Validators can verify miner classifications via exact matching
of canonical strings.

Key Features:
- Atomic decisions: One tool call per classification dimension
- Hierarchical trigger rules (SN mention > alias > name+anchor > NONE)
- Explicit abstain logic (subnet_id=0 for ties/unknown)
- Evidence extraction (exact spans + anchors for auditability)
"""

from openai import OpenAI
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import bittensor as bt
from difflib import SequenceMatcher

from .classifications import ContentType, Sentiment, TechnicalQuality, MarketAnalysis, ImpactPotential

# Import centralized config
try:
    from talisman_ai import config
except ImportError:
    config = None


@dataclass
class PostClassification:
    """Canonical classification result"""
    subnet_id: int
    subnet_name: str
    content_type: ContentType
    sentiment: Sentiment
    technical_quality: TechnicalQuality
    market_analysis: MarketAnalysis
    impact_potential: ImpactPotential
    relevance_confidence: str  # "high", "medium", "low"
    evidence_spans: List[str]  # Exact substrings that triggered the decision
    anchors_detected: List[str]  # BitTensor anchor words found
    
    def to_canonical_string(self) -> str:
        """Deterministic string for exact matching by validators"""
        sorted_evidence = "|".join(sorted([s.lower() for s in self.evidence_spans]))
        sorted_anchors = "|".join(sorted([s.lower() for s in self.anchors_detected]))
        return f"{self.subnet_id}|{self.content_type.value}|{self.sentiment.value}|{self.technical_quality.value}|{self.market_analysis.value}|{self.impact_potential.value}|{self.relevance_confidence}|{sorted_evidence}|{sorted_anchors}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization or database storage"""
        return {
            "subnet_id": self.subnet_id,
            "subnet_name": self.subnet_name,
            "content_type": self.content_type.value,
            "sentiment": self.sentiment.value,
            "technical_quality": self.technical_quality.value,
            "market_analysis": self.market_analysis.value,
            "impact_potential": self.impact_potential.value,
            "relevance_confidence": self.relevance_confidence,
            "evidence_spans": self.evidence_spans,
            "anchors_detected": self.anchors_detected,
        }
    
    def get_tokens_dict(self) -> dict:
        """Get subnet tokens dict for grader compatibility"""
        if self.subnet_id == 0:
            return {}
        return {self.subnet_name: 1.0}


# Atomic tool definitions - one per classification dimension
SUBNET_ID_TOOL = {
    "type": "function",
    "function": {
        "name": "identify_subnet",
        "description": "Identify which subnet this post is about",
        "parameters": {
            "type": "object",
            "properties": {
                "subnet_id": {"type": "integer", "description": "Subnet ID (0 if none/unclear)"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "evidence_spans": {"type": "array", "items": {"type": "string"}, "description": "Exact text spans that identify this subnet"},
                "anchors_detected": {"type": "array", "items": {"type": "string"}, "description": "BitTensor anchor words found"}
            },
            "required": ["subnet_id", "confidence", "evidence_spans", "anchors_detected"]
        }
    }
}

CONTENT_TYPE_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_content_type",
        "description": "Classify the type of content",
        "parameters": {
            "type": "object",
            "properties": {
                "content_type": {
                    "type": "string",
                    "enum": [ct.value for ct in ContentType],
                    "description": "Primary content type"
                }
            },
            "required": ["content_type"]
        }
    }
}

SENTIMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_sentiment",
        "description": "Classify market sentiment",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": [s.value for s in Sentiment],
                    "description": "Market sentiment"
                }
            },
            "required": ["sentiment"]
        }
    }
}

TECHNICAL_QUALITY_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_technical_quality",
        "description": "Assess technical content quality",
        "parameters": {
            "type": "object",
            "properties": {
                "quality": {
                    "type": "string",
                    "enum": [tq.value for tq in TechnicalQuality],
                    "description": "Technical quality level"
                }
            },
            "required": ["quality"]
        }
    }
}

MARKET_ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_market_analysis",
        "description": "Classify market analysis type",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": [ma.value for ma in MarketAnalysis],
                    "description": "Type of market analysis"
                }
            },
            "required": ["analysis_type"]
        }
    }
}

IMPACT_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_impact",
        "description": "Assess potential impact",
        "parameters": {
            "type": "object",
            "properties": {
                "impact": {
                    "type": "string",
                    "enum": [ip.value for ip in ImpactPotential],
                    "description": "Expected impact level"
                }
            },
            "required": ["impact"]
        }
    }
}


class SubnetRelevanceAnalyzer:
    """
    Deterministic X post classifier using atomic tool calls
    
    Each classification dimension is decided independently via its own tool call,
    eliminating compound decision variance.
    """
    
    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None, subnets: List[Dict] = None):
        """Initialize analyzer with subnet registry and LLM config"""
        self.subnet_registry = {}
        
        # Use provided values or fall back to centralized config
        if config:
            self.model = model or config.MODEL
            self.api_key = api_key or config.API_KEY
            self.llm_base = llm_base or config.LLM_BASE
        else:
            self.model = model
            self.api_key = api_key
            self.llm_base = llm_base
        
        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")
        
        self.client = OpenAI(base_url=self.llm_base, api_key=self.api_key)
        
        # Initialize subnets
        if subnets:
            self.subnets = {s["id"]: s for s in subnets}
            for s in subnets:
                self.subnet_registry[s["id"]] = s
        else:
            self.subnets = {}
        
        # Add NONE subnet
        self.subnets[0] = {
            "id": 0,
            "name": "NONE_OF_THE_ABOVE",
            "description": "General BitTensor content not specific to a listed subnet"
        }
        
        bt.logging.info(f"[ANALYZER] Initialized with model: {self.model}")
        if subnets:
            bt.logging.info(f"[ANALYZER] Registered {len(self.subnets)-1} subnets (+1 NONE)")
    
    def register_subnet(self, subnet_data: dict):
        """Register a subnet (backward compatibility)"""
        subnet_id = subnet_data['id']
        self.subnet_registry[subnet_id] = subnet_data
        self.subnets[subnet_id] = subnet_data
        bt.logging.debug(f"[ANALYZER] Registered subnet {subnet_id}: {subnet_data.get('name')}")
    
    def classify_keyword_based(self, text: str) -> Dict:
        """
        Keyword-based subnet classification using edit distance for fuzzy matching.
        
        Args:
            text: Post text to classify
            
        Returns:
            Dict with:
            - is_bittensor: Whether post is BitTensor-related
            - confidence: 'high' (anchor present) or 'low' (subnet name only, possible false positive)
            - subnet_scores: {subnet_id: score}
            - matched_subnets: [(subnet_id, name, score, evidence), ...] sorted by score
        """
        text_lower = text.lower()
        
        # Check for explicit BitTensor anchors
        has_anchor = bool(re.search(r'\bsn\d+\b|\bsubnet\b|\bbittensor\b|\btao\b|\$tao\b|\bopentensor\b', text_lower))
        
        # Extract words for matching (4+ chars, exclude ecosystem terms)
        ecosystem = re.compile(r'^(bittensor|opentensor|tao|subnets?|sn\d+)$')
        words = {w for w in re.findall(r'\b\w{4,}\b', text_lower) if not ecosystem.match(w)}
        
        # Find subnet matches
        matches = []
        for sid, data in self.subnets.items():
            if sid == 0:
                continue
            
            # SN pattern match (e.g., SN23, subnet 23) - (?!\d) prevents SN23 matching SN123
            if re.search(rf'\bsn\s*{sid}(?!\d)|\bsubnet\s+{sid}(?!\d)', text_lower):
                matches.append((sid, data['name'], 1.0, [f'SN{sid}']))
                continue
            
            # Check name and identifiers
            best = (0.0, [])
            for name in [data.get('name', '')] + data.get('unique_identifiers', []):
                if not name or len(name) < 4 or ecosystem.match(name.lower()):
                    continue
                name_lower = name.lower()
                
                # Exact match
                if name_lower in words:
                    best = (0.9, [name])
                    break
                
                # Fuzzy match (edit distance >= 0.85)
                for word in words:
                    sim = SequenceMatcher(None, word, name_lower).ratio()
                    if sim >= 0.85 and sim > best[0]:
                        best = (sim * 0.9, [f'{word}≈{name}'])
            
            if best[0] >= 0.75:
                matches.append((sid, data['name'], best[0], best[1]))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Determine result
        if not has_anchor and not matches:
            return {'is_bittensor': False, 'confidence': None, 'subnet_scores': {}, 'matched_subnets': []}
        
        return {
            'is_bittensor': True,
            'confidence': 'high' if has_anchor else 'low',
            'subnet_scores': {m[0]: m[2] for m in matches},
            'matched_subnets': matches
        }
    
    def _build_subnet_context(self) -> str:
        """Build rich semantic context for subnet identification"""
        contexts = []
        for sid in sorted(self.subnets.keys()):
            if sid == 0:
                continue
            s = self.subnets[sid]
            
            # Build comprehensive description (no truncation - let LLM handle it)
            ctx = f"SN{sid} ({s.get('name', 'Unknown')}): {s.get('description', '')}"
            
            # Add primary functions if available
            functions = s.get('primary_functions', [])
            if functions:
                ctx += f" | Functions: {', '.join(functions[:4])}"
            
            # Add all identifiers (not just first 3)
            ids = s.get('unique_identifiers', [])
            if ids:
                ctx += f" | IDs: {', '.join(ids)}"
            
            # Add distinguishing features if available
            features = s.get('distinguishing_features', [])
            if features:
                ctx += f" | Features: {', '.join(features[:3])}"
            
            contexts.append(ctx)
        
        return '\n'.join(contexts)
    
    def classify_post(self, text: str) -> Optional[PostClassification]:
        """
        Classify using atomic tool calls for each dimension
        
        Args:
            text: X post text to classify
            
        Returns:
            PostClassification if successful, None if parsing fails
        """
        try:
            # Step 1: Identify subnet (most critical decision)
            subnet_result = self._identify_subnet(text)
            
            # Step 2-6: Classify other dimensions atomically
            content_type = self._classify_content_type(text)
            sentiment = self._classify_sentiment(text)
            technical_quality = self._assess_technical_quality(text)
            market_analysis = self._classify_market_analysis(text)
            impact = self._assess_impact(text)
            
            # Build final classification
            return PostClassification(
                subnet_id=subnet_result['id'],
                subnet_name=subnet_result['name'],
                content_type=ContentType(content_type),
                sentiment=Sentiment(sentiment),
                technical_quality=TechnicalQuality(technical_quality),
                market_analysis=MarketAnalysis(market_analysis),
                impact_potential=ImpactPotential(impact),
                relevance_confidence=subnet_result['confidence'],
                evidence_spans=subnet_result['evidence'],
                anchors_detected=subnet_result['anchors']
            )
            
        except Exception as e:
            bt.logging.error(f"[ANALYZER] Classification error: {e}")
            return None
    
    def _identify_subnet(self, text: str) -> dict:
        """Identify subnet using keyword-based matching (no LLM)."""
        result = self.classify_keyword_based(text)
        
        if not result['is_bittensor']:
            return {'id': 0, 'name': "NONE_OF_THE_ABOVE", 'confidence': "low", 'evidence': [], 'anchors': []}
        
        if result['matched_subnets']:
            top = result['matched_subnets'][0]  # (sid, name, score, evidence)
            confidence = 'high' if top[2] >= 0.9 else 'medium' if top[2] >= 0.8 else 'low'
            return {
                'id': top[0],
                'name': top[1],
                'confidence': confidence,
                'evidence': top[3],
                'anchors': []
            }
        
        return {'id': 0, 'name': "NONE_OF_THE_ABOVE", 'confidence': "low", 'evidence': [], 'anchors': []}
    
    def _classify_content_type(self, text: str) -> str:
        """Atomic decision: Content type"""
        prompt = f"""You are classifying a social media post about BitTensor or cryptocurrency. Classify the content type.

Post: "{text}"

Instructions:
- Choose the MOST SPECIFIC category that best describes this post
- If multiple categories apply, choose the primary one
- Be precise: "announcement" is for official releases, not just mentions

Categories with examples:

- announcement: Official product launches, releases, updates
  Examples: "We're launching X", "Version 2.0 is live", "New feature released"
  Key: Official, formal releases from projects/teams
  
- partnership: Explicit collaborations, integrations, joint ventures
  Examples: "Partnering with X", "Integration with Y announced", "Joint venture with Z"
  Key: Two or more entities working together
  
- technical_insight: Technical analysis, architecture discussions, code
  Examples: "Using X architecture improves Y", "Code review shows...", "Technical deep dive into..."
  Key: Technical details, code, architecture discussions
  
- milestone: Achievements, metrics, progress updates
  Examples: "Reached 1M users", "50K transactions processed", "Milestone achieved: X"
  Key: Specific achievements, metrics, goals reached
  
- tutorial: How-to guides, step-by-step instructions
  Examples: "How to use X feature", "Step-by-step guide to...", "Tutorial: Getting started"
  Key: Educational, instructional content
  
- security: Audits, vulnerabilities, exploits, security updates
  Examples: "Security audit completed", "Vulnerability found in X", "Security patch released"
  Key: Security-related information
  
- governance: Voting, proposals, DAO decisions
  Examples: "DAO proposal to...", "Vote on proposal X", "Governance decision: Y"
  Key: Community/governance decisions
  
- market_discussion: Price talk, trading, speculation
  Examples: "Price prediction for X", "Trading strategy", "Market analysis shows..."
  Key: Focus on price/market/trading
  
- hiring: Job postings, recruitment
  Examples: "We're hiring X", "Job opening: Y role", "Looking for developers"
  Key: Employment opportunities
  
- meme: Jokes, entertainment, humor
  Examples: Memes, jokes, humorous content about crypto/tech
  Key: Entertainment, humor
  
- hype: Excitement, enthusiasm, promotional content
  Examples: "This is going to moon!", "Huge announcement coming!", "Get ready!"
  Key: Excitement without specific details
  
- opinion: Personal views, analysis, commentary
  Examples: "I think X will...", "In my opinion...", "Analysis: Y"
  Key: Personal perspective, analysis
  
- community: General chatter, engagement, discussions
  Examples: "What do you think about X?", "Discussion thread", "Community feedback"
  Key: Social interaction, general discussion
  
- fud: Fear, uncertainty, doubt, negative speculation
  Examples: "X is going to crash", "Major concerns about Y", "This is a scam"
  Key: Negative speculation, spreading doubt
  
- other: Doesn't fit any category above
  Examples: Unclear content, doesn't match any category
  Key: Catch-all for uncategorized content

Choose the category name that best fits this post."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[CONTENT_TYPE_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_content_type"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("content_type", "other")
        except:
            return "other"
    
    def _classify_sentiment(self, text: str) -> str:
        """Atomic decision: Sentiment"""
        prompt = f"""You are analyzing a social media post about BitTensor or cryptocurrency. Classify the market sentiment.

Post text: "{text}"

Analyze the overall tone and implied market outlook. Consider:
- Explicit sentiment indicators (emojis, language)
- Implied market direction (positive/negative/neutral)
- Intensity of the sentiment

Sentiment categories:

1. very_bullish - Extreme positive sentiment
   Indicators: 🚀, "moon", "ATH", "pump", "10x", "massive gains", "game changer", "revolutionary"
   Examples: "This is going to the moon! 🚀", "ATH incoming, this is huge", "10x potential here"
   Tone: Highly enthusiastic, expecting major gains
   
2. bullish - Positive but measured optimism
   Indicators: "looking good", "solid", "bullish", "positive trend", "optimistic"
   Examples: "Bullish on this project", "Fundamentals are solid", "Positive momentum building"
   Tone: Optimistic, expecting upward movement
   
3. neutral - Factual, balanced, no strong bias
   Indicators: Informational content, factual statements, balanced reporting
   Examples: "Version 2.0 has been released", "DAO proposal passed", "Metrics: 1M users"
   Tone: Informational, no implied market direction
   
4. bearish - Negative concerns or skepticism
   Indicators: "concerns", "worried", "downward", "issues", "skeptical"
   Examples: "Concerned about the roadmap", "Not convinced this will work", "Downward trend continues"
   Tone: Skeptical, expecting potential decline
   
5. very_bearish - Extreme negative sentiment
   Indicators: "crash", "failure", "exploit", "doomed", "rug pull", "scam"
   Examples: "This is going to crash", "Major exploit found", "Project is doomed"
   Tone: Highly negative, expecting major problems

Choose the category that best matches the post's sentiment. Be precise - distinguish between bullish (measured) and very_bullish (extreme). Consider the implied market impact and tone, not just explicit price mentions."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[SENTIMENT_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_sentiment"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("sentiment", "neutral")
        except:
            return "neutral"
    
    def _assess_technical_quality(self, text: str) -> str:
        """Atomic decision: Technical quality"""
        prompt = f"""Assess the technical quality and specificity of this post.

Post: "{text}"

Rate based on how many concrete, verifiable technical details are present:

- high: 2+ specific technical details
  Examples: "Using API v2.3.1", "Achieved 99.9% uptime", "Implemented SHA-256 hashing", "Reduced latency to 12ms"
  Must have: Specific versions, metrics, algorithms, protocols, technical specifications
  Criteria: At least two verifiable, objective technical facts
  
- medium: 1 specific technical detail
  Examples: "Version 2.0 released", "Uses REST API", "Performance improved by 50%"
  Has some specificity but limited details
  Criteria: One concrete technical detail (version, metric, protocol, etc.)
  
- low: Technical claims without specifics
  Examples: "Fast performance", "Secure system", "Scalable architecture", "High throughput"
  Mentions technology but no concrete details or verifiable facts
  Criteria: General technical claims without supporting specifics
  
- none: No technical content
  Examples: Price discussions, memes, general announcements without technical details
  Purely social/market content with no technical information
  Criteria: No technical information present

Count only verifiable, objective technical details (versions, metrics, protocols, algorithms). Subjective claims like "fast" or "secure" don't count unless quantified. General buzzwords don't count."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[TECHNICAL_QUALITY_TOOL],
                tool_choice={"type": "function", "function": {"name": "assess_technical_quality"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("quality", "none")
        except:
            return "none"
    
    def _classify_market_analysis(self, text: str) -> str:
        """Atomic decision: Market analysis type"""
        prompt = f"""Classify the type of market analysis in this post, if any.

Post: "{text}"

Only classify as a market analysis type if the post contains analytical reasoning about market dynamics. If the post just mentions price without analysis, choose "other".

Categories:

- technical: Technical/on-chain analysis
  Examples: "RSI shows oversold", "On-chain metrics indicate...", "Chart pattern suggests breakout", "MACD crossover signals..."
  Indicators: Charts, technical indicators, on-chain data, trading patterns
  Key: Uses technical/on-chain data to predict or analyze market movement
  
- economic: Economic fundamentals analysis
  Examples: "Tokenomics show inflation concerns", "Revenue model suggests growth", "Cost structure indicates profitability"
  Focus: Financial fundamentals, economics, business model, tokenomics
  Key: Analyzes economic factors affecting value
  
- political: Regulatory/governance analysis
  Examples: "Regulation will impact adoption", "DAO decision affects token distribution", "Policy changes mean..."
  Focus: Governance decisions, regulations, policy, legal factors
  Key: Analyzes how governance/regulatory factors affect market
  
- social: Social sentiment/narrative analysis
  Examples: "Community sentiment turning bullish", "Narrative shifting to DeFi", "Social signals show growing interest"
  Focus: Social dynamics, narratives, community sentiment, virality
  Key: Analyzes social factors and narratives affecting market perception
  
- other: Not a market analysis, or different type
  Examples: Simple price mentions ("Price is $X"), announcements without market analysis, memes, general discussion
  Key: No analytical reasoning about market dynamics, just informational content

If post just mentions price without analytical reasoning about market dynamics, choose "other". The post must analyze why/how market factors affect value."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[MARKET_ANALYSIS_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_market_analysis"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("analysis_type", "other")
        except:
            return "other"
    
    def _assess_impact(self, text: str) -> str:
        """Atomic decision: Impact potential"""
        prompt = f"""Assess the potential impact this post's content would have on the BitTensor ecosystem.

Post: "{text}"

Evaluate based on how significant the news/information would be to the ecosystem. Consider:
- Scope: How many users/projects would be affected?
- Significance: How important is this information?
- Actionability: Does it enable new capabilities or indicate major changes?

Impact levels:

- HIGH: Major, ecosystem-wide impact
  Examples: Critical security vulnerabilities, major protocol upgrades, significant partnerships with major players, major exploits, fundamental changes to network operations
  Impact: Would affect many users/projects across the ecosystem, significant market implications
  Criteria: Information that would materially change how people interact with or perceive the ecosystem
  
- MEDIUM: Notable but limited impact
  Examples: Feature releases, medium partnerships, notable milestones (but not groundbreaking), protocol updates (minor), new integrations
  Impact: Affects specific subnets/users, moderate significance
  Criteria: Important but not ecosystem-defining, affects a subset of the ecosystem
  
- LOW: Minor information
  Examples: Minor updates, small partnerships, routine announcements, feature additions (small), UI improvements
  Impact: Limited scope, minimal ecosystem-wide effect
  Criteria: Useful information but with limited reach or significance
  
- NONE: Chatter, no real impact
  Examples: Price speculation without news, memes, general discussion, personal opinions, routine social interaction
  Impact: No actionable information or ecosystem significance
  Criteria: No substantive information that would affect ecosystem operations or perception

Focus on the information's potential to affect the BitTensor ecosystem, not just engagement levels. News that affects multiple subnets or fundamental operations = HIGH. Routine updates = MEDIUM/LOW."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[IMPACT_TOOL],
                tool_choice={"type": "function", "function": {"name": "assess_impact"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("impact", "NONE")
        except:
            return "NONE"
    
    def analyze_post_complete(self, text: str) -> dict:
        """
        Analyze post and return rich classification data
        
        Maintains backward compatibility with existing interface.
        """
        start_time = time.time()
        bt.logging.info(f"[ANALYZER] Starting analysis for post (length: {len(text)} chars)")
        
        # Run atomic classification
        classification = self.classify_post(text)
        
        if classification is None:
            bt.logging.warning(f"[ANALYZER] Classification failed")
            return {
                "classification": None,
                "subnet_relevance": {},
                "timestamp": datetime.now().isoformat()
            }
        
        # Build subnet_relevance dict
        subnet_relevance = {}
        if classification.subnet_id != 0:
            subnet_name = classification.subnet_name
            subnet_relevance[subnet_name] = {
                "subnet_id": classification.subnet_id,
                "subnet_name": subnet_name,
                "relevance": 1.0,
                "relevance_confidence": classification.relevance_confidence,
                "content_type": classification.content_type.value,
                "sentiment": classification.sentiment.value,
                "technical_quality": classification.technical_quality.value,
                "market_analysis": classification.market_analysis.value,
                "impact_potential": classification.impact_potential.value,
                "evidence_spans": classification.evidence_spans,
                "anchors_detected": classification.anchors_detected,
            }
        
        total_time = time.time() - start_time
        bt.logging.info(f"[ANALYZER] Analysis completed in {total_time:.2f}s")
        
        # Sentiment mapping for backward compatibility
        sentiment_to_float = {
            "very_bullish": 1.0,
            "bullish": 0.5,
            "neutral": 0.0,
            "bearish": -0.5,
            "very_bearish": -1.0
        }
        sentiment_enum = classification.sentiment.value if classification else "neutral"
        sentiment_float = sentiment_to_float.get(sentiment_enum, 0.0)
        
        return {
            "classification": classification,
            "subnet_relevance": subnet_relevance,
            "sentiment": sentiment_float,
            "sentiment_enum": sentiment_enum,
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_classification(self, args: dict) -> Optional[PostClassification]:
        """Parse and validate function call arguments (backward compatibility)"""
        try:
            subnet_id = int(args["subnet_id"])
            if subnet_id not in self.subnets:
                bt.logging.warning(f"[ANALYZER] Unknown subnet_id: {subnet_id}")
                return None
            
            return PostClassification(
                subnet_id=subnet_id,
                subnet_name=self.subnets[subnet_id]["name"],
                content_type=ContentType(args["content_type"]),
                sentiment=Sentiment(args["sentiment"]),
                technical_quality=TechnicalQuality(args["technical_quality"]),
                market_analysis=MarketAnalysis(args["market_analysis"]),
                impact_potential=ImpactPotential(args["impact_potential"]),
                relevance_confidence=args["relevance_confidence"],
                evidence_spans=args.get("evidence_spans", []),
                anchors_detected=args.get("anchors_detected", [])
            )
        except (ValueError, KeyError) as e:
            bt.logging.error(f"[ANALYZER] Parse error: {e}")
            return None

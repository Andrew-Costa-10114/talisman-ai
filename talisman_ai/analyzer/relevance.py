from openai import OpenAI
import json
from typing import Dict, List
from datetime import datetime
import time
import bittensor as bt  # Add this import for logging

# Import centralized config (loads .miner_env and .vali_env)
from talisman_ai import config

class SubnetRelevanceAnalyzer:
    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None):
        self.subnet_registry = {}  # Initialize registry first
        
        # Use provided values or fall back to centralized config
        self.model = model or config.MODEL
        self.api_key = api_key or config.API_KEY
        self.llm_base = llm_base or config.LLM_BASE
        
        # Validate API_KEY is set
        if not self.api_key:
            raise ValueError(
                f"API_KEY environment variable is required.\n"
                f"Please set it in talisman_ai_subnet/.miner_env or .vali_env file."
            )
        
        self.client = OpenAI(base_url=self.llm_base, api_key=self.api_key)
        bt.logging.info(f"[ANALYZER] Initialized with model: {self.model}")
    
    def register_subnet(self, subnet_data: dict):
        """Register a subnet with its metadata"""
        subnet_id = subnet_data['id']
        self.subnet_registry[subnet_id] = subnet_data
        bt.logging.debug(f"[ANALYZER] Registered subnet {subnet_id}: {subnet_data.get('name')}")
    
    def analyze_tweet_complete(self, text: str) -> dict:
        """Complete analysis: relevance + sentiment"""
        start_time = time.time()
        bt.logging.info(f"[ANALYZER] Starting analysis for tweet (length: {len(text)} chars)")
        bt.logging.info(f"[ANALYZER] Total registered subnets: {len(self.subnet_registry)}")
        
        # Quick pre-filter: only evaluate subnets with keyword matches
        text_lower = text.lower()
        candidate_subnets = {}
        
        for subnet_id, subnet_data in self.subnet_registry.items():
            # Quick check: any identifier or function keyword in text?
            has_potential = False
            
            # Check unique identifiers (like SN13, etc)
            for identifier in subnet_data.get('unique_identifiers', []):
                if identifier.lower() in text_lower:
                    has_potential = True
                    bt.logging.debug(f"[ANALYZER] Subnet {subnet_data['name']} matched identifier: {identifier}")
                    break
            
            # If no identifier match, check if any primary function keywords appear
            if not has_potential:
                for func in subnet_data.get('primary_functions', []):
                    # Simple keyword check
                    if any(word.lower() in text_lower for word in func.split()[:3]):
                        has_potential = True
                        bt.logging.debug(f"[ANALYZER] Subnet {subnet_data['name']} matched function keyword: {func[:50]}")
                        break
            
            # Only evaluate subnets that passed pre-filter
            if has_potential:
                candidate_subnets[subnet_id] = subnet_data
        
        bt.logging.info(f"[ANALYZER] Pre-filter reduced {len(self.subnet_registry)} subnets to {len(candidate_subnets)} candidates")
        
        # Now only run expensive LLM evaluation on candidates
        subnet_relevance = {}
        llm_call_count = 0
        
        for i, (subnet_id, subnet_data) in enumerate(candidate_subnets.items(), 1):
            eval_start = time.time()
            bt.logging.debug(f"[ANALYZER] Evaluating subnet {i}/{len(candidate_subnets)}: {subnet_data['name']}")
            
            score_data = self._evaluate_single_subnet(text, subnet_data)
            llm_call_count += 1
            
            eval_time = time.time() - eval_start
            bt.logging.debug(f"[ANALYZER] Subnet {subnet_data['name']} evaluated in {eval_time:.2f}s - relevance: {score_data.get('relevance', 0.0)}")
            
            if score_data.get('relevance', 0.0) > 0:
                subnet_relevance[subnet_data['name']] = score_data
        
        bt.logging.info(f"[ANALYZER] {len(subnet_relevance)} subnets have non-zero relevance (from {llm_call_count} LLM calls)")
        
        # Limit to top 10 if needed
        if len(subnet_relevance) > 10:
            sorted_subnets = sorted(
                subnet_relevance.items(), 
                key=lambda x: x[1].get('relevance', 0.0), 
                reverse=True
            )[:10]
            subnet_relevance = dict(sorted_subnets)
            bt.logging.info(f"[ANALYZER] Limited to top 10 subnets by relevance")
        
        # Get sentiment once
        sentiment_start = time.time()
        sentiment_data = self._analyze_sentiment(text)
        sentiment_time = time.time() - sentiment_start
        bt.logging.debug(f"[ANALYZER] Sentiment analysis completed in {sentiment_time:.2f}s")
        
        total_time = time.time() - start_time
        bt.logging.info(f"[ANALYZER] Total analysis completed in {total_time:.2f}s ({llm_call_count + 1} total LLM calls)")
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "subnet_relevance": subnet_relevance,
            "sentiment": sentiment_data.get('sentiment', 0.0),
            "sentiment_reasoning": sentiment_data.get('reasoning', ''),
            "timestamp": datetime.now().isoformat()
        }
    
    def _evaluate_single_subnet(self, text: str, subnet: dict) -> dict:
        """Hybrid approach: LLM understanding + deterministic scoring"""
        
        # First, check for direct mentions (fully deterministic)
        text_lower = text.lower()
        direct_match = None
        
        # Check for BitTensor-specific subnet identifiers (SNxx format is unique to BitTensor)
        for identifier in subnet.get('unique_identifiers', []):
            # Check if it's a subnet number reference (SN2, SN13, etc)
            if identifier.lower().startswith('sn') and identifier.lower() in text_lower:
                direct_match = identifier
                break
            # For other identifiers, we'll rely on LLM context check
        
        # If direct SN match found, return immediately
        if direct_match:
            bt.logging.debug(f"[ANALYZER] Direct match found for subnet {subnet['name']}: {direct_match}")
            return {
                "relevance": 1.0,
                "confidence": "high",
                "matched_signals": [direct_match],
                "method": "direct_match"
            }
        
        # Otherwise, use LLM to identify concepts WITH BitTensor context check
        try:
            features = self._extract_features_llm(text, subnet)
            score = self._calculate_deterministic_score(features)
            return score
        except Exception as e:
            bt.logging.error(f"[ANALYZER] Error evaluating subnet {subnet['name']}: {e}")
            return {
                "relevance": 0.0,
                "confidence": "low",
                "matched_signals": [],
                "error": str(e)
            }
    
    def _extract_features_llm(self, text: str, subnet: dict) -> dict:
        """Use LLM to identify which subnet concepts are present"""
        
        system_prompt = "You identify BitTensor blockchain subnet concepts. Output only valid JSON."
        
        user_prompt = f"""Evaluate if this post is about a BitTensor blockchain subnet.

POST: "{text}"

BITTENSOR SUBNET TO EVALUATE: {subnet['name']} (Subnet {subnet['id']})
Primary Functions: {json.dumps(subnet.get('primary_functions', []))}
Unique Identifiers: {json.dumps(subnet.get('unique_identifiers', []))}

CRITICAL CONTEXT CHECK:
This evaluation is for BitTensor Network subnets - decentralized AI networks on the BitTensor blockchain.
The post MUST be about BitTensor/TAO/crypto/blockchain/decentralized AI to be relevant.

EVALUATION STEPS:
1. Is this post about BitTensor, TAO tokens, crypto subnets, or decentralized AI networks?
   - Look for: BitTensor, TAO, subnets, validators, miners, emissions, decentralized AI
   - If NO (e.g., traditional companies, non-crypto AI, unrelated topics) → return empty lists
   - If YES → continue to step 2

2. Does the post discuss any of this subnet's primary functions in a BitTensor context?
   - Only include functions clearly discussed in relation to BitTensor/crypto

Output format:
{{"is_bittensor_related": true/false, "functions_found": ["list of matched primary functions"], "relevant_terms": ["actual BitTensor-related terms from post"]}}

Examples:
- Post about electronics company → {{"is_bittensor_related": false, "functions_found": [], "relevant_terms": []}}
- Post about BitTensor subnet features → {{"is_bittensor_related": true, "functions_found": ["matched functions"], "relevant_terms": ["TAO", "subnet", etc]}}"""

        llm_start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        llm_time = time.time() - llm_start
        bt.logging.trace(f"[ANALYZER] LLM call for subnet {subnet['name']} took {llm_time:.2f}s")
        
        content = response.choices[0].message.content
        if "```" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        
        result = json.loads(content)
        
        # If not BitTensor related, return empty
        if not result.get('is_bittensor_related', False):
            return {"functions_found": [], "relevant_terms": []}
        
        return result
    
    def _calculate_deterministic_score(self, features: dict) -> dict:
        """Deterministic scoring based on extracted features"""
        
        functions_found = features.get('functions_found', [])
        relevant_terms = features.get('relevant_terms', [])
        
        # Deterministic scoring rules
        num_functions = len(functions_found)
        
        if num_functions >= 2:
            relevance = 0.7
            confidence = "high"
        elif num_functions == 1:
            relevance = 0.4
            confidence = "medium"
        elif len(relevant_terms) > 0:
            relevance = 0.2
            confidence = "low"
        else:
            relevance = 0.0
            confidence = "none"
        
        return {
            "relevance": relevance,
            "confidence": confidence,
            "matched_signals": functions_found + relevant_terms,
            "method": "llm_extraction"
        }
    
    def _analyze_sentiment(self, text: str) -> dict:
        """Crypto-focused sentiment analysis"""
        
        system_prompt = "You analyze sentiment in crypto/tech posts. Output only valid JSON."
        
        user_prompt = f"""Classify the sentiment of this post (may or may not be BitTensor-related).

POST: "{text}"

Determine the overall sentiment using these exact categories:
- 1.0 = Very bullish (excitement, major positive developments, strong endorsement)
- 0.5 = Moderately positive (optimistic, good news, mild endorsement)
- 0.0 = Neutral (factual, informative, balanced, or mixed signals)
- -0.5 = Moderately negative (concerns, skepticism, mild criticism)
- -1.0 = Very bearish (major issues, strong criticism, failure)

Consider:
1. Is the author promoting or criticizing?
2. Does it announce success or problems?
3. Is the tone enthusiastic or concerned?

Output format:
{{"sentiment": <value from above>, "reasoning": "one-line explanation"}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        
        content = response.choices[0].message.content
        if "```" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        
        try:
            result = json.loads(content)
            # Ensure sentiment is in valid range
            sentiment = result.get('sentiment', 0.0)
            result['sentiment'] = max(-1.0, min(1.0, sentiment))
            return result
        except:
            return {"sentiment": 0.0, "reasoning": "Failed to parse"}

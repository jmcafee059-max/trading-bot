import os
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIMarketAnalyzer:
    """Use OpenAI GPT to analyze market conditions and identify trading opportunities"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.enabled = os.getenv('OPENAI_ENABLED', 'false').lower() == 'true'
        self.client = None
        
        if self.enabled and self.api_key and self.api_key != 'your_openai_api_key_here':
            try:
                import openai
                openai.api_key = self.api_key
                self.client = openai
                logger.info("OpenAI client initialized successfully")
            except ImportError:
                logger.warning("OpenAI library not installed. Install with: pip install openai")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.enabled = False
        else:
            logger.info("OpenAI analyzer disabled")
    
    def analyze_market_conditions(self, price_data: pd.DataFrame, current_price: float) -> Dict:
        """Analyze current market conditions using OpenAI"""
        if not self.enabled or not self.client:
            return self._get_fallback_analysis(price_data, current_price)
        
        try:
            # Prepare market data summary
            market_summary = self._prepare_market_summary(price_data, current_price)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(market_summary)
            
            # Get OpenAI analysis
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trading analyst. Analyze market conditions and provide trading signals."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            return self._parse_openai_response(analysis, current_price)
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._get_fallback_analysis(price_data, current_price)
    
    def _prepare_market_summary(self, price_data: pd.DataFrame, current_price: float) -> str:
        """Prepare market data summary for OpenAI analysis"""
        if len(price_data) < 10:
            return "Insufficient data for analysis"
        
        recent_prices = price_data['close'].tail(20)
        
        # Calculate basic indicators
        price_change = ((current_price - price_data['close'].iloc[0]) / price_data['close'].iloc[0]) * 100
        volatility = recent_prices.std()
        trend = "UP" if recent_prices.iloc[-1] > recent_prices.iloc[0] else "DOWN"
        
        summary = f"""
        Current Price: ${current_price:.2f}
        Price Change (last 20 periods): {price_change:.2f}%
        Volatility: {volatility:.4f}
        Trend: {trend}
        Recent High: ${recent_prices.max():.2f}
        Recent Low: ${recent_prices.min():.2f}
        Current vs High: {((current_price - recent_prices.max()) / recent_prices.max() * 100):.2f}%
        Current vs Low: {((current_price - recent_prices.min()) / recent_prices.min() * 100):.2f}%
        """
        
        return summary
    
    def _create_analysis_prompt(self, market_summary: str) -> str:
        """Create analysis prompt for OpenAI"""
        prompt = f"""
        Analyze the following cryptocurrency market data and provide a trading recommendation:
        
        {market_summary}
        
        Based on this data, provide:
        1. Market sentiment (BULLISH/BEARISH/NEUTRAL)
        2. Trading recommendation (BUY/SELL/HOLD)
        3. Confidence level (1-10)
        4. Brief reasoning
        
        Format your response as:
        SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
        RECOMMENDATION: [BUY/SELL/HOLD]
        CONFIDENCE: [1-10]
        REASONING: [brief explanation]
        """
        return prompt
    
    def _parse_openai_response(self, response: str, current_price: float) -> Dict:
        """Parse OpenAI response into structured format"""
        try:
            lines = response.strip().split('\n')
            result = {
                'sentiment': 'NEUTRAL',
                'recommendation': 'HOLD',
                'confidence': 5,
                'reasoning': 'Analysis failed',
                'source': 'openai',
                'current_price': current_price
            }
            
            for line in lines:
                if 'SENTIMENT:' in line:
                    result['sentiment'] = line.split('SENTIMENT:')[1].strip().upper()
                elif 'RECOMMENDATION:' in line:
                    result['recommendation'] = line.split('RECOMMENDATION:')[1].strip().upper()
                elif 'CONFIDENCE:' in line:
                    try:
                        result['confidence'] = int(line.split('CONFIDENCE:')[1].strip())
                    except:
                        result['confidence'] = 5
                elif 'REASONING:' in line:
                    result['reasoning'] = line.split('REASONING:')[1].strip()
            
            logger.info(f"OpenAI Analysis: {result['recommendation']} (confidence: {result['confidence']})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            return self._get_fallback_analysis(pd.DataFrame(), current_price)
    
    def _get_fallback_analysis(self, price_data: pd.DataFrame, current_price: float) -> Dict:
        """Provide fallback analysis when OpenAI is unavailable"""
        return {
            'sentiment': 'NEUTRAL',
            'recommendation': 'HOLD',
            'confidence': 3,
            'reasoning': 'OpenAI disabled - using neutral stance',
            'source': 'fallback',
            'current_price': current_price
        }
    
    def cross_reference_signals(self, openai_signal: Dict, ml_buy_score: int, ml_sell_score: int) -> Dict:
        """Cross-reference OpenAI and ML signals for final decision"""
        combined_score = 0
        final_recommendation = 'HOLD'
        confidence = 0
        reasoning = []
        
        # OpenAI signal contribution
        if openai_signal['recommendation'] == 'BUY':
            combined_score += openai_signal['confidence']
            reasoning.append(f"OpenAI BUY (confidence: {openai_signal['confidence']})")
        elif openai_signal['recommendation'] == 'SELL':
            combined_score -= openai_signal['confidence']
            reasoning.append(f"OpenAI SELL (confidence: {openai_signal['confidence']})")
        
        # ML signal contribution
        if ml_buy_score >= 3:
            combined_score += ml_buy_score * 2
            reasoning.append(f"ML BUY (score: {ml_buy_score})")
        elif ml_sell_score >= 3:
            combined_score -= ml_sell_score * 2
            reasoning.append(f"ML SELL (score: {ml_sell_score})")
        
        # Determine final recommendation
        if combined_score >= 8:
            final_recommendation = 'STRONG BUY'
            confidence = min(10, combined_score // 2)
        elif combined_score >= 4:
            final_recommendation = 'BUY'
            confidence = min(8, combined_score // 2)
        elif combined_score <= -8:
            final_recommendation = 'STRONG SELL'
            confidence = min(10, abs(combined_score) // 2)
        elif combined_score <= -4:
            final_recommendation = 'SELL'
            confidence = min(8, abs(combined_score) // 2)
        else:
            final_recommendation = 'HOLD'
            confidence = 3
            reasoning.append("Signals conflicting or weak")
        
        return {
            'recommendation': final_recommendation,
            'confidence': confidence,
            'combined_score': combined_score,
            'reasoning': '; '.join(reasoning),
            'openai_signal': openai_signal,
            'ml_buy_score': ml_buy_score,
            'ml_sell_score': ml_sell_score
        }

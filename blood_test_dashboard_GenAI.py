import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import requests
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import io

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

# Time series models
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(
    page_title="AI Blood Test Forecasting System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AIForecastOptimizer:
    def __init__(self, api_key=None):
        # Load API key from environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Debug: Check if API key is loaded (don't print the actual key for security)
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables")
        else:
            print(f"API Key loaded successfully (length: {len(self.api_key)})")
    
    def optimize_forecast_strategy(self, historical_data, parameter_info, patient_context=None):
        """Use GenAI to determine optimal forecasting strategy"""
        if not self.api_key:
            return {"error": "API key not configured. Please check your .env file."}
            
        try:
            context = self._prepare_optimization_context(historical_data, parameter_info, patient_context)
            
            prompt = f"""
You are an AI medical forecasting expert. Analyze this blood parameter data to optimize prediction strategy:

{context}

Based on the data patterns, provide specific recommendations for:

1. OPTIMAL MODEL SELECTION: Which forecasting approach (trend-based, seasonal, ML ensemble) would work best for this parameter
2. FEATURE ENGINEERING: What additional features should be considered (time-based patterns, parameter interactions)
3. PREDICTION CONFIDENCE: How reliable should predictions be at different time horizons
4. CLINICAL INSIGHTS: What medical factors might influence future values
5. ALERT THRESHOLDS: Recommended early warning indicators

Focus on actionable AI strategies to improve forecast accuracy, not model comparison.
Return response in JSON format with keys: model_recommendation, features_to_add, confidence_levels, clinical_factors, alert_strategy
"""

            response = self._call_gemini_api(prompt)
            try:
                # Try to parse JSON response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {"model_recommendation": response}
            except:
                return {"model_recommendation": response}
                
        except Exception as e:
            return {"error": f"Optimization error: {str(e)}"}
    
    def enhance_predictions_with_ai(self, base_predictions, historical_data, parameter_name):
        """Use AI to enhance and refine base model predictions"""
        if not self.api_key:
            return base_predictions
            
        try:
            context = f"""
Historical values: {historical_data[-10:].tolist()}
Base predictions: {base_predictions.tolist()}
Parameter: {parameter_name}
"""
            
            prompt = f"""
As an AI medical forecasting specialist, enhance these blood test predictions:

{context}

Provide enhanced predictions considering:
1. Medical plausibility of predicted values
2. Physiological constraints and normal ranges
3. Trend smoothing based on medical knowledge
4. Risk-adjusted predictions for patient safety

Return only the enhanced prediction values as a comma-separated list, same length as input predictions.
Example format: 12.5, 12.3, 12.1, 11.9
"""

            response = self._call_gemini_api(prompt)
            
            # Parse enhanced predictions
            try:
                enhanced_values = [float(x.strip()) for x in response.split(',')]
                if len(enhanced_values) == len(base_predictions):
                    return np.array(enhanced_values)
            except:
                pass
            
            return base_predictions  # Return original if parsing fails
            
        except Exception as e:
            return base_predictions
    
    def generate_clinical_insights(self, forecast_results, parameter_info, patient_context=None):
        """Generate clinical insights about forecast results"""
        if not self.api_key:
            return "Clinical insights require API key configuration."
            
        try:
            context = f"""
Forecast results: {forecast_results}
Parameter: {parameter_info['ylabel']}
Normal range: {parameter_info['normal_range']}
Critical threshold: {parameter_info['critical_low']}
"""
            
            # ADD PATIENT CONTEXT TO THE PROMPT
            if patient_context:
                context += f"\nPatient Context: {patient_context}\n"
            
            prompt = f"""
Provide clinical insights for these blood test forecasts:

{context}

Generate insights focusing on:
1. CLINICAL INTERPRETATION: What these predictions mean medically
2. RISK ASSESSMENT: Potential health risks if trends continue
3. INTERVENTION POINTS: When medical intervention might be needed
4. MONITORING RECOMMENDATIONS: How frequently to retest
5. LIFESTYLE FACTORS: What might influence these parameters

{"Consider the patient's age, medical conditions, and context when providing recommendations." if patient_context else ""}

Keep response concise and medically relevant.
"""

            response = self._call_gemini_api(prompt)
            return response
            
        except Exception as e:
            return f"Clinical insights error: {str(e)}"
    
    def _prepare_optimization_context(self, historical_data, parameter_info, patient_context):
        """Prepare context for optimization analysis"""
        context = f"PARAMETER: {parameter_info['ylabel']}\n"
        context += f"NORMAL RANGE: {parameter_info['normal_range']}\n"
        context += f"CRITICAL THRESHOLD: {parameter_info['critical_low']}\n\n"
        
        if len(historical_data) > 0:
            context += "HISTORICAL DATA ANALYSIS:\n"
            context += f"- Data points: {len(historical_data)}\n"
            context += f"- Current value: {historical_data.iloc[-1]:.2f}\n"
            context += f"- Trend: {'Increasing' if historical_data.iloc[-1] > historical_data.iloc[0] else 'Decreasing'}\n"
            context += f"- Volatility: {historical_data.std():.3f}\n"
            context += f"- Min/Max: {historical_data.min():.2f} / {historical_data.max():.2f}\n\n"
        
        if patient_context:
            context += f"PATIENT CONTEXT: {patient_context}\n"
        
        return context
    
    def _call_gemini_api(self, prompt):
        """Call Gemini API with error handling"""
        if not self.api_key:
            return "API key not configured."
            
        try:
            headers = {'Content-Type': 'application/json'}
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1500
                }
            }
            
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"API Error: Status {response.status_code}, Response: {response.text}")
                return f"API Error: {response.status_code} - {response.text[:100]}"
                
        except Exception as e:
            print(f"API Connection Error: {str(e)}")
            return f"API connection issue: {str(e)}"

class SmartEnsembleForecaster:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ml_models = {
            'rf': RandomForestRegressor(n_estimators=50, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
    
    def create_features(self, dates, values):
        """Create time-based and statistical features"""
        features = []
        
        for i in range(len(values)):
            feature_row = []
            
            # Time-based features
            date = dates.iloc[i]
            feature_row.extend([
                date.dayofweek,
                date.month,
                (date - dates.min()).days
            ])
            
            # Statistical features (rolling windows)
            if i >= 2:
                recent_values = values.iloc[max(0, i-3):i+1]
                feature_row.extend([
                    recent_values.mean(),
                    recent_values.std() if len(recent_values) > 1 else 0,
                    recent_values.iloc[-1] - recent_values.iloc[0] if len(recent_values) > 1 else 0
                ])
            else:
                feature_row.extend([values.iloc[i], 0, 0])
            
            # Lag features
            if i >= 1:
                feature_row.append(values.iloc[i-1])
            else:
                feature_row.append(values.iloc[i])
                
            if i >= 2:
                feature_row.append(values.iloc[i-2])
            else:
                feature_row.append(values.iloc[0])
            
            features.append(feature_row)
        
        return np.array(features)
    
    def smart_ensemble_forecast(self, dates, values, periods=4):
        """AI-enhanced ensemble forecasting"""
        try:
            if len(values) < 5:
                return None, None, None
            
            # Create features
            X = self.create_features(dates, values)
            y = values.values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train ML models
            predictions_dict = {}
            
            for name, model in self.ml_models.items():
                model.fit(X_scaled, y)
                
                # Generate future predictions
                future_preds = []
                current_values = values.copy()
                current_dates = dates.copy()
                
                for step in range(periods):
                    # Create future date
                    future_date = current_dates.iloc[-1] + timedelta(weeks=1)
                    new_dates = pd.concat([current_dates, pd.Series([future_date])], ignore_index=True)
                    
                    # Create features for prediction
                    future_X = self.create_features(new_dates, 
                                                  pd.concat([current_values, pd.Series([current_values.iloc[-1]])], 
                                                           ignore_index=True))
                    future_X_scaled = self.scaler.transform(future_X[-1:])
                    
                    # Predict
                    pred = model.predict(future_X_scaled)[0]
                    future_preds.append(pred)
                    
                    # Update for next iteration
                    current_values = pd.concat([current_values, pd.Series([pred])], ignore_index=True)
                    current_dates = new_dates
                
                predictions_dict[name] = np.array(future_preds)
            
            # Weighted ensemble (Random Forest gets higher weight due to stability)
            weights = {'rf': 0.6, 'gb': 0.4}
            ensemble_pred = np.zeros(periods)
            
            for name, pred in predictions_dict.items():
                ensemble_pred += weights[name] * pred
            
            # Calculate confidence intervals based on model agreement
            pred_std = np.std([pred for pred in predictions_dict.values()], axis=0)
            margin = pred_std + 0.05 * np.abs(ensemble_pred)
            
            lower_bound = ensemble_pred - margin
            upper_bound = ensemble_pred + margin
            
            return ensemble_pred, (lower_bound, upper_bound), ensemble_pred
            
        except Exception as e:
            return None, None, None

class AIBloodTestForecaster:
    def __init__(self):
        self.parameters = {
            'Hemoglobin g/dL 12.0 - 15.0': {
                'ylabel': 'Hemoglobin (g/dL)', 
                'normal_range': (12.0, 15.0),
                'color': '#2E8B57',
                'critical_low': 8.0,
                'unit': 'g/dL'
            },
            'RBC Count mill/cmm 3.80 - 4.80': {
                'ylabel': 'RBC Count (mill/cmm)', 
                'normal_range': (3.80, 4.80),
                'color': '#4169E1',
                'critical_low': 3.0,
                'unit': 'mill/cmm'
            },
            'Leukocyte Count (TLC/WBC) /cumm 4.00 - 10.00': {
                'ylabel': 'Leukocyte Count (/cumm)', 
                'normal_range': (4.0, 10.0),
                'color': '#DC143C',
                'critical_low': 2.5,
                'unit': '/cumm'
            },
            'Platelet Count Lakh/Cumm 1.50 - 4.5': {
                'ylabel': 'Platelet Count (Lakh/Cumm)', 
                'normal_range': (1.50, 4.5),
                'color': '#9932CC',
                'critical_low': 1.0,
                'unit': 'Lakh/Cumm'
            }
        }
        self.ai_optimizer = AIForecastOptimizer()
        self.ensemble_forecaster = SmartEnsembleForecaster()
    
    def get_best_forecast_method(self, dates, values, param_info, patient_context=None):
        """AI-powered selection of best forecasting method"""
        # Get AI recommendation
        optimization_result = self.ai_optimizer.optimize_forecast_strategy(
            values, param_info, patient_context
        )
        
        recommended_method = "ensemble"  # Default to ensemble
        
        if isinstance(optimization_result, dict) and 'model_recommendation' in optimization_result:
            rec = optimization_result['model_recommendation'].lower()
            if 'prophet' in rec and PROPHET_AVAILABLE:
                recommended_method = "prophet"
            elif 'arima' in rec and ARIMA_AVAILABLE:
                recommended_method = "arima"
            elif any(term in rec for term in ['ensemble', 'ml', 'machine learning']):
                recommended_method = "ensemble"
        
        return recommended_method, optimization_result
    
    def generate_ai_enhanced_forecast(self, dates, values, periods, param_info, patient_context=None):
        """Generate AI-enhanced forecast using best method"""
        # Get AI recommendation for best method
        best_method, ai_insights = self.get_best_forecast_method(dates, values, param_info, patient_context)
        
        # Generate base forecast
        base_predictions = None
        confidence_intervals = None
        
        if best_method == "ensemble":
            result = self.ensemble_forecaster.smart_ensemble_forecast(dates, values, periods)
        elif best_method == "prophet" and PROPHET_AVAILABLE:
            result = self.prophet_forecast(dates, values, periods)
        elif best_method == "arima" and ARIMA_AVAILABLE:
            result = self.arima_forecast(values, periods)
        else:
            # Fallback to ensemble
            result = self.ensemble_forecaster.smart_ensemble_forecast(dates, values, periods)
        
        if result and result[0] is not None:
            base_predictions, confidence_intervals, _ = result
            
            # Enhance predictions with AI
            enhanced_predictions = self.ai_optimizer.enhance_predictions_with_ai(
                base_predictions, values, param_info['ylabel']
            )
            
            return enhanced_predictions, confidence_intervals, best_method, ai_insights
        
        return None, None, "none", ai_insights
    
    def prophet_forecast(self, dates, values, periods=4):
        """Prophet forecast implementation"""
        if not PROPHET_AVAILABLE or len(values) < 5:
            return None, None, None
        try:
            prophet_df = pd.DataFrame({'ds': dates, 'y': values}).dropna()
            if len(prophet_df) < 5:
                return None, None, None
            
            model = Prophet(
                growth='linear',
                seasonality_mode='additive',
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=periods, freq='W')
            forecast = model.predict(future)
            
            predictions = forecast['yhat'].iloc[-periods:].values
            lower_bound = forecast['yhat_lower'].iloc[-periods:].values
            upper_bound = forecast['yhat_upper'].iloc[-periods:].values
            
            return predictions, (lower_bound, upper_bound), predictions
        except:
            return None, None, None
    
    def arima_forecast(self, values, periods=4):
        """ARIMA forecast implementation"""
        if not ARIMA_AVAILABLE or len(values) < 8:
            return None, None, None
        try:
            clean_values = pd.Series(values).dropna()
            if len(clean_values) < 8:
                return None, None, None
            
            model = ARIMA(clean_values, order=(1, 1, 1))
            fitted_model = model.fit()
            
            forecast_result = fitted_model.forecast(steps=periods)
            predictions = np.array(forecast_result)
            
            margin = np.abs(predictions) * 0.15
            lower_bound = predictions - margin
            upper_bound = predictions + margin
            
            return predictions, (lower_bound, upper_bound), predictions
        except:
            return None, None, None

def create_ai_dashboard():
    st.title("✨ AI-Enhanced Blood Test Forecasting System")
    st.markdown("*Leveraging AI and GenAI for Intelligent Medical Forecasting*")
    
    # Check API key status - only show error if fails
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found! Please check your .env file.")
        st.info("Make sure you have a .env file in your project directory with: GEMINI_API_KEY=your_api_key_here")
    
    forecaster = AIBloodTestForecaster()
    
    # Sidebar
    with st.sidebar:
        st.header("🔬 AI Configuration")
        uploaded_file = st.file_uploader("Upload Blood Test Report", type=['xlsx', 'xls'])
        prediction_weeks = st.slider("Forecast Horizon (Weeks)", 1, 12, 4)
        
        st.subheader("🧠 AI Features")
        enable_ai_enhancement = st.checkbox("AI-Enhanced Predictions", value=True)
        enable_clinical_insights = st.checkbox("Clinical AI Insights", value=True)
        show_forecast_strategy = st.checkbox("Show AI Strategy", value=True)
        
        st.subheader("👤 Patient Context (Optional)")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=0)
        patient_conditions = st.text_area("Medical Conditions/Notes", height=80)
    
    # Prepare patient context string
    patient_context = None
    if patient_age > 0 or patient_conditions.strip():
        patient_context = f"Patient Age: {patient_age if patient_age > 0 else 'Not specified'}"
        if patient_conditions.strip():
            patient_context += f"\nMedical Conditions/Notes: {patient_conditions.strip()}"
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
            
            for param in forecaster.parameters.keys():
                if param in df.columns:
                    df[param] = pd.to_numeric(df[param], errors='coerce')
            
            # Dashboard metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Records", len(df))
            with col2:
                days_range = (df['Date'].max() - df['Date'].min()).days
                st.metric("📅 Data Span", f"{days_range} days")
            with col3:
                st.metric("🔍 Latest Test", df['Date'].max().strftime('%Y-%m-%d'))
            with col4:
                available_params = sum(1 for param in forecaster.parameters.keys() if param in df.columns)
                st.metric("🧪 Parameters", available_params)
            
            # Show patient context if provided
            if patient_context:
                st.info(f"👤 **Patient Context:** {patient_context.replace(chr(10), ' | ')}")
            
            # Main forecasting section
            st.header("💫 AI-Powered Forecasting Results")
            
            for param_name, param_info in forecaster.parameters.items():
                if param_name not in df.columns:
                    continue
                
                param_data = df[['Date', param_name]].dropna()
                if len(param_data) < 4:
                    st.warning(f"Insufficient data for {param_info['ylabel']} (need at least 4 points)")
                    continue
                
                dates = param_data['Date']
                values = param_data[param_name]
                
                st.subheader(f"🩸 {param_info['ylabel']}")
                
                # Generate AI-enhanced forecast with patient context
                with st.spinner(f"AI is analyzing {param_info['ylabel']} patterns..."):
                    enhanced_predictions, confidence_intervals, best_method, ai_insights = forecaster.generate_ai_enhanced_forecast(
                        dates, values, prediction_weeks, param_info, patient_context
                    )
                
                if enhanced_predictions is not None:
                    # Create visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=dates, y=values,
                        mode='lines+markers',
                        name='Historical Data',
                        line=dict(color=param_info['color'], width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Normal range
                    lower_normal, upper_normal = param_info['normal_range']
                    fig.add_hrect(
                        y0=lower_normal, y1=upper_normal,
                        fillcolor="rgba(0,255,0,0.1)",
                        annotation_text="Normal Range",
                        annotation_position="top left"
                    )
                    
                    # Critical threshold
                    fig.add_hline(
                        y=param_info['critical_low'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Critical Level"
                    )
                    
                    # AI-enhanced predictions
                    future_dates = [dates.max() + timedelta(weeks=i) for i in range(1, prediction_weeks + 1)]
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=enhanced_predictions,
                        mode='lines+markers',
                        name=f'AI-Enhanced Forecast ({best_method.title()})',
                        line=dict(color='#FF6B6B', width=3, dash='dot'),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    
                    # Confidence intervals
                    if confidence_intervals:
                        lower_ci, upper_ci = confidence_intervals
                        fig.add_trace(go.Scatter(
                            x=future_dates + future_dates[::-1],
                            y=list(upper_ci) + list(lower_ci[::-1]),
                            fill='toself',
                            fillcolor='rgba(255, 107, 107, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval',
                            showlegend=True
                        ))
                    
                    fig.update_layout(
                        title=f"AI-Enhanced Forecast: {param_info['ylabel']}",
                        xaxis_title="Date",
                        yaxis_title=f"{param_info['ylabel']} ({param_info['unit']})",
                        hovermode='x unified',
                        height=500,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        current_val = values.iloc[-1]
                        predicted_val = enhanced_predictions[-1]
                        trend = "↗️" if predicted_val > current_val else "↘️" if predicted_val < current_val else "➡️"
                        st.metric(
                            "Current → Predicted", 
                            f"{current_val:.2f} → {predicted_val:.2f}",
                            delta=f"{predicted_val - current_val:.2f}",
                            delta_color="inverse" if predicted_val < param_info['critical_low'] else "normal"
                        )
                    
                    with col2:
                        risk_level = "🔴 HIGH" if min(enhanced_predictions) < param_info['critical_low'] else \
                                   "🟡 MEDIUM" if min(enhanced_predictions) < param_info['normal_range'][0] else "🟢 LOW"
                        st.metric("Risk Level", risk_level)
                    
                    with col3:
                        st.metric("AI Method", best_method.title())
                    
                    # AI Strategy insights
                    if show_forecast_strategy and isinstance(ai_insights, dict):
                        with st.expander("🧠 AI Forecasting Strategy", expanded=False):
                            for key, value in ai_insights.items():
                                if key != 'error':
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    
                    # Clinical insights - made collapsed by default
                    if enable_clinical_insights:
                        with st.expander("🏥 Clinical AI Insights", expanded=False):
                            clinical_insights = forecaster.ai_optimizer.generate_clinical_insights(
                                enhanced_predictions, param_info, patient_context
                            )
                            st.markdown(clinical_insights)
                
                else:
                    st.error(f"Unable to generate forecast for {param_info['ylabel']}")
                
                st.markdown("---")
            
            # Overall health trend summary
            st.header("📈 Overall Health Trend Analysis")
            
            # Calculate risk scores for each parameter
            risk_scores = []
            for param_name, param_info in forecaster.parameters.items():
                if param_name in df.columns:
                    param_data = df[param_name].dropna()
                    if len(param_data) > 0:
                        latest_val = param_data.iloc[-1]
                        normal_min, normal_max = param_info['normal_range']
                        
                        if latest_val < param_info['critical_low']:
                            risk_score = 3  # High risk
                        elif latest_val < normal_min:
                            risk_score = 2  # Medium risk
                        elif latest_val > normal_max:
                            risk_score = 1  # Mild concern
                        else:
                            risk_score = 0  # Normal
                        
                        risk_scores.append({
                            'Parameter': param_info['ylabel'],
                            'Current Value': latest_val,
                            'Risk Score': risk_score,
                            'Status': ['Normal', 'Mild Concern', 'Medium Risk', 'High Risk'][risk_score]
                        })
            
            if risk_scores:
                risk_df = pd.DataFrame(risk_scores)
                
                col1, col2 = st.columns(2)
                with col1:
                    overall_risk = np.mean([r['Risk Score'] for r in risk_scores])
                    risk_color = ['🟢', '🟡', '🟠', '🔴'][min(int(overall_risk), 3)]
                    st.metric("Overall Health Score", f"{risk_color} {4-overall_risk:.1f}/4")
                
                with col2:
                    high_risk_count = sum(1 for r in risk_scores if r['Risk Score'] >= 2)
                    st.metric("Parameters Needing Attention", high_risk_count)
                
                st.dataframe(risk_df, use_container_width=True)
            
            # Medical disclaimer
            st.warning("""
            **⚠️ Important Medical Disclaimer:** 
            This AI-powered analysis is for informational and research purposes only. 
            Predictions are based on statistical models and should not replace professional medical advice. 
            Always consult qualified healthcare professionals for medical decisions and treatment plans.
            """)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure your Excel file follows the expected format.")
    
    else:
        st.info("📁 Upload your BloodTestReport.xlsx file to begin AI-powered forecasting")
        
        # Show expected format
        st.subheader("📋 Expected Data Format")
        sample_data = {
            'Date': ['2025-05-23', '2025-05-04', '2025-04-21'],
            'Hemoglobin g/dL 12.0 - 15.0': [9.7, 8.3, 8.6],
            'RBC Count mill/cmm 3.80 - 4.80': [2.67, 2.32, 2.39],
            'Leukocyte Count (TLC/WBC) /cumm 4.00 - 10.00': [4.29, 2.29, 3.02],
            'Platelet Count Lakh/Cumm 1.50 - 4.5': [0.75, 0.92, 1.52]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    create_ai_dashboard()
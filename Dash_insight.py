import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools import PythonREPLTool
import google.generativeai as genai
from PIL import Image
import io
import json
import re
import os
import glob
import numpy as np
from datetime import datetime
import warnings
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
warnings.filterwarnings('ignore')

# State schema for LangGraph
class AnalysisState(TypedDict):
    df: pd.DataFrame
    metric: str
    current_month: str
    prev_months: List[str]
    group_columns: List[str]
    agg_func: str
    prompt: str
    delta_df: pd.DataFrame
    insight: str
    validation: str
    executive_summary: str
    user_question: str
    qna_answer: str

# Dynamic Analysis Functions
def detect_column_types(df, exclude_columns=['month_year']):
    """
    Automatically detect categorical and numerical columns from the dataframe
    
    Parameters:
    df (pd.DataFrame): The dataframe to analyze
    exclude_columns (list): Columns to exclude from analysis
    
    Returns:
    dict: Dictionary with 'categorical' and 'numerical' column lists
    """
    categorical_columns = []
    numerical_columns = []
    
    for col in df.columns:
        if col not in exclude_columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_columns.append(col)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # If numeric but few unique values, treat as categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1 and df[col].nunique() <= 20:
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
    
    return {
        'categorical': categorical_columns,
        'numerical': numerical_columns
    }

def determine_analysis_strategy(df, column_types):
    """
    Determine the best analysis strategy based on data characteristics
    
    Parameters:
    df (pd.DataFrame): The dataframe to analyze
    column_types (dict): Dictionary with column type information
    
    Returns:
    dict: Analysis strategy with grouping columns and aggregation function
    """
    categorical_cols = column_types['categorical']
    numerical_cols = column_types['numerical']
    
    # Determine grouping columns (prioritize low cardinality categorical columns)
    group_columns = []
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= 50:  # Only include columns with reasonable cardinality
            group_columns.append(col)
    
    # Limit to top 3 most important categorical columns to avoid explosion
    if len(group_columns) > 3:
        # Sort by cardinality and take top 3
        group_columns = sorted(group_columns, key=lambda x: df[x].nunique())[:3]
    
    # Determine aggregation function based on data characteristics
    if numerical_cols:
        # If we have numerical columns, use sum for meaningful aggregation
        agg_func = "sum"
        metric = "count"  # Still use count as the base metric
    else:
        # If only categorical, use count
        agg_func = "count"
        metric = "count"
    
    return {
        'group_columns': group_columns,
        'agg_func': agg_func,
        'metric': metric
    }

def setup_dynamic_analysis_state(df, event_type, target_month_str, prev_months):
    """
    Setup analysis state with dynamic column detection and strategy
    
    Parameters:
    df (pd.DataFrame): The aggregated dataframe
    event_type (str): 'spike' or 'dip'
    target_month_str (str): Target month for analysis
    prev_months (list): Previous months for comparison
    
    Returns:
    dict: Complete initial state for LangGraph
    """
    # Detect column types
    column_types = detect_column_types(df)
    
    # Determine analysis strategy
    strategy = determine_analysis_strategy(df, column_types)
    
    # Create initial state
    initial_state = {
        "df": df,
        "metric": strategy['metric'],
        "current_month": target_month_str,
        "prev_months": prev_months,
        "group_columns": strategy['group_columns'],
        "agg_func": strategy['agg_func'],
        "user_question": f"Analyze the {event_type} in {target_month_str} vs previous months",
        "prompt": "",
        "delta_df": pd.DataFrame(),
        "insight": "",
        "validation": "",
        "executive_summary": "",
        "qna_answer": ""
    }
    
    return initial_state, column_types, strategy

# CSV Aggregation Function
def aggregate_csv_data(csv_file_path, columns_to_use):
    """
    Aggregate CSV data by month and categorical columns
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    columns_to_use (list): List of column names to use from the CSV
    
    Returns:
    pandas.DataFrame: Aggregated data with counts and averages
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None
    
    # Check if specified columns exist
    missing_columns = [col for col in columns_to_use if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        return None
    
    # Select only specified columns
    df = df[columns_to_use].copy()
    
    # Auto-detect date column (look for date-related keywords)
    date_column = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year', 'created', 'create']):
            date_column = col
            break
    
    if date_column is None:
        st.error("No date column found. Column names should contain 'date', 'time', 'month', 'year', 'created', etc.")
        return None
    
    # Convert date column to datetime
    def convert_to_datetime(series):
        try:
            return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        except:
            return pd.to_datetime(series, errors='coerce')
    
    df[date_column] = convert_to_datetime(df[date_column])
    
    # Remove rows with invalid dates
    initial_count = len(df)
    df = df.dropna(subset=[date_column])
    removed_count = initial_count - len(df)
    
    if df.empty:
        st.error("No valid dates found")
        return None
    
    # Create month-year column
    df['month_year'] = df[date_column].dt.to_period('M')
    
    # Auto-detect categorical and numerical columns
    categorical_columns = []
    numerical_columns = []
    
    for col in df.columns:
        if col not in [date_column, 'month_year']:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_columns.append(col)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # If numeric but few unique values, treat as categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() <= 50:
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
    
    if not categorical_columns:
        st.error("Need at least one categorical column")
        return None
    
    # Create the final result DataFrame
    all_combinations = []
    
    # Handle missing values for all categorical columns at once
    df_temp = df.copy()
    for cat_col in categorical_columns:
        df_temp[cat_col] = df_temp[cat_col].fillna('Unknown')
    
    # Generate all combinations: month_year + all categorical columns
    for month in df_temp['month_year'].unique():
        month_data = df_temp[df_temp['month_year'] == month]
        
        # Get all unique combinations of categorical values for this month
        if len(categorical_columns) == 1:
            # Single categorical column
            cat_col = categorical_columns[0]
            for category_value in month_data[cat_col].unique():
                category_data = month_data[month_data[cat_col] == category_value]
                
                # Create row for this combination
                row = {
                    'month_year': str(month),
                    cat_col: category_value,
                    'count': len(category_data)
                }
                
                # Add averages for all numerical columns
                for num_col in numerical_columns:
                    if len(category_data) > 0:
                        avg_value = category_data[num_col].mean()
                        row[f'{num_col}_avg'] = round(avg_value, 2)
                    else:
                        row[f'{num_col}_avg'] = 0
                
                all_combinations.append(row)
        
        else:
            # Multiple categorical columns - get all combinations
            unique_combinations = month_data[categorical_columns].drop_duplicates()
            
            for _, combo_row in unique_combinations.iterrows():
                # Filter data for this specific combination
                category_data = month_data.copy()
                for cat_col in categorical_columns:
                    category_data = category_data[category_data[cat_col] == combo_row[cat_col]]
                
                # Create row for this combination
                row = {
                    'month_year': str(month),
                    'count': len(category_data)
                }
                
                # Add categorical values
                for cat_col in categorical_columns:
                    row[cat_col] = combo_row[cat_col]
                
                # Add averages for all numerical columns
                for num_col in numerical_columns:
                    if len(category_data) > 0:
                        avg_value = category_data[num_col].mean()
                        row[f'{num_col}_avg'] = round(avg_value, 2)
                    else:
                        row[f'{num_col}_avg'] = 0
                
                all_combinations.append(row)
    
    # Create DataFrame from all combinations
    if all_combinations:
        final_df = pd.DataFrame(all_combinations)
        
        # Sort by month_year and categorical columns
        sort_columns = ['month_year'] + categorical_columns
        final_df = final_df.sort_values(sort_columns).reset_index(drop=True)
        
        return final_df
    
    else:
        st.error("No results generated")
        return None

# LangGraph Agent Functions
def setup_llm():
    """Setup the LLM for all agents"""
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_base="http://localhost:5436/v1",
        openai_api_key="dummy",
        openai_organization="1e8f80c8-5285-4ef8-a69f-b6ea45de1f8a",
        default_headers={
            "Rpc-Service": "genai-api",
            "Rpc-Caller": "english_assessment_tool-llm-staging"
        },
        temperature=0
    )

def call_llm(prompt):
    """Helper function to call LLM"""
    llm = setup_llm()
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {e}"

def enhanced_prompt_agent_tool(state):
    """Enhanced PromptAgent: Creates strategy with dynamic analysis"""
    df = state['df']
    metric = state['metric']
    current = state['current_month']
    prev_months = state['prev_months']
    group_columns = state['group_columns']
    
    # Detect column types
    column_types = detect_column_types(df)
    
    # Check if we're analyzing a spike or dip
    user_question = state.get('user_question', '')
    
    if 'dip' in user_question.lower():
        event_type = 'dips'
        event_action = 'decrease'
    elif 'spike' in user_question.lower():
        event_type = 'spikes'
        event_action = 'increase'
    else:
        event_type = 'spikes'
        event_action = 'increase'
    
    # Create enhanced prompt with data context
    prompt = f"""
    Analyze {event_type} in '{metric}' for {current} vs average of {prev_months}.
    
    Available data structure:
    - Categorical columns: {column_types['categorical']}
    - Numerical columns: {column_types['numerical']}
    - Current grouping: {', '.join(group_columns)}
    - Total records: {len(df)}
    
    Task: Identify segments with significant {event_action} and recommend:
    1. Best aggregation method (sum/count) based on data characteristics
    2. Key insights about the {event_type}
    3. Potential root causes
    
    Focus on actionable insights with supporting data.
    """
    
    strategy = call_llm(prompt)
    state['prompt'] = strategy
    
    # Enhanced aggregation detection
    strategy_lower = strategy.lower()
    if any(word in strategy_lower for word in ['sum', 'total', 'aggregate', 'cumulative']):
        state['agg_func'] = 'sum'
    elif any(word in strategy_lower for word in ['count', 'frequency', 'occurrences']):
        state['agg_func'] = 'count'
    else:
        # Default based on data characteristics
        if column_types['numerical']:
            state['agg_func'] = 'sum'
        else:
            state['agg_func'] = 'count'
    
    return state

def insight_agent_tool(state):
    """InsightAgent: Analyzes delta data and provides insights dynamically for spikes/dips"""
    delta_df = state['delta_df']
    metric = state['metric']
    user_question = state.get('user_question', '')
    
    # Determine if we're analyzing a spike or dip
    if 'dip' in user_question.lower():
        event_type = 'dips'
        change_word = 'decreases'
        direction = 'negative'
    elif 'spike' in user_question.lower():
        event_type = 'spikes'
        change_word = 'increases'
        direction = 'positive'
    else:
        # Default to spikes
        event_type = 'spikes'
        change_word = 'increases'
        direction = 'positive'
    
    prompt = (
        f"Here's delta data:\n{delta_df.to_markdown()}\n"
        f"Which segments saw significant {change_word} in {metric}? "
        f"Focus on {direction} delta values for {event_type} analysis. Explain the key findings."
    )
    
    state['insight'] = call_llm(prompt)
    return state

def df_agent_tool(state):
    """DfAgent: Performs DataFrame analysis and calculations"""
    df, metric, selected_month, prev_months, group_cols, agg_func = (
        state['df'], state['metric'], state['current_month'],
        state['prev_months'], state['group_columns'], state.get('agg_func', 'count')
    )
    df['month_year'] = df['month_year'].astype(str)
    prev = df[df['month_year'].isin(prev_months)]
    curr = df[df['month_year'] == selected_month]

    if agg_func == 'sum':
        prev_agg = prev.groupby(group_cols)[metric].sum().reset_index().rename(columns={metric: 'Prev_Avg'})
        curr_agg = curr.groupby(group_cols)[metric].sum().reset_index().rename(columns={metric: 'Current'})
    else:
        prev_agg = prev.groupby(group_cols)[metric].count().reset_index().rename(columns={metric: 'Prev_Avg'})
        curr_agg = curr.groupby(group_cols)[metric].count().reset_index().rename(columns={metric: 'Current'})

    merged = pd.merge(prev_agg, curr_agg, on=group_cols, how='outer').fillna(0)
    merged['Delta'] = merged['Current'] - merged['Prev_Avg']
    state['delta_df'] = merged
    return state

def validator_agent_tool(state):
    """ValidatorAgent: Validates insights for clarity and data backing"""
    insight = state['insight']
    validation = call_llm(f"Validate this insight: '{insight}'. Is it clear and data-backed? Reply PASS/FAIL.")
    state['validation'] = validation
    return state

def rewrite_agent_tool(state):
    """RewriteAgent: Rewrites insights for executive summary with action points"""
    insight = state['insight']
    state['executive_summary'] = call_llm(f"Rewrite this for executives with action points: '{insight}'")
    return state

def qna_agent_tool(state):
    """QnAAgent: Answers user questions with supporting data"""
    question = state['user_question']
    delta_df, insight = state['delta_df'], state['insight']
    answer = call_llm(
        f"User Q: '{question}'\nInsight: '{insight}'\nData:\n{delta_df.to_markdown()}\nAnswer with supporting data."
    )
    state['qna_answer'] = answer
    return state

# Setup LangGraph
def setup_analysis_graph():
    """Setup the LangGraph for analysis with enhanced prompt agent"""
    graph_builder = StateGraph(AnalysisState)
    
    # Add nodes (using enhanced prompt agent)
    graph_builder.add_node("PromptAgent", enhanced_prompt_agent_tool)
    graph_builder.add_node("DfAgent", df_agent_tool)
    graph_builder.add_node("InsightAgent", insight_agent_tool)
    graph_builder.add_node("ValidatorAgent", validator_agent_tool)
    graph_builder.add_node("RewriteAgent", rewrite_agent_tool)
    graph_builder.add_node("QnAAgent", qna_agent_tool)
    
    # Add edges
    graph_builder.add_edge("PromptAgent", "DfAgent")
    graph_builder.add_edge("DfAgent", "InsightAgent")
    graph_builder.add_edge("InsightAgent", "ValidatorAgent")
    graph_builder.add_edge("ValidatorAgent", "RewriteAgent")
    graph_builder.add_edge("RewriteAgent", "QnAAgent")
    
    # Set entry point
    graph_builder.set_entry_point("PromptAgent")
    
    return graph_builder.compile()

# Page config
st.set_page_config(
    page_title="Chart Analysis & Data Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {
        background-color:#FFFFFF !important;
        color:  #000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stApp {
        background-color: #FFFFFF;
    }
    
    .analysis-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 24px;
        margin: 16px 0;
        color: #333333;
        text-align: center;
    }
    
    .chart-summary {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
        color: #333333;
    }
    
    .stButton > button {
        background-color: #007bff;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 12px 24px;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
    }
    
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 6px;
        color: #333333;
        padding: 12px;
    }
    
    .stSelectbox > div > div > div {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #ced4da;
        border-radius: 6px;
    }
    
    .image-container {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        background-color: #ffffff;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3 {
        color: #212529;
        font-weight: 600;
    }
    
    .stDataFrame {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)
import time

# Custom CSS for floating chat button and chat interface
def add_chatbot_css():
    st.markdown("""
    <style>
    /* Floating Chat Button */
    .chat-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        transition: all 0.3s ease;
        border: none;
    }
    
    .chat-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .chat-icon {
        color: white;
        font-size: 24px;
    }
    
    /* Chat Window */
    .chat-window {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 350px;
        height: 500px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        z-index: 1001;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        animation: slideUp 0.3s ease;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chat-close {
        background: none;
        border: none;
        color: white;
        font-size: 18px;
        cursor: pointer;
        padding: 5px;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .chat-close:hover {
        background: rgba(255,255,255,0.2);
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 15px;
        background: #f8f9fa;
    }
    
    .message {
        margin-bottom: 10px;
        padding: 8px 12px;
        border-radius: 12px;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .user-message {
        background: #007bff;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .bot-message {
        background: white;
        color: #333;
        border: 1px solid #e9ecef;
    }
    
    .chat-input-container {
        padding: 15px;
        background: white;
        border-top: 1px solid #e9ecef;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        background: white;
        border-radius: 12px;
        margin-bottom: 10px;
        border: 1px solid #e9ecef;
        max-width: 80px;
    }
    
    .typing-dots {
        display: flex;
        gap: 3px;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        background: #999;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
    
    /* Hide Streamlit elements in chat */
    .chat-window .stTextInput > div > div > input {
        border-radius: 20px;
        border: 1px solid #ddd;
        padding: 10px 15px;
    }
    
    .chat-window .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 20px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for chat
def initialize_chat_state():
    if 'chat_open' not in st.session_state:
        st.session_state.chat_open = False
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "bot", "content": "Hi! I'm your support Agent. How can I help you with the Dash Data?"}
        ]
    if 'is_typing' not in st.session_state:
        st.session_state.is_typing = False

# Simulate bot response (replace with your actual chatbot logic)
def get_bot_response(user_message):
    """Replace this function with your actual chatbot/API integration"""
    responses = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi there! What can I help you with?",
        "help": "I can help you with ticket analysis, data insights, and answering questions about your support system.",
        "ticket": "I can help you analyze tickets, check response times, or provide insights about your support data.",
        "data": "I can help you understand your data, create reports, and provide analytical insights.",
        "chart": "I can help you analyze charts, understand trends, and explain data visualizations.",
        "report": "I can help you generate reports, analyze metrics, and provide business insights.",
        "bye": "Goodbye! Feel free to reach out if you need any assistance.",
        "thanks": "You're welcome! Is there anything else I can help you with?",
        "default": "I understand you're asking about: '{}'. Let me help you with that. Could you provide more details?".format(user_message)
    }
    
    user_message_lower = user_message.lower()
    for key in responses:
        if key in user_message_lower:
            return responses[key]
    return responses["default"]

# Chat interface component - RIGHT CORNER POPUP WITH PROPER UI
def render_chat_interface():
    if st.session_state.chat_open:
        # Create a proper modal/popup using Streamlit's dialog
        @st.dialog("🤖 Analytics Assistant")
        def chat_modal():
            # Messages display area
            # st.markdown("### Conversation")
            
            # Create scrollable message container
            with st.container(height=300):
                for message in st.session_state.chat_messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div style="text-align: right; margin: 8px 0;">
                            <div style="background: #007bff; color: white; padding: 10px 15px; border-radius: 18px; display: inline-block; max-width: 80%; font-size: 14px; box-shadow: 0 2px 4px rgba(0,123,255,0.3);">
                                {message["content"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="text-align: left; margin: 8px 0;">
                            <div style="background: #f8f9fa; color: #333; padding: 10px 15px; border-radius: 18px; display: inline-block; max-width: 80%; font-size: 14px; border: 1px solid #e9ecef; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                {message["content"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Typing indicator
                if st.session_state.is_typing:
                    st.markdown("""
                    <div style="text-align: left; margin: 8px 0;">
                        <div style="background: #f8f9fa; color: #666; padding: 10px 15px; border-radius: 18px; display: inline-block; font-size: 14px; border: 1px solid #e9ecef; font-style: italic;">
                            🤖 Typing<span class="dots">...</span>
                        </div>
                    </div>
                    <style>
                    .dots { animation: blink 1.5s infinite; }
                    @keyframes blink { 0%, 50% { opacity: 0; } 51%, 100% { opacity: 1; } }
                    </style>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Input section - PROPERLY CONTAINED IN MODAL
            st.markdown("###  Send a Message")
            
            # Use columns for better layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_input = st.text_input(
                    label="Your message",
                    placeholder="Type your message here...",
                    key="modal_chat_input",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_clicked = st.button(" Send", key="modal_send", use_container_width=True, type="primary")
            
            # # Quick buttons row
            # st.markdown("**Quick Actions:**")
            # col1, col2, col3 = st.columns(3)
            
            # with col1:
            #     if st.button("👋 Hello", key="quick_hello", use_container_width=True):
            #         st.session_state.chat_messages.append({"role": "user", "content": "hello"})
            #         st.session_state.is_typing = True
            #         st.rerun()
            
            # with col2:
            #     if st.button("❓ Help", key="quick_help_modal", use_container_width=True):
            #         st.session_state.chat_messages.append({"role": "user", "content": "help"})
            #         st.session_state.is_typing = True
            #         st.rerun()
            
            # with col3:
            #     if st.button("🗑️ Clear", key="quick_clear_modal", use_container_width=True):
            #         st.session_state.chat_messages = [
            #             {"role": "bot", "content": "Hi! I'm your support assistant. How can I help you today?"}
            #         ]
            #         st.rerun()
            
            # Handle text input send
            if send_clicked and user_input.strip():
                st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})
                st.session_state.is_typing = True
                st.rerun()
            
            # Close button
            st.markdown("---")
            if st.button(" Close Chat", key="modal_close", use_container_width=True):
                st.session_state.chat_open = False
                st.rerun()
            
            # Message count
            st.caption(f" {len(st.session_state.chat_messages)} messages in this conversation")
        
        # Show the modal
        chat_modal()
        
        # Process bot response
        if st.session_state.is_typing:
            time.sleep(1)
            bot_response = get_bot_response(st.session_state.chat_messages[-1]["content"])
            st.session_state.chat_messages.append({"role": "bot", "content": bot_response})
            st.session_state.is_typing = False
            st.rerun()

# Alternative: Floating chat button with better positioning
def render_chat_button():
    if not st.session_state.chat_open:
        # Create floating button container
        st.markdown("""
        <div style="position: fixed; bottom: 20px; right: 20px; z-index: 999;">
        """, unsafe_allow_html=True)
        
        if st.button("💬", key="floating_chat_btn", help="Open Support Chat"):
            st.session_state.chat_open = True
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add CSS to style the floating button
        st.markdown("""
        <style>
        /* Style the floating chat button */
        div[data-testid="stButton"] button[title="Open Support Chat"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 50% !important;
            width: 60px !important;
            height: 60px !important;
            font-size: 24px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-testid="stButton"] button[title="Open Support Chat"]:hover {
            transform: scale(1.1) !important;
            box-shadow: 0 6px 20px rgba(0,0,0,0.4) !important;
        }
        
        /* Make modal wider */
        div[data-testid="stModal"] > div {
            width: 90% !important;
            max-width: 500px !important;
        }
        </style>
        """, unsafe_allow_html=True)

# Floating chat button
def render_chat_button():
    if not st.session_state.chat_open:
        if st.button("💬", key="chat_button", help="Open Chat Assistant"):
            st.session_state.chat_open = True
            st.rerun()
        
        # Add the floating button CSS
        st.markdown("""
        <script>
        // Move the chat button to floating position
        setTimeout(function() {
            const button = document.querySelector('[data-testid="stButton"] button[title="Open Chat Assistant"]');
            if (button) {
                button.className = 'chat-button';
                button.innerHTML = '<span class="chat-icon">💬</span>';
                button.style.position = 'fixed';
                button.style.bottom = '20px';
                button.style.right = '20px';
                button.style.zIndex = '1000';
            }
        }, 100);
        </script>
        """, unsafe_allow_html=True)

# Handle close chat functionality
def handle_close_chat():
    if st.session_state.get('close_chat'):
        st.session_state.chat_open = False
        st.session_state.close_chat = False
        st.rerun()

# Main chatbot component to add to your page
def add_chatbot():
    """Main function to add chatbot to any Streamlit page"""
    initialize_chat_state()
    add_chatbot_css()
    handle_close_chat()
    render_chat_interface()
    render_chat_button()
def get_screenshot_files(folder_path):
    """Get all image files from the specified folder"""
    if not os.path.exists(folder_path):
        return []
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    return sorted(image_files)

def find_csv_in_folder(folder_path):
    """Find CSV file in the same folder as images"""
    if not os.path.exists(folder_path):
        return None
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if csv_files:
        return csv_files[0]
    return None

class ChartAnalyzer:
    def __init__(self, api_key):
        """Initialize the chart analyzer with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def analyze_chart_with_gemini(self, image_bytes):
        """Analyze chart using Gemini to extract spikes and dips"""
        image = Image.open(io.BytesIO(image_bytes))
        
        prompt = """
        Analyze the line chart in the uploaded image that may contain multiple series/legends. For each series, provide a structured summary using this exact JSON format:
        {
          "x_axis": "Label and range of X axis",
          "y_axis": "Label and range of Y axis",
          "metric_failed": "The key metric name represented on the Y axis",
          "series": [
            {
              "legend": "Series Name",
              "spikes_and_dips": [
                {
                  "event": "Spike/Dip from X=Date1 to X=Date2",
                  "deviation": "Percentage change (e.g., '50%' or '-20%')"
                }
              ]
            }
          ]
        }
        Rules:
        - For each consecutive data point in each series, calculate the deviation as:
            deviation % = ((current value - previous value) / previous value) * 100
        - Report spikes as positive deviations and dips as negative deviations.
        - Include the time period (date, hour, etc.) for each event in each series.
        - Only return valid JSON (no markdown or extra commentary).
        """
        
        try:
            response = self.model.generate_content([prompt, image], stream=True)
            response.resolve()
            return response.text
        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")
            return None
    
    def process_chart_analysis(self, image_bytes, chart_name):
        """Process chart analysis and extract structured data"""
        result_text = self.analyze_chart_with_gemini(image_bytes)
        
        if not result_text:
            return None
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if not json_match:
                st.error("No valid JSON found in Gemini output")
                return None
            
            clean_json = json_match.group(0)
            data = json.loads(clean_json)
            
            series_data = data.get("series", [])
            spikes_data = []
            dips_data = []
            
            for idx, series in enumerate(series_data):
                for event_idx, event in enumerate(series.get("spikes_and_dips", [])):
                    deviation_str = event.get("deviation", "0%")
                    deviation_value = float(deviation_str.replace('%', '').strip())
                    
                    if abs(deviation_value) >= 0:
                        event_info = {
                            "id": f"event_{idx}_{event_idx}",
                            "series": series.get("legend", "Unknown"),
                            "event": event.get("event", ""),
                            "deviation": deviation_value,
                            "deviation_str": deviation_str,
                            "type": "Spike" if deviation_value > 0 else "Dip"
                        }
                        
                        if deviation_value > 0:
                            spikes_data.append(event_info)
                        else:
                            dips_data.append(event_info)
            
            data["file"] = chart_name
            
            return {
                "chart_data": data,
                "series_data": series_data,
                "spikes_data": spikes_data,
                "dips_data": dips_data
            }
            
        except Exception as e:
            st.error(f"Error parsing chart analysis: {e}")
            return None

def main():
    st.markdown("""
<h2 style='text-align: center; color: #212529; font-weight: 600; margin-bottom: 2rem;'>
    Chart Analysis & Data Insights
</h2>
""", unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'chart_analysis' not in st.session_state:
        st.session_state.chart_analysis = None
    if 'df_aggregated' not in st.session_state:
        st.session_state.df_aggregated = None
    if 'analysis_graph' not in st.session_state:
        st.session_state.analysis_graph = None
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'selected_chart_for_analysis' not in st.session_state:
        st.session_state.selected_chart_for_analysis = None
    
    # Setup analysis graph
    if st.session_state.analysis_graph is None:
        st.session_state.analysis_graph = setup_analysis_graph()
    
    # Navigation sidebar
    st.sidebar.title("Navigation")
    if st.sidebar.button("Home"):
        st.session_state.current_page = 'home'
    if st.sidebar.button("Chart Analysis"):
        st.session_state.current_page = 'chart_analysis'
    if st.sidebar.button("Interactive Analysis"):
        st.session_state.current_page = 'interactive_analysis'
    
    # Route to appropriate page
    if st.session_state.current_page == 'home':
        home_page()
    elif st.session_state.current_page == 'chart_analysis':
        chart_analysis_page()
    elif st.session_state.current_page == 'interactive_analysis':
        interactive_analysis_page()

def home_page():
    """Home page displaying all available charts"""
    folder_path = "ticket_closed5"
    image_files = get_screenshot_files(folder_path)
    
    if image_files:
        cols = st.columns(3)
        add_chatbot()
        for i, image_file in enumerate(image_files):
            col = cols[i % 3]
            
            with col:
                image = Image.open(image_file)
                st.image(image, caption=os.path.basename(image_file), use_container_width=True)
                
                if st.button(f"Analyze", key=f"analyze_{i}"):
                    st.session_state.selected_chart_for_analysis = image_file
                    st.session_state.current_page = 'chart_analysis'
                    st.rerun()
    else:
        st.error(f"No image files found in {folder_path} folder")

def chart_analysis_page():
    """Chart analysis page using LangGraph with full-width analysis results"""
    gemini_api_key = "AIzaSyBjP_udxJe0ozU0OkF6mkubbDI2G7vKm1E"
    folder_path = "ticket_closed5"
    
    # Initialize session state for analysis display
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    if 'selected_event_data' not in st.session_state:
        st.session_state.selected_event_data = None
    if 'selected_event_type' not in st.session_state:
        st.session_state.selected_event_type = None
    
    # Load and aggregate CSV file if not already loaded
    if st.session_state.df_aggregated is None:
        csv_file = find_csv_in_folder(folder_path)
        if csv_file:
            try:
                columns_to_use = ['create_month', 'account_country', 'level_2', 'age_in_hours']
                df_aggregated = aggregate_csv_data(csv_file, columns_to_use)
                
                if df_aggregated is not None:
                    st.session_state.df_aggregated = df_aggregated
                else:
                    st.error("Failed to aggregate data")
                    
            except Exception as e:
                st.error(f"Error loading/aggregating data: {e}")
        else:
            st.error(f"No CSV file found in {folder_path} folder")
    
    # Check if a chart was selected from home page
    if st.session_state.selected_chart_for_analysis:
        selected_image_path = st.session_state.selected_chart_for_analysis
        st.session_state.selected_image = selected_image_path
        
        image = Image.open(selected_image_path)
        st.image(image, caption=f"Analyzing: {os.path.basename(selected_image_path)}", use_container_width=True)
        
        # Auto-analyze the chart
        if st.session_state.df_aggregated is not None and st.session_state.chart_analysis is None:
            with st.spinner("Analyzing chart..."):
                try:
                    chart_analyzer = ChartAnalyzer(gemini_api_key)
                    
                    with open(selected_image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                    
                    analysis_results = chart_analyzer.process_chart_analysis(
                        image_bytes, os.path.basename(selected_image_path)
                    )
                    
                    if analysis_results:
                        st.session_state.chart_analysis = analysis_results
                    else:
                        st.error("Failed to analyze chart")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Show analysis results if triggered
    if st.session_state.show_analysis and st.session_state.selected_event_data:
        st.markdown("---")
        # Full-width analysis section
        analyze_event_with_langgraph(
            st.session_state.selected_event_data, 
            st.session_state.selected_event_type
        )
        
        # Back to selection button
        if st.button("← Back to Event Selection", key="back_to_selection"):
            st.session_state.show_analysis = False
            st.session_state.selected_event_data = None
            st.session_state.selected_event_type = None
            st.rerun()
        
        return  # Exit early to show only analysis
    
    # Event selection section (only show if not analyzing)
    if st.session_state.chart_analysis:
        analysis = st.session_state.chart_analysis
        
        st.subheader("Chart Overview")
        chart_data = analysis["chart_data"]
        st.write(f"**X-axis:** {chart_data['x_axis']}")
        st.write(f"**Y-axis:** {chart_data['y_axis']}")
        
        st.subheader("Detected Events - Click to Analyze")
        col1, col2 = st.columns(2)
        
        # Spikes section
        with col1:
            st.markdown("### **Spikes** ")
            
            if analysis["spikes_data"]:
                spike_options = []
                spike_mapping = {}
                
                for idx, spike in enumerate(analysis["spikes_data"]):
                    display_text = f"{spike['series']}: {spike['event']} ({spike['deviation_str']})"
                    spike_options.append(display_text)
                    spike_mapping[display_text] = spike
                
                selected_spike = st.selectbox(
                    "Select a spike to analyze:",
                    options=["Select a spike..."] + spike_options,
                    key="spike_selector"
                )
                
                if selected_spike != "Select a spike...":
                    if st.button("Analyze Spike", key="analyze_spike_btn", type="primary"):
                        st.session_state.selected_event_data = spike_mapping[selected_spike]
                        st.session_state.selected_event_type = "spike"
                        st.session_state.show_analysis = True
                        st.rerun()
                        
            else:
                st.info("No spikes detected in this chart")
        
        # Dips section
        with col2:
            st.markdown("### **Dips**")
            
            if analysis["dips_data"]:
                dip_options = []
                dip_mapping = {}
                
                for idx, dip in enumerate(analysis["dips_data"]):
                    display_text = f"{dip['series']}: {dip['event']} ({dip['deviation_str']})"
                    dip_options.append(display_text)
                    dip_mapping[display_text] = dip
                
                selected_dip = st.selectbox(
                    "Select a dip to analyze:",
                    options=["Select a dip..."] + dip_options,
                    key="dip_selector"
                )
                
                if selected_dip != "Select a dip...":
                    if st.button("Analyze Dip", key="analyze_dip_btn", type="primary"):
                        st.session_state.selected_event_data = dip_mapping[selected_dip]
                        st.session_state.selected_event_type = "dip"
                        st.session_state.show_analysis = True
                        st.rerun()
                        
            else:
                st.info("No dips detected in this chart")

def interactive_analysis_page():
    """Interactive analysis page using LangGraph"""
    st.header("Interactive Data Analysis")
    st.markdown("Ask questions about your aggregated data and get comprehensive insights")
    
    folder_path = "ticket_closed5"
    
    # Load and aggregate CSV file if not already loaded
    if st.session_state.df_aggregated is None:
        csv_file = find_csv_in_folder(folder_path)
        
        if csv_file:
            try:
                columns_to_use = ['create_month', 'account_country', 'level_2', 'age_in_hours']
                df_aggregated = aggregate_csv_data(csv_file, columns_to_use)
                
                if df_aggregated is not None:
                    st.session_state.df_aggregated = df_aggregated
                else:
                    st.error("Failed to aggregate data")
                    return
                    
            except Exception as e:
                st.error(f"Error loading/aggregating data: {e}")
                return
        else:
            st.error(f"No CSV file found in {folder_path} folder")
            return
    
    # Get initial data summary using LangGraph
    if st.session_state.df_aggregated is not None and st.session_state.data_summary is None:
        with st.spinner("Generating data summary..."):
            try:
                df = st.session_state.df_aggregated
                available_months = sorted(df['month_year'].unique())
                
                # Use dynamic analysis for data summary
                initial_state, _, _ = setup_dynamic_analysis_state(
                    df, "overview", available_months[-1], available_months[-3:-1]
                )
                
                initial_state["user_question"] = "Provide a comprehensive data overview and summary"
                
                result = st.session_state.analysis_graph.invoke(initial_state)
                st.session_state.data_summary = result.get("executive_summary", "No summary available")
                st.session_state.data_summary_states = result  # Store all states
                
            except Exception as e:
                st.error(f"Error generating summary: {e}")
    
    # Display data summary
    if st.session_state.data_summary:
        st.subheader("Data Overview")
        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
        st.write(st.session_state.data_summary)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show agent outputs for data summary
        if st.session_state.get("data_summary_states"):
            states = st.session_state.data_summary_states
            with st.expander("View Data Summary Agent Outputs"):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**PromptAgent:**")
                    st.write(states.get("prompt", "No prompt")[:200] + "...")
                    st.write(f"**Agg Function:** {states.get('agg_func', 'N/A')}")
                
                with col2:
                    st.write("**DfAgent Delta:**")
                    delta_df = states.get("delta_df", pd.DataFrame())
                    if not delta_df.empty:
                        st.dataframe(delta_df.head(3))
                    else:
                        st.write("No data")
                
                with col3:
                    st.write("**InsightAgent:**")
                    insight = states.get("insight", "No insight")
                    st.write(insight[:200] + "..." if len(insight) > 200 else insight)
    
    # Show aggregated data sample
    if st.session_state.df_aggregated is not None:
        st.subheader("Aggregated Data Sample")
        st.dataframe(st.session_state.df_aggregated.head(10))
    
    # User input section
    st.subheader("Ask Your Questions")
    user_question = st.text_input("Ask any question about your data:", key="user_question")
    
    if user_question and st.button("Analyze", key="analyze_question"):
        if st.session_state.df_aggregated is not None:
            with st.spinner(f"Analyzing: {user_question}..."):
                try:
                    df = st.session_state.df_aggregated
                    available_months = sorted(df['month_year'].unique())
                    
                    # Use dynamic analysis for user questions
                    initial_state, _, _ = setup_dynamic_analysis_state(
                        df, "question", available_months[-1], available_months[-3:-1]
                    )
                    
                    initial_state["user_question"] = user_question
                    
                    result = st.session_state.analysis_graph.invoke(initial_state)
                    
                    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
                    st.subheader(f"Analysis: {user_question}")
                    
                    # Show all agent outputs for interactive question
                    st.subheader("Agent Pipeline Outputs")
                    
                    with st.expander("Agent 1: PromptAgent Output"):
                        st.write("**Strategy:**")
                        st.write(result.get("prompt", "No prompt generated"))
                        st.write(f"**Aggregation Function:** {result.get('agg_func', 'Not determined')}")
                    
                    with st.expander("Agent 2: DfAgent Output"):
                        st.write("**Delta DataFrame:**")
                        delta_df = result.get("delta_df", pd.DataFrame())
                        if not delta_df.empty:
                            st.dataframe(delta_df)
                        else:
                            st.write("No delta data generated")
                    
                    with st.expander("Agent 3: InsightAgent Output"):
                        st.write("**Raw Insights:**")
                        st.write(result.get("insight", "No insights generated"))
                    
                    with st.expander("Agent 4: ValidatorAgent Output"):
                        st.write("**Validation Result:**")
                        st.write(result.get("validation", "No validation performed"))
                    
                    with st.expander("Agent 5: RewriteAgent Output"):
                        st.write("**Executive Summary:**")
                        st.write(result.get("executive_summary", "No executive summary generated"))
                    
                    with st.expander("Agent 6: QnAAgent Output"):
                        st.write("**Q&A Answer:**")
                        st.write(result.get("qna_answer", "No Q&A answer generated"))
                    
                    # Final answer
                    st.subheader("Final Answer")
                    st.write(result.get("qna_answer", "No analysis available"))
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.error("Data not loaded. Please refresh the page.")

def extract_target_month_from_event(event_text):
    """Extract the target month from event text with proper year handling"""
    import re
    
    # First try to extract full date patterns (e.g., "Feb 1, 2025 to Mar 1, 2025")
    # We want the FROM month (the spike/dip month), not the TO month
    date_pattern = r'from\s+(\w+)\s+\d+,\s+(\d{4})\s+to'
    match = re.search(date_pattern, event_text, re.IGNORECASE)
    
    if match:
        month_name = match.group(1).lower()
        year = match.group(2)
        
        month_to_number = {
            'jan': '01', 'january': '01',
            'feb': '02', 'february': '02', 
            'mar': '03', 'march': '03',
            'apr': '04', 'april': '04',
            'may': '05', 'may': '05',
            'jun': '06', 'june': '06',
            'jul': '07', 'july': '07',
            'aug': '08', 'august': '08',
            'sep': '09', 'september': '09',
            'oct': '10', 'october': '10',
            'nov': '11', 'november': '11',
            'dec': '12', 'december': '12'
        }
        
        month_number = month_to_number.get(month_name)
        if month_number:
            return f"{year}-{month_number}"
    
    # Fallback: try to extract the FROM month without full date
    from_pattern = r'from\s+(\w+)'
    match = re.search(from_pattern, event_text, re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    # If no FROM pattern found, look for any month names (first occurrence)
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    
    found_months = []
    for month in months:
        if month in event_text.lower():
            found_months.append(month)
    
    if found_months:
        return found_months[0]  # Return first month found, not last
    
    return None

def analyze_event_with_langgraph(event_data, event_type):
    """Analyze a specific spike or dip event using LangGraph - Dynamic Version"""
    if st.session_state.df_aggregated is None:
        st.error("Please ensure aggregated data is loaded first")
        return
    
    # Full-width header
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #212529; margin: 0;">Deep Dive Analysis: {event_data['event']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    event_text = event_data['event']
    target_month = extract_target_month_from_event(event_text)
    event_id = event_data['id']
    
    # Check if analysis already done
    if f"analysis_done_{event_id}" not in st.session_state:
        st.session_state[f"analysis_done_{event_id}"] = False
    if f"analysis_result_{event_id}" not in st.session_state:
        st.session_state[f"analysis_result_{event_id}"] = None
    if f"analysis_states_{event_id}" not in st.session_state:
        st.session_state[f"analysis_states_{event_id}"] = None
    
    if not st.session_state[f"analysis_done_{event_id}"]:
        try:
            with st.spinner("Analyzing event data..."):
                # Get previous months for comparison
                df = st.session_state.df_aggregated
                available_months = sorted(df['month_year'].unique())
                
                # Month processing logic (same as before)
                if target_month:
                    target_month_str = None
                    
                    if '-' in str(target_month) and len(str(target_month)) == 7:
                        if target_month in available_months:
                            target_month_str = target_month
                    else:
                        year_match = re.search(r'(\d{4})', event_text)
                        if year_match:
                            year = year_match.group(1)
                            
                            month_to_number = {
                                'jan': '01', 'january': '01', 'feb': '02', 'february': '02', 
                                'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
                                'may': '05', 'may': '05', 'jun': '06', 'june': '06',
                                'jul': '07', 'july': '07', 'aug': '08', 'august': '08',
                                'sep': '09', 'september': '09', 'oct': '10', 'october': '10',
                                'nov': '11', 'november': '11', 'dec': '12', 'december': '12'
                            }
                            
                            target_month_number = month_to_number.get(target_month.lower())
                            if target_month_number:
                                candidate_month = f"{year}-{target_month_number}"
                                if candidate_month in available_months:
                                    target_month_str = candidate_month
                        
                        if not target_month_str:
                            target_month_number = month_to_number.get(target_month.lower())
                            if target_month_number:
                                for month in reversed(available_months):
                                    if f"-{target_month_number}" in month:
                                        target_month_str = month
                                        break
                    
                    if target_month_str:
                        target_idx = available_months.index(target_month_str)
                        prev_months = available_months[max(0, target_idx-2):target_idx] if target_idx > 0 else available_months[:2]
                    else:
                        target_month_str = available_months[-1]
                        prev_months = available_months[-3:-1] if len(available_months) >= 3 else available_months[:-1]
                else:
                    target_month_str = available_months[-1]
                    prev_months = available_months[-3:-1] if len(available_months) >= 3 else available_months[:-1]
                
                # Setup dynamic analysis state
                initial_state, column_types, strategy = setup_dynamic_analysis_state(
                    df, event_type, target_month_str, prev_months
                )
                
                # Store debug info including dynamic analysis details
                st.session_state[f"debug_info_{event_id}"] = {
                    "event_text": event_text,
                    "target_month": target_month,
                    "available_months": available_months,
                    "target_month_str": target_month_str,
                    "prev_months": prev_months,
                    "detected_categorical": column_types['categorical'],
                    "detected_numerical": column_types['numerical'],
                    "selected_group_columns": strategy['group_columns'],
                    "selected_agg_func": strategy['agg_func'],
                    "analysis_strategy": strategy
                }
                
                # Run LangGraph analysis
                result = st.session_state.analysis_graph.invoke(initial_state)
                
                # Store results
                st.session_state[f"analysis_states_{event_id}"] = result
                st.session_state[f"analysis_result_{event_id}"] = result.get("executive_summary", "No analysis available")
                st.session_state[f"analysis_done_{event_id}"] = True
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            return
    
    # Display debug information in full width
    if st.session_state.get(f"debug_info_{event_id}"):
        debug_info = st.session_state[f"debug_info_{event_id}"]
        
        st.markdown("## Debug Info:")
        
        # Create columns for debug info
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Event text:** {debug_info['event_text']}")
            st.write(f"**Target month:** {debug_info['target_month_str']}")
            st.write(f"**Previous months:** {debug_info['prev_months']}")
            
        with col2:
            st.write(f"**Detected categorical columns:** {debug_info['detected_categorical']}")
            st.write(f"**Detected numerical columns:** {debug_info['detected_numerical']}")
            st.write(f"**Selected grouping:** {debug_info['selected_group_columns']}")
            st.write(f"**Selected aggregation:** {debug_info['selected_agg_func']}")
        
        st.markdown("---")
    
    # Display agent outputs in full width
    if st.session_state.get(f"analysis_states_{event_id}"):
        states = st.session_state[f"analysis_states_{event_id}"]
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Orchestration", 
            "Data Analysis", 
            "Insights", 
            "Validation", 
            "Final Answer"
        ])
        
        with tab1:
            st.markdown("### Strategy & Approach")
            st.write(states.get("prompt", "No prompt generated"))
        
        with tab2:
            st.markdown("### Delta Analysis Results")
            delta_df = states.get("delta_df", pd.DataFrame())
            if not delta_df.empty:
                st.dataframe(delta_df, use_container_width=True)
            else:
                st.write("No delta data generated")
        
        with tab3:
            st.markdown("### Key Insights")
            st.write(states.get("insight", "No insights generated"))
        
        with tab4:
            st.markdown("### Validation Results")
            validation = states.get("validation", "No validation performed")
            if "PASS" in validation:
                st.success(f"✅ {validation}")
            else:
                st.warning(f"⚠️ {validation}")
        
        with tab5:
            st.markdown("### Final Analysis")
            st.write(states.get("qna_answer", "No Q&A answer generated"))

if __name__ == "__main__":
    main()
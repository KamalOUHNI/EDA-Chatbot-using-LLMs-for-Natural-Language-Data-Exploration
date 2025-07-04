import streamlit as st
from src.llm_agent import run_pandas_agent
from langchain_community.utilities import SQLDatabase 
from sqlalchemy import create_engine
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Configure matplotlib for Streamlit
plt.style.use('default')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

original_show = plt.show
def streamlit_show():
    if plt.get_fignums():
        st.pyplot(plt.gcf())
    original_show()

plt.show = streamlit_show

st.title("Eda Chatbot")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'table_loaded' not in st.session_state:
    st.session_state.table_loaded = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Database connection (only for ssms)
params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;" 
    "DATABASE=AdventureWorksDW2019;" 
    "trusted_connection=yes;"
)
connection_uri = f"mssql+pyodbc:///?odbc_connect={params}"## change this to your database connection string.
## it can be postgres, mysql, sqlite, etc.

try:
    db = SQLDatabase.from_uri(connection_uri)
    available_tables = db.get_usable_table_names()
    st.write("Available tables:", available_tables)
    table_name = st.selectbox("Select a table:", [""] + available_tables)
    
    if table_name and not st.session_state.table_loaded:
        try:
            engine = create_engine(connection_uri)
            st.session_state.df = pd.read_sql_table(table_name, con=engine)
            st.session_state.table_loaded = True
            st.success(f"Table '{table_name}' loaded successfully! Shape: {st.session_state.df.shape}")
            st.dataframe(st.session_state.df.head())
        except Exception as e:
            st.error(f"Error loading table: {str(e)}")
    
    if st.session_state.table_loaded and st.session_state.df is not None:
        st.subheader("Chat with your data")
        
        #chat history
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.chat_message("user").markdown(message['content'])
            else:
                st.chat_message("assistant").markdown(message['content'])
        
        prompt = st.chat_input("Ask a question about your data...")
        
        if prompt:
            # Add user message to chat
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            
            try:
                # Get response from pandas agent
                with st.spinner("Analyzing..."):
                    response = run_pandas_agent(prompt, st.session_state.df)
                
                # Add assistant response to chat
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                
            except Exception as e:
                error_msg = f"Error processing your request: {str(e)}"
                st.chat_message("assistant").markdown(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
    
    # reset button
    if st.session_state.table_loaded:
        if st.button("Reset / Load Different Table"):
            st.session_state.df = None
            st.session_state.table_loaded = False
            st.session_state.messages = []
            st.rerun()

except Exception as e:
    st.error(f"Database connection error: {str(e)}")
    st.info("Please check your database connection settings.")
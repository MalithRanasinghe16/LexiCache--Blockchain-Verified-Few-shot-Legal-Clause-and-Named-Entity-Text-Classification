#!/bin/bash

# Activate virtual environment if it exists
if [ -d "env/bin" ]; then
    source env/bin/activate
elif [ -d "env/Scripts" ]; then
    source env/Scripts/activate
fi

# Start Streamlit application
echo "Starting LexiCache Streamlit application..."
streamlit run app/main.py

# Use official lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Pre-install dependencies that are needed during setup of other packages
RUN pip install --upgrade pip && pip install requests beautifulsoup4

# Install remaining project dependencies
RUN pip install -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY streamlit_app.py .

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]

mkdir -p ~/.streamlit/

echo "[theme]
primaryColor=‘#1fc857‘
backgroundColor=‘#100f21‘
secondaryBackgroundColor=‘#1e1d32‘
textColor=‘#ffffff‘
font=‘sans serif‘
[server]
headless=true
enableCORS=false
port=$PORT
" > ~/.streamlit/config.toml
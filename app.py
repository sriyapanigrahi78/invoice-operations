import os
import streamlit as st
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Page Config & Secrets
st.set_page_config(page_title="Invoice Extraction Agent", layout="wide")
st.title("Automated Invoice & Expense Agent")

# Streamlit retrieves your key securely from its Secrets manager
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Vendor or merchant name")
    tax_id: str | None = Field(default=None, description="GSTIN or Tax ID")
    invoice_number: str = Field(description="Invoice or receipt number")
    invoice_date: str | None = Field(default=None, description="YYYY-MM-DD")
    subtotal: float = Field(default=0.0)
    tax_amount: float = Field(default=0.0)
    total_amount: float = Field(description="Final payable total")

# 2. File Ingestion UI
uploaded_file = st.file_uploader(
    "Upload an invoice (Image, PDF, or Excel)", 
    type=["png", "jpg", "jpeg", "xlsx", "csv"]
)

if uploaded_file and api_key:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    
    col1, col2 = st.columns([1, 1])

    # Case A: Tabular Sheets (Excel/CSV)
    if file_ext in ["xlsx", "csv"]:
        df = pd.read_excel(uploaded_file) if file_ext == "xlsx" else pd.read_csv(uploaded_file)
        with col1:
            st.subheader("Spreadsheet Preview")
            st.dataframe(df.head(10))
        with col2:
            st.info("Tabular file detected. Process columns deterministically via pandas.")

    # Case B: Images / Visual Invoices
    elif file_ext in ["png", "jpg", "jpeg"]:
        img = Image.open(uploaded_file)
        with col1:
            st.subheader("Document Preview")
            st.image(img, use_container_width=True)
            
        with col2:
            st.subheader("Agent Extraction & Verification")
            if st.button("Run Extraction Agent"):
                with st.spinner("Analyzing document with Gemini Flash..."):
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[img, "Extract invoice fields accurately according to the schema."],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=InvoiceData,
                            temperature=0.0,
                        )
                    )
                    extracted = InvoiceData.model_validate_json(response.text)
                    
                    # Display structured output
                    st.json(extracted.model_dump())
                    
                    # Verification check
                    expected_total = extracted.subtotal + extracted.tax_amount
                    if abs(expected_total - extracted.total_amount) <= 0.05:
                        st.success("Arithmetic Check Passed: Subtotal + Tax = Total")
                    else:
                        st.error(f"Arithmetic Mismatch! Math says {expected_total:.2f}, document says {extracted.total_amount:.2f}")
elif not api_key:
    st.warning("Please set your GEMINI_API_KEY in Streamlit Secrets or Environment Variables.")

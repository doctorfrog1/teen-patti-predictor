import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os
import joblib

# Machine Learning imports
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

# --- Configuration ---
MODEL_FOLDER_ID = "1CZepfjRZxWV_wfmEQuZLnbj9H2yAS9Ac"
PLAYER_A_FIXED_CARDS_STR = {'J♣', '10♠', '9♠'}
PREDICTION_ROUNDS_CONSIDERED = 10
STREAK_THRESHOLD = 3
OVER_UNDER_BIAS_THRESHOLD = 0.6


PATTERNS_TO_WATCH = {
    'OOO_U': ['Over 21', 'Over 21', 'Over 21', 'Under 21'],
    'UUU_O': ['Under 21', 'Under 21', 'Under 21', 'Over 21'],
    'OUOU': ['Over 21', 'Under 21', 'Over 21', 'Under 21'],
    'UOUO': ['Under 21', 'Over 21', 'Under 21', 'Over 21'],
    'OO': ['Over 21', 'Over 21'],
    'UU': ['Under 21', 'Under 21'],
    'O_U': ['Over 21', 'Under 21'],
    'U_O': ['Under 21', 'Over 21'],
    'O_O': ['Over 21', 'Over 21'],
    'U_U': ['Under 21', 'Under 21'],
    'OUU': ['Over 21', 'Under 21', 'Under 21'],
    'UOO': ['Under 21', 'Over 21', 'Over 21'],
    'OUO': ['Over 21', 'Under 21', 'Over 21'],
    'UOU': ['Under 21', 'Over 21', 'Under 21'],
    'OOOU': ['Over 21', 'Over 21', 'Over 21', 'Under 21'],
    'UUUO': ['Under 21', 'Under 21', 'Under 21', 'Over 21'],
    'OOUU': ['Over 21', 'Over 21', 'Under 21', 'Under 21'],
    'UUOO': ['Under 21', 'Under 21', 'Over 21', 'Over 21'],
}



# --- Google Sheets Authentication and Data Loading ---

@st.cache_resource(ttl="1h")
def get_gspread_client():
    try:
        # Load credentials from Streamlit secrets
        gcp_service_account_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            gcp_service_account_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Error authenticating with Google Sheets: {e}")
        st.stop() # Stop the app if authentication fails

def get_drive_service():
    try:
        gcp_service_account_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            gcp_service_account_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"Error authenticating with Google Drive: {e}")
        return None


def load_data(client):
    try:
        spreadsheet_name = st.secrets["private_gsheets_url"].split('/')[5]
        # Attempt to open by name first
        sh = client.open(spreadsheet_name)
    except SpreadsheetNotFound:
        # If not found by name, try opening by URL
        sh = client.open_by_url(st.secrets["private_gsheets_url"])
    
    worksheet = sh.worksheet("Sheet1") # Or your specific worksheet name
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    # Data Cleaning and Type Conversion
    if not df.empty:
        # Convert 'Date' to datetime objects
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Convert card columns to string to handle mixed types if any
        for col in ['Card 1', 'Card 2', 'Card 3']:
            df[col] = df[col].astype(str)
        # Convert numeric columns, coerce errors will turn non-numeric into NaN
        for col in ['Round', 'Player A Cards Sum', 'Player B Cards Sum', 'Player C Cards Sum', 'Player D Cards Sum']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where essential numerical conversions failed
        df.dropna(subset=['Round', 'Player A Cards Sum'], inplace=True)
        
        # Ensure 'Outcome' column exists
        if 'Outcome' not in df.columns:
            df['Outcome'] = '' # Add it if missing

        # Ensure 'Deck_ID' exists and is numeric (or handled as string if preferred)
        if 'Date' in df.columns and not df['Date'].isnull().all():
            if 'Deck_ID' not in df.columns:
                df['Deck_ID'] = df.groupby('Date').ngroup() + 1 # Simple sequential ID per date
            else:
                # Try to convert to numeric, if fails, keep as string or generate new
                df['Deck_ID'] = pd.to_numeric(df['Deck_ID'], errors='coerce').fillna(df.groupby('Date').ngroup() + 1)
        else:
            st.warning("Date column is missing or invalid. Generating simple Deck_ID based on row index.")
            df['Deck_ID'] = (df.index // PREDICTION_ROUNDS_CONSIDERED) + 1 # Fallback if Date is problematic
            
    return df

def update_data(client, new_data):
    try:
        spreadsheet_name = st.secrets["private_gsheets_url"].split('/')[5]
        sh = client.open(spreadsheet_name)
        worksheet = sh.worksheet("Sheet1")
        
        # Append new data
        worksheet.append_row(new_data)
        st.success("Data recorded successfully!")
        st.experimental_rerun() # Rerun to refresh the data display
    except Exception as e:
        st.error(f"Error updating Google Sheet: {e}")

# --- AI Model Training and Prediction ---

# Global variable to store the model and encoder
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None


def save_model_to_drive(drive_service, model, le, folder_id):
    try:
        # Save model and encoder to BytesIO objects
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)

        le_buffer = io.BytesIO()
        joblib.dump(le, le_buffer)
        le_buffer.seek(0)

        # Upload/Update model file
        model_file_metadata = {
            'name': 'logistic_regression_model.joblib',
            'parents': [folder_id],
            'mimeType': 'application/octet-stream'
        }
        
        # Check if file exists to update or create
        response = drive_service.files().list(
            q=f"name='logistic_regression_model.joblib' and '{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id)'
        ).execute()
        
        files = response.get('files', [])
        if files:
            file_id = files[0]['id']
            media_body = MediaFileUpload(
                filename='logistic_regression_model.joblib', # Dummy filename for MediaFileUpload
                mimetype='application/octet-stream',
                resumable=True,
                body=model_buffer
            )
            drive_service.files().update(fileId=file_id, media_body=media_body).execute()
            st.toast("AI model updated on Google Drive.")
        else:
            media_body = MediaFileUpload(
                filename='logistic_regression_model.joblib',
                mimetype='application/octet-stream',
                resumable=True,
                body=model_buffer
            )
            drive_service.files().create(body=model_file_metadata, media_body=media_body, fields='id').execute()
            st.toast("AI model saved to Google Drive.")

        # Upload/Update LabelEncoder file
        le_file_metadata = {
            'name': 'label_encoder.joblib',
            'parents': [folder_id],
            'mimeType': 'application/octet-stream'
        }

        response = drive_service.files().list(
            q=f"name='label_encoder.joblib' and '{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id)'
        ).execute()

        files = response.get('files', [])
        if files:
            file_id = files[0]['id']
            media_body = MediaFileUpload(
                filename='label_encoder.joblib',
                mimetype='application/octet-stream',
                resumable=True,
                body=le_buffer
            )
            drive_service.files().update(fileId=file_id, media_body=media_body).execute()
            st.toast("LabelEncoder updated on Google Drive.")
        else:
            media_body = MediaFileUpload(
                filename='label_encoder.joblib',
                mimetype='application/octet-stream',
                resumable=True,
                body=le_buffer
            )
            drive_service.files().create(body=le_file_metadata, media_body=media_body, fields='id').execute()
            st.toast("LabelEncoder saved to Google Drive.")

    except Exception as e:
        st.error(f"Error saving model/encoder to Google Drive: {e}")

def load_model_from_drive(drive_service, folder_id):
    model = None
    le = None
    try:
        # Load model file
        model_response = drive_service.files().list(
            q=f"name='logistic_regression_model.joblib' and '{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id)'
        ).execute()
        
        model_files = model_response.get('files', [])
        if model_files:
            file_id = model_files[0]['id']
            request = drive_service.files().get_media(fileId=file_id)
            model_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(model_buffer, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            model_buffer.seek(0)
            model = joblib.load(model_buffer)
            st.toast("AI model loaded from Google Drive.")
        else:
            st.warning("AI model file not found on Google Drive.")

        # Load LabelEncoder file
        le_response = drive_service.files().list(
            q=f"name='label_encoder.joblib' and '{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id)'
        ).execute()

        le_files = le_response.get('files', [])
        if le_files:
            file_id = le_files[0]['id']
            request = drive_service.files().get_media(fileId=file_id)
            le_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(le_buffer, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            le_buffer.seek(0)
            le = joblib.load(le_buffer)
            st.toast("LabelEncoder loaded from Google Drive.")
        else:
            st.warning("LabelEncoder file not found on Google Drive.")

    except Exception as e:
        st.error(f"Error loading model/encoder from Google Drive: {e}")
    return model, le

def train_ai_model(df):
    st.info("Training AI Model... This may take a moment.")
    print("Starting train_ai_model function (for sequence prediction)...")

    # NEW DEBUG PRINT: Check columns before sorting
    print(f"DEBUG (TRAINING): Columns in df received by train_ai_model: {df.columns.tolist()}")

    # Check if 'Deck_ID' actually exists before sorting
    if 'Deck_ID' not in df.columns:
        st.error("Error: 'Deck_ID' column is missing in the data. Cannot train AI model. Please ensure your data has a 'Date' column and is not empty.")
        return None, None # Exit early if critical column is missing

    df_sorted = df.sort_values(by=['Deck_ID', 'Round']).copy()

    # Ensure LabelEncoder is fitted on all unique outcomes that exist in the data
    # This prevents issues with unseen labels during transformation
    le = LabelEncoder()
    # Fit on all unique outcomes in the entire historical data
    le.fit(df_sorted['Outcome'].unique())
    print(f"DEBUG (TRAINING): LabelEncoder classes after fit: {le.classes_.tolist()}")

    df_sorted['Outcome_Encoded'] = le.transform(df_sorted['Outcome'])

    # Create lagged features
    lag_features = []
    for i in range(1, PREDICTION_ROUNDS_CONSIDERED + 1):
        col_name = f'Outcome_Lag{i}'
        df_sorted[col_name] = df_sorted.groupby('Deck_ID')['Outcome_Encoded'].shift(i)
        lag_features.append(col_name)

    df_train = df_sorted.dropna(subset=lag_features).copy()

    if df_train.empty:
        st.warning("Not enough data to train the AI model after creating lagged features. Please record more rounds.")
        return None, None

    X = df_train[lag_features]
    y_encoded = df_train['Outcome_Encoded']

    if y_encoded.nunique() < 2:
        st.warning("Not enough unique outcomes in the training data to train a classification model. Need at least 2 distinct outcomes.")
        return None, None

    print(f"DEBUG (TRAINING): Starting Logistic Regression training for sequence prediction with {len(X)} samples.")
    print(f"DEBUG (TRAINING): Shape of X (features): {X.shape}")
    print(f"DEBUG (TRAINING): Columns of X (features): {X.columns.tolist()}")
    print(f"DEBUG (TRAINING): Number of unique outcomes in y_encoded: {y_encoded.nunique()}")

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y_encoded)

    print(f"DEBUG: Model fitted successfully for sequence prediction.")
    if hasattr(model, 'feature_names_in_'):
        print(f"DEBUG (TRAINING): Model feature names learned: {model.feature_names_in_.tolist()}")
    else:
        print(f"DEBUG (TRAINING): Model has no feature_names_in_ attribute.")

    st.success("AI Model trained successfully!")
    return model, le

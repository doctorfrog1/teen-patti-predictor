import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os

# --- NEW IMPORTS FOR AI and Google Drive ---
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib  # For saving/loading the model

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.oauth2.service_account import Credentials
# import io # For in-memory file handling (if needed, but direct file is better here)

# ... (rest of your existing imports) ...

# --- Configuration ---
PLAYER_A_FIXED_CARDS_STR = {'J♣', '10♠', '9♠'}  # Player A's fixed cards (assuming they are always out of play)
PREDICTION_ROUNDS_CONSIDERED = 10  # Number of previous rounds to consider for simple prediction
STREAK_THRESHOLD = 3  # Minimum streak length to highlight
OVER_UNDER_BIAS_THRESHOLD = 0.6  # If Over/Under > 60% of rounds, show bias

# --- AI Configuration ---
SEQUENCE_LENGTH = 3  # You can adjust this based on how many past outcomes you think matter
MODEL_FILE = "prediction_model.joblib"
ENCODER_FILE = "label_encoder.joblib"
MODEL_DIR = ".streamlit/data"  # Define the directory for models and encoders

# Add this PATTERNS_TO_WATCH dictionary here
PATTERNS_TO_WATCH = {
    # Existing patterns (specific longer sequences)
    'OOO_U': ['Over 21', 'Over 21', 'Over 21', 'Under 21'],
    'UUU_O': ['Under 21', 'Under 21', 'Under 21', 'Over 21'],
    'OUOU': ['Over 21', 'Under 21', 'Over 21', 'Under 21'],
    'UOUO': ['Under 21', 'Over 21', 'Under 21', 'Over 21'],

    # Shorter, "binary-like" patterns and reversals (O=Over, U=Under)
    'OO': ['Over 21', 'Over 21'],
    'UU': ['Under 21', 'Under 21'],
    'O_U': ['Over 21', 'Under 21'],
    'U_O': ['Under 21', 'Over 21'],
    'OOU': ['Over 21', 'Over 21', 'Under 21'],
    'UUO': ['Under 21', 'Under 21', 'Over 21'],
    'OUU': ['Over 21', 'Under 21', 'Under 21'],
    'UOO': ['Under 21', 'Over 21', 'Over 21'],

    # Longer streaks
    'OOO': ['Over 21', 'Over 21', 'Over 21'],
    'UUU': ['Under 21', 'Under 21', 'Under 21'],
    'OOOO': ['Over 21', 'Over 21', 'Over 21', 'Over 21'],
    'UUUU': ['Under 21', 'Under 21', 'Under 21', 'Under 21'],

    # Alternating sequences (O=Over, U=Under)
    'Alt_O_U_O': ['Over 21', 'Under 21', 'Over 21'],
    'Alt_U_O_U': ['Under 21', 'Over 21', 'Under 21'],

    # Patterns involving "Exactly 21" (E=Exactly 21)
    'E': ['Exactly 21'],
    'EE': ['Exactly 21', 'Exactly 21'],
    'OE': ['Over 21', 'Exactly 21'],
    'UE': ['Under 21', 'Exactly 21'],
    'EO': ['Exactly 21', 'Over 21'],
    'EU': ['Exactly 21', 'Under 21'],
    'OEO': ['Over 21', 'Exactly 21', 'Over 21'],
    'UEU': ['Under 21', 'Exactly 21', 'Under 21'],
    'E_O_O': ['Exactly 21', 'Over 21', 'Over 21'],
    'E_U_U': ['Exactly 21', 'Under 21', 'Under 21'],
    'O_E_U': ['Over 21', 'Exactly 21', 'Under 21'],
    'U_E_O': ['Under 21', 'Exactly 21', 'Over 21'],
}

# Define card values (J, Q, K are 11, 12, 13 as per your request)
card_values = {
    'A♠': 1, '2♠': 2, '3♠': 3, '4♠': 4, '5♠': 5, '6♠': 6, '7♠': 7, '8♠': 8, '9♠': 9, '10♠': 10, 'J♠': 11, 'Q♠': 12, 'K♠': 13,
    'A♦': 1, '2♦': 2, '3♦': 3, '4♦': 4, '5♦': 5, '6♦': 6, '7♦': 7, '8♦': 8, '9♦': 9, '10♦': 10, 'J♦': 11, 'Q♦': 12, 'K♦': 13,
    'A♣': 1, '2♣': 2, '3♣': 3, '4♣': 4, '5♣': 5, '6♣': 6, '7♣': 7, '8♣': 8, '9♣': 9, '10♣': 10, 'J♣': 11, 'Q♣': 12, 'K♣': 13,
    'A♥': 1, '2♥': 2, '3♥': 3, '4♥': 4, '5♥': 5, '6♥': 6, '7♥': 7, '8♥': 8, '9♥': 9, '10♥': 10, 'J♥': 11, 'Q♥': 12, 'K♥': 13
}
ALL_CARDS = list(card_values.keys())

# --- HELPER FUNCTIONS ---

# Function to get gspread client from Streamlit secrets
@st.cache_resource
def get_gspread_and_drive_clients():
    """Authenticates with Google Sheets and Google Drive using service account credentials."""
    try:
        # Load credentials from st.secrets
        creds_info = st.secrets.gcp_service_account

        # Use credentials to authenticate gspread
        gc = gspread.service_account_from_dict(creds_info)

        # Authenticate PyDrive2 with the same service account
        # Define scopes for Drive access
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',  # For gspread
            'https://www.googleapis.com/auth/drive'          # For PyDrive2
        ]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)

        gauth = GoogleAuth()
        gauth.credentials = creds
        drive = GoogleDrive(gauth)

        return gc, drive
    except Exception as e:
        st.error(f"Error loading Google Cloud credentials for Sheets/Drive: {e}. Please ensure st.secrets are configured correctly with service account details.")
        st.stop()  # Stop the app if credentials are not loaded
        return None, None  # Return None for both clients on error

# --- NEW: AI Model Training Function (Modified for Daily Learning & Direct File I/O) ---
def train_and_save_prediction_model(all_rounds_df, sequence_length=SEQUENCE_LENGTH):
    # ... (Your existing code for preparing data and training model/encoder) ...

    st.info(f"Training AI model with {len(X)} samples from today's data...")
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Get Google Sheets and Drive clients
    gc, drive = get_gspread_and_drive_clients()
    if not drive:
        st.error("Google Drive client not available. Cannot save AI model.")
        return False

    # Define temporary local paths for saving before upload
    temp_model_path = MODEL_FILE  # You can keep these names or make them more temporary
    temp_encoder_path = ENCODER_FILE

    try:
        model.fit(X, y)

        # Save model and encoder to temporary local files first
        with open(temp_model_path, "wb") as f:
            joblib.dump(model, f)
        with open(temp_encoder_path, "wb") as f:
            joblib.dump(le, f)

        # Get the folder ID from secrets
        model_folder_id = st.secrets.google_drive.model_folder_id

        # Function to upload/update a file in Google Drive
        def upload_or_update_file(drive_client, local_file_path, drive_file_name, parent_folder_id):
            # Search for the file within the specific folder
            file_list = drive_client.ListFile({
                'q': f"'{parent_folder_id}' in parents and title='{drive_file_name}' and trashed=false"
            }).GetList()

            if file_list:
                # File exists, update it
                file = file_list[0]
                file.SetContentFile(local_file_path)
                file.Upload()
                st.info(f"Updated '{drive_file_name}' on Google Drive in folder ID {parent_folder_id}.")
            else:
                # File does not exist, create it
                file = drive_client.CreateFile({'title': drive_file_name, 'parents': [{'id': parent_folder_id}]})
                file.SetContentFile(local_file_path)
                file.Upload()
                st.info(f"Uploaded new '{drive_file_name}' to Google Drive in folder ID {parent_folder_id}.")

        # Upload the model and encoder files
        upload_or_update_file(drive, temp_model_path, MODEL_FILE, model_folder_id)
        upload_or_update_file(drive, temp_encoder_path, ENCODER_FILE, model_folder_id)

        st.success("AI prediction model trained and saved successfully to Google Drive!")
        return True
    except Exception as e:
        st.error(f"Error during AI model training or saving to Google Drive: {str(e)}")
        return False
    finally:
        # Clean up temporary local files
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        if os.path.exists(temp_encoder_path):
            os.remove(temp_encoder_path)

# --- NEW: AI Model Loading Function (Modified for Direct File I/O) ---
@st.cache_resource
def load_ai_model():
    model = None
    le = None

    # Get Google Sheets and Drive clients
    gc, drive = get_gspread_and_drive_clients()
    if not drive:
        st.sidebar.error("Google Drive client not available. Cannot load AI model.")
        return None, None

    # Define temporary local paths for downloaded files
    download_model_path = MODEL_FILE  # Will download to app's working directory
    download_encoder_path = ENCODER_FILE

    try:
        model_folder_id = st.secrets.google_drive.model_folder_id

        # Download model file
        file_list = drive.ListFile({
            'q': f"'{model_folder_id}' in parents and title='{MODEL_FILE}' and trashed=false"
        }).GetList()
        if not file_list:
            st.sidebar.warning(f"AI Prediction Model '{MODEL_FILE}' not found on Google Drive in folder ID {model_folder_id}.")
            return None, None
        file = file_list[0]
        file.GetContentFile(download_model_path)

        # Download encoder file
        file_list_encoder = drive.ListFile({
            'q': f"'{model_folder_id}' in parents and title='{ENCODER_FILE}' and trashed=false"
        }).GetList()
        if not file_list_encoder:
            st.sidebar.warning(f"Label Encoder '{ENCODER_FILE}' not found on Google Drive in folder ID {model_folder_id}.")
            # Clean up potentially downloaded model if encoder is missing
            if os.path.exists(download_model_path):
                os.remove(download_model_path)
            return None, None
        file_encoder = file_list_encoder[0]
        file_encoder.GetContentFile(download_encoder_path)

        # Load from local downloaded files
        with open(download_model_path, "rb") as f:
            model = joblib.load(f)
        with open(download_encoder_path, "rb") as f:
            le = joblib.load(f)

        st.sidebar.success("AI Prediction Model Loaded from Google Drive.")
        return model, le
    except Exception as e:
        st.sidebar.error(f"Error loading AI model from Google Drive: {str(e)}")
        return None, None
    finally:
        # Clean up temporary downloaded files
        if os.path.exists(download_model_path):
            os.remove(download_model_path)
        if os.path.exists(download_encoder_path):
            os.remove(download_encoder_path)

# --- NEW: Function to load all rounds from sheet for training ---
def load_all_historical_rounds_from_sheet():
    gc, _ = get_gspread_and_drive_clients()
    if not gc:
        return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])  # Return empty DF

    try:
        spreadsheet = gc.open("Casino Card Game Log")
        worksheet = spreadsheet.worksheet("Sheet1")
        data = worksheet.get_all_records()
        if data:
            df = pd.DataFrame(data)
            # Ensure Deck_ID is numeric
            if 'Deck_ID' in df.columns:
                df['Deck_ID'] = pd.to_numeric(df['Deck_ID'], errors='coerce').fillna(1).astype(int)  # Default to 1 if NaN after coerce
            else:
                df['Deck_ID'] = 1  # Add Deck_ID if missing
            return df
        else:
            return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
    except SpreadsheetNotFound:
        st.error("Google Sheet 'Casino Card Game Log' not found. Please ensure it exists and is shared with the service account.")
        return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
    except Exception as e:
        st.error(f"Error loading all historical rounds from Google Sheet: {e}")
        return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

def load_rounds():
    gc, _ = get_gspread_and_drive_clients()
    if not gc:
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
        st.session_state.played_cards = set(PLAYER_A_FIXED_CARDS_STR)  # Only Player A's cards initially if data load fails.
        return

    try:
        spreadsheet = gc.open("Casino Card Game Log")

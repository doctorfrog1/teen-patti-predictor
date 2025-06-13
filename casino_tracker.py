import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os

# --- NEW IMPORTS FOR AI and Google Drive ---
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib # For saving/loading the model

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.oauth2.service_account import Credentials
import io # For in-memory file handling (if needed, but direct file is better here)

# --- Constants ---
# For Google Sheets
SPREADSHEET_NAME = "CasinoCardTracker"
SHEET_NAME = "CardRounds"
DECK_SHEET_NAME = "DeckIDs"

# For AI Model
MODEL_FILE = "prediction_model.joblib"
ENCODER_FILE = "label_encoder.joblib"
SEQUENCE_LENGTH = 3 # Number of past rounds to consider for AI prediction

# --- Streamlit Session State Initialization ---
if "deck_id" not in st.session_state:
    st.session_state.deck_id = None
if "current_deck" not in st.session_state:
    st.session_state.current_deck = []
if "ai_model" not in st.session_state:
    st.session_state.ai_model = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None
if "total_hands_played" not in st.session_state:
    st.session_state.total_hands_played = 0

# --- Helper Functions ---

@st.cache_resource
def get_gspread_and_drive_clients():
    """Authenticates with Google Sheets and Google Drive using service account credentials."""
    try:
        creds_info = st.secrets.gcp_service_account
        
        # Authenticate gspread
        gc = gspread.service_account_from_dict(creds_info)

        # Authenticate PyDrive2 with the same service account
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets', # For gspread
            'https://www.googleapis.com/auth/drive'         # For PyDrive2
        ]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        
        gauth = GoogleAuth()
        gauth.credentials = creds
        drive = GoogleDrive(gauth)

        return gc, drive
    except Exception as e:
        st.error(f"Error loading Google Cloud credentials for Sheets/Drive: {e}. Please ensure st.secrets are configured correctly with service account details.")
        st.stop() # Stop the app if credentials are not loaded
        return None, None

def load_all_historical_rounds_from_sheet():
    """Loads all historical rounds from the Google Sheet."""
    try:
        gc, _ = get_gspread_and_drive_clients()
        if not gc: return pd.DataFrame() # Return empty DataFrame if client failed
        spreadsheet = gc.open(SPREADSHEET_NAME)
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        # Ensure Timestamp is datetime for filtering
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        return df
    except SpreadsheetNotFound:
        st.error(f"Spreadsheet '{SPREADSHEET_NAME}' not found. Please ensure it exists and the service account has access.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading historical rounds: {e}")
        return pd.DataFrame()

def generate_features_for_ai(rounds_df, sequence_length):
    features = []
    outcomes = []

    # Sort data by Timestamp to ensure correct sequence
    rounds_df = rounds_df.sort_values(by='Timestamp').reset_index(drop=True)

    for i in range(len(rounds_df) - sequence_length):
        sequence_df = rounds_df.iloc[i : i + sequence_length]
        
        # Ensure all required columns are present and not empty
        required_cols = ['Card1', 'Card2', 'Card3', 'Sum', 'Outcome']
        if not all(col in sequence_df.columns and not sequence_df[col].isnull().any() for col in required_cols):
            continue # Skip incomplete sequences

        # Extract features (sum of cards from the sequence)
        # Flatten the list of sums from the sequence
        seq_sums = sequence_df['Sum'].tolist()
        
        # Extract the outcome of the round immediately following the sequence
        target_outcome = rounds_df.iloc[i + sequence_length]['Outcome']

        # Only add if the sequence and target outcome are valid
        if all(isinstance(s, (int, float)) for s in seq_sums) and pd.notna(target_outcome):
            features.append(seq_sums) # Features are the sums of the sequence
            outcomes.append(target_outcome) # Outcome is the actual outcome of the next round

    return pd.DataFrame(features, columns=[f'Sum_R{j+1}' for j in range(sequence_length)]), pd.Series(outcomes)

def get_latest_rounds_for_prediction(rounds_df, sequence_length):
    # Sort by timestamp to ensure the latest rounds are correctly identified
    rounds_df_sorted = rounds_df.sort_values(by='Timestamp', ascending=False)
    
    # Get the last 'sequence_length' rounds for prediction
    latest_rounds = rounds_df_sorted.head(sequence_length)
    
    # Ensure we have enough rounds
    if len(latest_rounds) < sequence_length:
        return None
    
    # The order needs to be chronological for feature generation
    latest_rounds = latest_rounds.sort_values(by='Timestamp', ascending=True)
    
    # Extract sums for the sequence
    current_features = latest_rounds['Sum'].tolist()
    
    # Return as a DataFrame for prediction
    return pd.DataFrame([current_features], columns=[f'Sum_R{j+1}' for j in range(sequence_length)])

def update_deck_ids_sheet(new_deck_id):
    try:
        gc, _ = get_gspread_and_drive_clients()
        if not gc: return
        spreadsheet = gc.open(SPREADSHEET_NAME)
        try:
            worksheet = spreadsheet.worksheet(DECK_SHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=DECK_SHEET_NAME, rows=1, cols=1)
            worksheet.update_cell(1, 1, "Last_Deck_ID") # Add header
        
        # Update the last_deck_id
        worksheet.update_cell(2, 1, new_deck_id)
        st.success(f"Deck ID {new_deck_id} updated in Google Sheet.")
    except Exception as e:
        st.error(f"Error updating Deck ID in Google Sheet: {e}")

def create_new_deck():
    try:
        gc, _ = get_gspread_and_drive_clients()
        if not gc: return
        spreadsheet = gc.open(SPREADSHEET_NAME)
        try:
            worksheet = spreadsheet.worksheet(DECK_SHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=DECK_SHEET_NAME, rows=1, cols=1)
            worksheet.update_cell(1, 1, "Last_Deck_ID") # Add header
        
        # Get the current max ID
        try:
            existing_id = int(worksheet.cell(2, 1).value)
        except (ValueError, TypeError, IndexError):
            existing_id = 0
            
        new_id = existing_id + 1
        worksheet.update_cell(2, 1, new_id)
        return new_id
    except Exception as e:
        st.error(f"Error creating new deck ID: {e}")
        return None

# --- Main Training Function ---
def train_and_save_prediction_model(all_rounds_df, sequence_length=SEQUENCE_LENGTH):
    st.info(f"Preparing data for AI model training from {len(all_rounds_df)} historical rounds.")

    # Filter for today's data for retraining logic
    today = datetime.now().date()
    daily_rounds_df = all_rounds_df[
        pd.to_datetime(all_rounds_df['Timestamp']).dt.date == today
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    st.info(f"Training AI model with {len(daily_rounds_df)} samples from today's data ({today})...")

    if len(daily_rounds_df) < sequence_length + 1:
        st.warning(f"Not enough recent rounds data to train the AI model. Need at least {sequence_length + 1} rounds from today.")
        st.info("Cleared old AI model files due to insufficient data.")
        # We need to clear the cache and potentially unset the model if it was loaded from a prior valid training
        st.cache_resource.clear()
        st.session_state.ai_model = None
        st.session_state.label_encoder = None
        return False

    # Generate features and outcomes
    X, y_outcomes = generate_features_for_ai(daily_rounds_df, sequence_length)

    if X.empty or y_outcomes.empty:
        st.warning("No valid feature-outcome pairs could be generated from today's data.")
        st.info("Cleared old AI model files due to insufficient data.")
        st.cache_resource.clear()
        st.session_state.ai_model = None
        st.session_state.label_encoder = None
        return False
    
    # --- Debugging LabelEncoder and classes before fitting ---
    st.write("--- Debugging AI Model Training ---")
    st.write(f"Unique outcomes in training data (before encoding): {y_outcomes.unique()}")

    le = LabelEncoder()
    try:
        y = le.fit_transform(y_outcomes)
        st.write(f"LabelEncoder classes after fit_transform: {le.classes_}")
        st.write(f"Encoded outcomes (y): {y}")
        st.write(f"Number of unique encoded outcomes: {len(le.classes_)}")

        if len(le.classes_) < 2:
            st.error(f"Error during AI model training: This solver needs samples of at least 2 classes in the data, but the data contains only one class: {le.classes_[0]}. Please play more rounds with varied outcomes.")
            st.cache_resource.clear()
            st.session_state.ai_model = None
            st.session_state.label_encoder = None
            return False

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        st.info("AI model training complete.")

        # Get Google Sheets and Drive clients
        gc, drive = get_gspread_and_drive_clients()
        if not drive:
            st.error("Google Drive client not available. Cannot save AI model.")
            return False

        # Define temporary local paths for saving before upload
        temp_model_path = MODEL_FILE
        temp_encoder_path = ENCODER_FILE

        # Save model and encoder to temporary local files first
        with open(temp_model_path, "wb") as f:
            joblib.dump(model, f)
        with open(temp_encoder_path, "wb") as f:
            joblib.dump(le, f)

        # Get the folder ID from secrets
        try:
            model_folder_id = st.secrets.google_drive.model_folder_id
        except AttributeError:
            st.error("Google Drive model_folder_id not found in st.secrets. Please configure it.")
            return False

        # Function to upload/update a file in Google Drive
        def upload_or_update_file(drive_client, local_file_path, drive_file_name, parent_folder_id):
            file_list = drive_client.ListFile({
                'q': f"'{parent_folder_id}' in parents and title='{drive_file_name}' and trashed=false"
            }).GetList()

            if file_list:
                file = file_list[0]
                file.SetContentFile(local_file_path)
                file.Upload()
                st.info(f"Updated '{drive_file_name}' on Google Drive in folder ID {parent_folder_id}.")
            else:
                file = drive_client.CreateFile({'title': drive_file_name, 'parents': [{'id': parent_folder_id}]})
                file.SetContentFile(local_file_path)
                file.Upload()
                st.info(f"Uploaded new '{drive_file_name}' to Google Drive in folder ID {parent_folder_id}.")

        # Upload the model and encoder files
        upload_or_update_file(drive, temp_model_path, MODEL_FILE, model_folder_id)
        upload_or_update_file(drive, temp_encoder_path, ENCODER_FILE, model_folder_id)

        st.success("AI prediction model trained and saved successfully to Google Drive!")
        st.write("--- End Debugging AI Model Training ---")
        return True
    except Exception as e:
        st.error(f"Error during AI model training or saving to Google Drive: {str(e)}")
        st.write("--- End Debugging AI Model Training ---")
        st.cache_resource.clear()
        st.session_state.ai_model = None
        st.session_state.label_encoder = None
        return False
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        if os.path.exists(temp_encoder_path):
            os.remove(temp_encoder_path)

# --- Update load_ai_model to download

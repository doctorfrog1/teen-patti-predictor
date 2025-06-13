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
# import io # Not strictly needed if using direct file I/O

# --- Configuration ---
PLAYER_A_FIXED_CARDS_STR = {'J♣', '10♠', '9♠'} # Player A's fixed cards (assuming they are always out of play)
PREDICTION_ROUNDS_CONSIDERED = 10 # Number of previous rounds to consider for simple prediction
STREAK_THRESHOLD = 3 # Minimum streak length to highlight
OVER_UNDER_BIAS_THRESHOLD = 0.6 # If Over/Under > 60% of rounds, show bias

# --- AI Configuration ---
SEQUENCE_LENGTH = 3 # You can adjust this based on how many past outcomes you think matter
MODEL_FILE = "prediction_model.joblib"
ENCODER_FILE = "label_encoder.joblib"
# MODEL_DIR = ".streamlit/data" # Not directly used with PyDrive2 and temp files

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

# --- NEW: AI Model Training Function (Modified for Daily Learning & Direct File I/O) ---
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
        st.cache_resource.clear() # Clear entire cache
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

    gc, drive = get_gspread_and_drive_clients()
    if not drive:
        st.sidebar.error("Google Drive client not available. Cannot load AI model.")
        return None, None

    # Define temporary local paths for downloaded files
    download_model_path = MODEL_FILE # Will download to app's working directory
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
            if os.path.exists(download_model_path): os.remove(download_model_path)
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

# --- REVISED: load_rounds function to use get_gspread_and_drive_clients() ---
def load_rounds():
    gc, _ = get_gspread_and_drive_clients() # Use the combined client getter
    if not gc:
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
        st.session_state.played_cards = set(PLAYER_A_FIXED_CARDS_STR) # Only Player A's cards initially if data load fails.
        return

    try:
        spreadsheet = gc.open(SPREADSHEET_NAME) # Use SPREADSHEET_NAME constant
        worksheet = spreadsheet.worksheet(SHEET_NAME) # Use SHEET_NAME constant

        data = worksheet.get_all_records()
        if data:
            df = pd.DataFrame(data)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            if 'Deck_ID' not in df.columns:
                df['Deck_ID'] = 1 # Default to 1 if missing
            df['Deck_ID'] = pd.to_numeric(df['Deck_ID'], errors='coerce').fillna(1).astype(int)
            st.session_state.rounds = df
        else:
            st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

    except SpreadsheetNotFound:
        st.error(f"Google Sheet '{SPREADSHEET_NAME}' not found. Please ensure the name is correct and it's shared with the service account.")
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
    except Exception as e:
        st.error(f"Error loading rounds from Google Sheet: {e}. Starting with empty history.")
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

    # --- Crucial REVISED played_cards Initialization Logic ---
    st.session_state.played_cards = set() # Always start fresh for the current deck's played cards here

    # Add cards played in the current deck from the history
    if 'current_deck_id' in st.session_state and not st.session_state.rounds.empty:
        current_deck_rounds = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]
        for _, row in current_deck_rounds.iterrows():
            st.session_state.played_cards.add(row['Card1'])
            st.session_state.played_cards.add(row['Card2'])
            st.session_state.played_cards.add(row['Card3'])

    # Always add Player A's fixed cards (they are never available in any deck)
    for card in PLAYER_A_FIXED_CARDS_STR:
        st.session_state.played_cards.add(card)


def save_rounds():
    gc, _ = get_gspread_and_drive_clients() # Use the combined client getter
    if not gc:
        st.warning("Cannot save rounds: Google Sheets client not available.")
        return

    try:
        spreadsheet = gc.open(SPREADSHEET_NAME)
        worksheet = spreadsheet.worksheet(SHEET_NAME)

        data_to_write = [st.session_state.rounds.columns.tolist()] + st.session_state.rounds.astype(str).values.tolist()

        worksheet.clear()
        worksheet.update('A1', data_to_write)

    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Cannot save: Google Sheet '{SPREADSHEET_NAME}' not found. Please create the sheet and share it correctly.")
    except Exception as e:
        st.error(f"Error saving rounds to Google Sheet: {e}")

def get_current_streak(df):
    """Calculates the current streak of 'Over' or 'Under'."""
    if df.empty:
        return None, 0

    current_outcome = df.iloc[-1]['Outcome']
    streak_count = 0
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]['Outcome'] == current_outcome:
            streak

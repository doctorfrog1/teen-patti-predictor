import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os # Keep this for local file operations like model saving
import joblib
import traceback # Import traceback for detailed error logging
import numpy as np # NEW: Added for robust NaN handling and Deck_ID generation

# Machine Learning imports
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Correct Google Authentication import for service accounts
from google.oauth2.service_account import Credentials # For gspread

# NEW: Imports for Google API Client Library for Drive operations
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io # Needed for downloading files


# --- Configuration ---
MODEL_FOLDER_ID = "1CZepfjRZxWV_wfmEQuZLnbj9H2yAS9Ac"
PLAYER_A_FIXED_CARDS_STR = {'J♣', '10♠', '9♠'}
PREDICTION_ROUNDS_CONSIDERED = 10 # Number of previous rounds to consider for AI sequence prediction
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
    'OOU': ['Over 21', 'Over 21', 'Under 21'],
    'UUO': ['Under 21', 'Under 21', 'Over 21'],
    'OUU': ['Over 21', 'Under 21', 'Under 21'],
    'UOO': ['Under 21', 'Over 21', 'Over 21'],
    'OOO': ['Over 21', 'Over 21', 'Over 21'],
    'UUU': ['Under 21', 'Under 21', 'Under 21'],
    'OOOO': ['Over 21', 'Over 21', 'Over 21', 'Over 21'],
    'UUUU': ['Under 21', 'Under 21', 'Under 21', 'Under 21'],
    'Alt_O_U_O': ['Over 21', 'Under 21', 'Over 21'],
    'Alt_U_O_U': ['Under 21', 'Over 21', 'Under 21'],
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

card_values = {
    'A♠': 1, '2♠': 2, '3♠': 3, '4♠': 4, '5♠': 5, '6♠': 6, '7♠': 7, '8♠': 8, '9♠': 9, '10♠': 10, 'J♠': 11, 'Q♠': 12, 'K♠': 13,
    'A♦': 1, '2♦': 2, '3♦': 3, '4♦': 4, '5♦': 5, '6♦': 6, '7♦': 7, '8♦': 8, '9♦': 9, '10♦': 10, 'J♦': 11, 'Q♦': 12, 'K♦': 13,
    'A♣': 1, '2♣': 2, '3♣': 3, '4♣': 4, '5♣': 5, '6♣': 6, '7♣': 7, '8♣': 8, '9♣': 9, '10♣': 10, 'J♣': 11, 'Q♣': 12, 'K♣': 13,
    'A♥': 1, '2♥': 2, '3♥': 3, '4♥': 4, '5♥': 5, '6♥': 6, '7♥': 7, '8♥': 8, '9♥': 9, '10♥': 10, 'J♥': 11, 'Q♥': 12, 'K♥': 13
}
ALL_CARDS = list(card_values.keys())

# --- HELPER FUNCTIONS ---

@st.cache_resource
def get_gspread_and_drive_clients():
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Google Cloud service account credentials not found in `st.secrets`. Please configure `secrets.toml`.")
            return None, None
        creds_dict = st.secrets["gcp_service_account"]
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        gspread_credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)

        # Authorize gspread client
        gc = gspread.authorize(gspread_credentials)

        # Build Drive service client
        drive_service = build('drive', 'v3', credentials=gspread_credentials)

        return gc, drive_service
    except Exception as e:
        st.error(f"Error loading Google Cloud credentials for Sheets/Drive: {e}. Please ensure st.secrets are configured correctly with service account details.")
        st.caption("For more info on Streamlit secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management")
        return None, None

def delete_model_files_from_drive(drive_service, folder_id):
    """Deletes old model and label encoder files from Google Drive."""
    try:
        query = f"'{folder_id}' in parents and trashed=false and (name='prediction_model.joblib' or name='label_encoder.joblib')"
        results = drive_service.files().list(q=query, fields="files(id)").execute()
        items = results.get('files', [])
        for item in items:
            drive_service.files().delete(fileId=item['id']).execute()
            print(f"Deleted old file: {item['id']}")
        return True
    except Exception as e:
        st.error(f"Error deleting old model files from Google Drive: {e}")
        return False

def save_ai_model(model, label_encoder):
    """Saves the trained AI model and label encoder locally."""
    try:
        joblib.dump(model, 'prediction_model.joblib')
        joblib.dump(label_encoder, 'label_encoder.joblib')
        st.success("AI model saved locally.")
        return True
    except Exception as e:
        st.error(f"Error saving AI model locally: {e}")
        return False

def save_ai_model_to_drive():
    """Uploads the locally saved AI model and label encoder to Google Drive."""
    gc, drive_service = get_gspread_and_drive_clients()
    if drive_service is None:
        return False

    # Delete existing files first
    if not delete_model_files_from_drive(drive_service, MODEL_FOLDER_ID):
        st.warning("Could not clear old model files from Drive. Attempting to upload anyway, which might create duplicates.")

    try:
        file_metadata_model = {
            'name': 'prediction_model.joblib',
            'parents': [MODEL_FOLDER_ID]
        }
        media_model = MediaFileUpload('prediction_model.joblib', mimetype='application/octet-stream')
        drive_service.files().create(body=file_metadata_model, media_body=media_model, fields='id').execute()

        file_metadata_encoder = {
            'name': 'label_encoder.joblib',
            'parents': [MODEL_FOLDER_ID]
        }
        media_encoder = MediaFileUpload('label_encoder.joblib', mimetype='application/octet-stream')
        drive_service.files().create(body=file_metadata_encoder, media_body=media_encoder, fields='id').execute()

        st.success("AI model and label encoder uploaded to Google Drive.")
        return True
    except Exception as e:
        st.error(f"Error uploading AI model to Google Drive: {e}")
        return False
    finally:
        # Clean up local files after upload attempt
        if os.path.exists('prediction_model.joblib'):
            os.remove('prediction_model.joblib')
        if os.path.exists('label_encoder.joblib'):
            os.remove('label_encoder.joblib')

@st.cache_data
def load_all_historical_rounds_from_sheet():
    gc, _ = get_gspread_and_drive_clients()
    if gc is None:
        return pd.DataFrame(columns=['Timestamp', 'Round', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID']) # Changed Round_ID to Round

    try:
        spreadsheet = gc.open("Casino Card Game Log")
        worksheet = spreadsheet.worksheet("Sheet1")
        data = worksheet.get_all_records()
        if not data: # Handle empty sheet immediately
            st.warning("No data found in Google Sheet. Returning empty DataFrame.")
            return pd.DataFrame(columns=['Timestamp', 'Round', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID']) # Changed Round_ID to Round

        df = pd.DataFrame(data)
        df.columns = df.columns.str.replace(' ', '_') # Normalize column names (e.g., 'Round ID' to 'Round_ID')

        # Convert Timestamp and handle NaT
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        else:
            df['Timestamp'] = pd.NaT # Assign NaT if column is missing

        # Generate Deck_ID: Prioritize existing Deck_ID, then Timestamp, then sequential fallback
        if 'Deck_ID' in df.columns:
            df['Deck_ID'] = pd.to_numeric(df['Deck_ID'], errors='coerce')
        else:
            df['Deck_ID'] = np.nan # Initialize as NaN if not present

        # If Timestamp is valid for grouping, use it to generate Deck_ID
        if not df['Timestamp'].isnull().all():
            # Use a temporary column to assign new Deck_ID based on Timestamp groups for new/missing ones
            df['temp_Deck_ID'] = df.groupby(df['Timestamp'].dt.date).ngroup() + 1
            # Fill existing NaN Deck_ID values with the newly generated ones
            df['Deck_ID'] = df['Deck_ID'].fillna(df['temp_Deck_ID'])
            df.drop(columns=['temp_Deck_ID'], errors='ignore', inplace=True)
        else:
            st.warning("Timestamp column is missing or entirely invalid. Generating simple sequential Deck_ID.")
            # Fallback to sequential Deck_ID based on blocks of rounds if Timestamp is unusable
            df['Deck_ID'] = (df.index // PREDICTION_ROUNDS_CONSIDERED) + 1

        # Ensure Deck_ID is integer
        df['Deck_ID'] = df['Deck_ID'].fillna(1).astype(int)

        # Standardize Outcome column
        df['Outcome'] = df['Outcome'].astype(str).str.strip()
        df['Standard_Outcome_Char'] = df['Outcome'].apply(lambda x: {
            "Over 21": "O", "Under 21": "U", "Exactly 21": "E"
        }.get(x, x[0] if isinstance(x, str) and x and x[0] in ['O', 'U', 'E'] else None))
        df['Outcome'] = df['Standard_Outcome_Char'].map({
            "O": "Over 21", "U": "Under 21", "E": "Exactly 21"
        })
        df = df.drop(columns=['Standard_Outcome_Char'])
        valid_outcomes = ['Over 21', 'Under 21', 'Exactly 21']
        df = df[df['Outcome'].isin(valid_outcomes)]

        # Ensure 'Round_ID' and 'Sum' are numeric and handle NaNs
        # It's 'Round_ID' in the sheet, but train_ai_model expects 'Round'
        if 'Round_ID' in df.columns:
            df['Round'] = pd.to_numeric(df['Round_ID'], errors='coerce') # Create 'Round' column from 'Round_ID'
            df.drop(columns=['Round_ID'], inplace=True) # Drop original 'Round_ID'
        else:
            df['Round'] = np.nan # If 'Round_ID' is missing, set 'Round' to NaN

        if 'Sum' in df.columns: # Corresponds to 'Player A Cards Sum' in the previous version
            df['Sum'] = pd.to_numeric(df['Sum'], errors='coerce')
        else:
            df['Sum'] = np.nan # If 'Sum' is missing, set to NaN


        # Drop rows where critical columns for training are NaN
        df.dropna(subset=['Round', 'Sum', 'Outcome', 'Deck_ID'], inplace=True)

        if df.empty: # Check if it became empty after dropping NaNs
            st.warning("All data rows were dropped after cleaning. Cannot train AI model with empty data.")
            return pd.DataFrame(columns=['Timestamp', 'Round', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID']) # Changed Round_ID to Round

        return df

    except SpreadsheetNotFound:
        st.error(f"Google Sheet 'Casino Card Game Log' not found. Please ensure the sheet exists and the service account has access.")
        return pd.DataFrame(columns=['Timestamp', 'Round', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID']) # Changed Round_ID to Round
    except Exception as e:
        st.error(f"Error loading historical rounds from Google Sheet: {e}. Starting with empty history. Full error: {traceback.format_exc()}")
        return pd.DataFrame(columns=['Timestamp', 'Round', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID']) # Changed Round_ID to Round


def train_ai_model(df):
    st.info("Training AI Model... This may take a moment.")
    print("Starting train_ai_model function (for sequence prediction)...")

    if df.empty:
        st.error("No data available to train the AI model. Please ensure rounds are loaded from Google Sheets.")
        print("DEBUG (TRAINING): DataFrame is empty. Exiting train_ai_model.")
        return None, None

    # NEW DEBUG PRINT: Check columns before sorting
    print(f"DEBUG (TRAINING): Columns in df received by train_ai_model: {df.columns.tolist()}")

    # Check if 'Deck_ID' actually exists before sorting
    if 'Deck_ID' not in df.columns:
        st.error("Error: 'Deck_ID' column is missing in the data. Cannot train AI model. Please ensure your data has a 'Deck_ID' column after loading.")
        return None, None # Exit early if critical column is missing

    # Check for 'Round' column explicitly
    if 'Round' not in df.columns:
        st.error("Error: 'Round' column is missing in the data. Cannot train AI model. Ensure 'Round_ID' from Google Sheet is correctly renamed to 'Round'.")
        return None, None

    df_sorted = df.sort_values(by=['Deck_ID', 'Round']).copy()
    print(f"DEBUG (TRAINING): DataFrame sorted by Deck_ID and Round. Shape: {df_sorted.shape}")

    # Create lagged features for sequence prediction
    features = []
    labels = []

    # Iterate through each deck separately
    for deck_id, deck_df in df_sorted.groupby('Deck_ID'):
        outcomes_in_deck = deck_df['Outcome'].tolist()
        print(f"DEBUG (TRAINING): Deck {deck_id} has {len(outcomes_in_deck)} outcomes.")

        if len(outcomes_in_deck) > PREDICTION_ROUNDS_CONSIDERED:
            for i in range(len(outcomes_in_deck) - PREDICTION_ROUNDS_CONSIDERED):
                # Features are the last PREDICTION_ROUNDS_CONSIDERED outcomes
                lagged_outcomes = outcomes_in_deck[i : i + PREDICTION_ROUNDS_CONSIDERED]
                # Label is the outcome immediately following the lagged sequence
                next_outcome = outcomes_in_deck[i + PREDICTION_ROUNDS_CONSIDERED]

                feature_dict = {f'Outcome_Lag{j+1}': outcome for j, outcome in enumerate(reversed(lagged_outcomes))} # Reverse to match conventional lag order
                features.append(feature_dict)
                labels.append(next_outcome)

    if not features:
        st.warning(f"Not enough historical data to train the AI model. Need more than {PREDICTION_ROUNDS_CONSIDERED} rounds per deck.")
        print("DEBUG (TRAINING): No features generated. Exiting train_ai_model.")
        return None, None

    X = pd.DataFrame(features)
    y = pd.Series(labels)
    print(f"DEBUG (TRAINING): Value counts of y (labels) before encoding: \n{y.value_counts()}")

    # Encode categorical outcomes
    label_encoder = LabelEncoder()
    try:
        # **FIX 1: Explicitly fit LabelEncoder on all possible outcomes**
        all_possible_outcomes = ['Over 21', 'Under 21', 'Exactly 21']
        label_encoder.fit(all_possible_outcomes) # Ensure encoder knows all categories

        y_encoded = label_encoder.transform(y)
        X_encoded = X.apply(lambda col: label_encoder.transform(col) if col.name.startswith('Outcome_Lag') else col)
        print(f"DEBUG (TRAINING): LabelEncoder classes after fit_transform: {label_encoder.classes_.tolist()}")
    except ValueError as e:
        st.error(f"Error encoding outcomes during training: {e}. Ensure your 'Outcome' column only contains expected values ('Over 21', 'Under 21', 'Exactly 21').")
        print(f"DEBUG (TRAINING): Encoding error: {e}. Labels: {y.unique().tolist()}")
        return None, None

    print(f"DEBUG (TRAINING): Shape of X_encoded: {X_encoded.shape}")
    print(f"DEBUG (TRAINING): Columns of X_encoded: {X_encoded.columns.tolist()}") # Debug print for columns
    print(f"DEBUG (TRAINING): Shape of y_encoded: {y_encoded.shape}")

    # Train a Logistic Regression model
    try:
        model = LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter
        model.fit(X_encoded, y_encoded)
        # **FIX 2: Store feature names on the model after training**
        model.feature_columns_ = X_encoded.columns.tolist()
        st.success("AI Model trained successfully.")
        print("DEBUG (TRAINING): Model training successful.")
        return model, label_encoder
    except Exception as e:
        st.error(f"Error training the Logistic Regression model: {e}. Check data integrity and feature scaling if issues persist.")
        print(f"DEBUG (TRAINING): Model training failed: {e}")
        return None, None

def train_and_save_prediction_model():
    all_rounds_df = load_all_historical_rounds_from_sheet()

    if all_rounds_df.empty:
        st.warning("Cannot train model: No valid historical data found or all data was filtered out.")
        return False

    # Call the core training function
    st.info(f"Training AI model with {len(all_rounds_df)} samples from history.")
    model, label_encoder = train_ai_model(all_rounds_df.copy()) # Pass a copy

    if model is None or label_encoder is None:
        st.error("AI model training failed. Please check logs for details.")
        return False

    # Save the trained model locally
    local_save_success = save_ai_model(model, label_encoder)

    if local_save_success:
        st.info("Attempting to upload AI model to Google Drive for persistent storage...")
        drive_upload_success = save_ai_model_to_drive() # This function now handles deletion itself

        if drive_upload_success:
            st.success("AI prediction model trained and loaded into session state!")
            return True
        else:
            st.error("Failed to save AI model to Google Drive. Training complete but model not persistently stored.")
            return False
    else:
        st.error("Failed to save AI model locally. Training complete but model not persistently stored.")
        return False

@st.cache_data
def load_ai_model_from_drive():
    st.info("Attempting to load AI model from Google Drive...")
    gc, drive_service = get_gspread_and_drive_clients()
    if gc is None or drive_service is None:
        return None, None

    temp_model_path = 'prediction_model_downloaded.joblib'
    temp_encoder_path = 'label_encoder_downloaded.joblib'

    try:
        query = f"'{MODEL_FOLDER_ID}' in parents and trashed=false and (name='prediction_model.joblib' or name='label_encoder.joblib')"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        model_file_id = None
        encoder_file_id = None
        for item in items:
            if item['name'] == "prediction_model.joblib":
                model_file_id = item['id']
            elif item['name'] == "label_encoder.joblib":
                encoder_file_id = item['id']

        if not model_file_id or not encoder_file_id:
            st.warning("AI Model or Label Encoder files not found on Google Drive. Please train and upload the model.")
            return None, None

        # Download model file
        request_model = drive_service.files().get_media(fileId=model_file_id)
        fh_model = io.BytesIO()
        downloader_model = MediaIoBaseDownload(fh_model, request_model)
        done = False
        while done is False:
            status, done = downloader_model.next_chunk()
        with open(temp_model_path, 'wb') as f:
            f.write(fh_model.getvalue())

        # Download encoder file
        request_encoder = drive_service.files().get_media(fileId=encoder_file_id)
        fh_encoder = io.BytesIO()
        downloader_encoder = MediaIoBaseDownload(fh_encoder, request_encoder)
        done = False
        while done is False:
            status, done = downloader_encoder.next_chunk()
        with open(temp_encoder_path, 'wb') as f:
            f.write(fh_encoder.getvalue())

        # Load models
        model = joblib.load(temp_model_path)
        encoder = joblib.load(temp_encoder_path)

        st.session_state.ai_model = model
        st.session_state.label_encoder = encoder
        st.sidebar.success("AI Prediction Model Loaded from Google Drive.")
        return model, encoder

    except Exception as e:
        st.error(f"Error loading AI model from Google Drive: {e}")
        return None, None
    finally:
        # Clean up temporary downloaded files
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        if os.path.exists(temp_encoder_path):
            os.remove(temp_encoder_path)

def load_rounds():
    gc, _ = get_gspread_and_drive_clients()
    if not gc:
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
        st.session_state.played_cards = set(PLAYER_A_FIXED_CARDS_STR)
        return

    try:
        spreadsheet = gc.open("Casino Card Game Log")
        worksheet = spreadsheet.worksheet("Sheet1")

        data = worksheet.get_all_records()
        if data:
            df = pd.DataFrame(data)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            if 'Deck_ID' not in df.columns:
                df['Deck_ID'] = 1
            df['Deck_ID'] = df['Deck_ID'].astype(int)
            st.session_state.rounds = df
        else:
            st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

    except SpreadsheetNotFound:
        st.error("Google Sheet 'Casino Card Game Log' not found. Please ensure the name is correct and it's shared with the service account.")
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
    except Exception as e:
        st.error(f"Error loading rounds from Google Sheet: {e}. Starting with empty history.")
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

    st.session_state.played_cards = set()

    if not st.session_state.rounds.empty:
        current_deck_rounds = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]
        for _, row in current_deck_rounds.iterrows():
            st.session_state.played_cards.add(row['Card1'])
            st.session_state.played_cards.add(row['Card2'])
            st.session_state.played_cards.add(row['Card3'])

    for card in PLAYER_A_FIXED_CARDS_STR:
        st.session_state.played_cards.add(card)

def save_rounds():
    gc, _ = get_gspread_and_drive_clients()
    if not gc:
        st.warning("Cannot save rounds: Google Sheets client not available.")
        return

    try:
        spreadsheet = gc.open("Casino Card Game Log")
        worksheet = spreadsheet.worksheet("Sheet1")

        data_to_write = [st.session_state.rounds.columns.tolist()] + st.session_state.rounds.astype(str).values.tolist()

        worksheet.clear()
        worksheet.update(range_name='A1', values=data_to_write)


    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Cannot save: Google Sheet 'Casino Card Game Log' not found. Please create the sheet and share it correctly.")
    except Exception as e:
        st.error(f"Error saving rounds to Google Sheet: {e}")

def get_current_streak(df):
    if df.empty:
        return None, 0

    current_outcome = df.iloc[-1]['Outcome']
    streak_count = 0
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]['Outcome'] == current_outcome:
            streak_count += 1
        else:
            break
    return current_outcome, streak_count

def predict_next_outcome_from_pattern(df_all_rounds, pattern_sequence):
    if df_all_rounds.empty or not pattern_sequence:
        return None, 0

    next_outcomes = []
    pattern_len = len(pattern_sequence)

    for deck_id, deck_df in df_all_rounds.groupby('Deck_ID'):
        outcomes_in_deck = deck_df['Outcome'].tolist()

        for i in range(len(outcomes_in_deck) - pattern_len):
            if outcomes_in_deck[i : i + pattern_len] == pattern_sequence:
                if (i + pattern_len) < len(outcomes_in_deck):
                    next_outcomes.append(outcomes_in_deck[i + pattern_len])

    if not next_outcomes:
        return None, 0

    outcome_counts = pd.Series(next_outcomes).value_counts()
    most_likely_outcome = outcome_counts.index[0]
    confidence_percentage = (outcome_counts.iloc[0] / len(next_outcomes)) * 100

    return most_likely_outcome, confidence_percentage

def find_patterns(df, patterns_to_watch):
    pattern_counts = {name: 0 for name in patterns_to_watch.keys()}
    outcomes = df['Outcome'].tolist()

    for pattern_name, pattern_sequence in patterns_to_watch.items():
        pattern_len = len(pattern_sequence)
        for i in range(len(outcomes) - pattern_len + 1):
            if outcomes[i:i+pattern_len] == pattern_sequence:
                pattern_counts[pattern_name] += 1
    return pattern_counts

def reset_deck():
    st.session_state.current_deck_id += 1
    st.session_state.played_cards = set()

    for card in PLAYER_A_FIXED_CARDS_STR:
        st.session_state.played_cards.add(card)

    st.success(f"Starting New Deck: Deck {st.session_state.current_deck_id}. Played cards reset for this deck.")

# --- AI Model Initialization (Call load_ai_model_from_drive here, before session state or UI) ---
ai_model_initial_load, label_encoder_initial_load = load_ai_model_from_drive()

# --- Session State Initialization ---
if 'rounds' not in st.session_state:
    st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

if 'current_deck_id' not in st.session_state:
    temp_gc, _ = get_gspread_and_drive_clients()
    temp_df = pd.DataFrame()
    if temp_gc:
        try:
            temp_spreadsheet = temp_gc.open("Casino Card Game Log")
            temp_worksheet = temp_spreadsheet.worksheet("Sheet1")
            temp_data = temp_worksheet.get_all_records()
            if temp_data:
                temp_df = pd.DataFrame(temp_data)
                if 'Deck_ID' in temp_df.columns and not temp_df.empty:
                    st.session_state.current_deck_id = temp_df['Deck_ID'].max()
                else:
                    st.session_state.current_deck_id = 1
            else:
                st.session_state.current_deck_id = 1
        except Exception:
            st.session_state.current_deck_id = 1
    else:
        st.session_state.current_deck_id = 1

if 'played_cards' not in st.session_state:
    st.session_state.played_cards = set()

if 'ai_model' not in st.session_state:
    st.session_state.ai_model = ai_model_initial_load
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = label_encoder_initial_load

if 'historical_patterns' not in st.session_state:
    st.session_state.historical_patterns = pd.DataFrame(columns=['Timestamp', 'Deck_ID', 'Pattern_Name', 'Pattern_Sequence', 'Start_Round_ID', 'End_Round_ID'])

# --- Load data on app startup ---
load_rounds()

st.title("Casino Card Game Tracker & Predictor")

# --- Streamlit Sidebar ---
st.sidebar.header(f"Current Deck: ID {st.session_state.current_deck_id}")
if st.sidebar.button("New Deck (Reset Learning)"):
    reset_deck()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("AI Model Management")

if st.sidebar.button("Train/Retrain AI Model"):
    with st.spinner("Training AI model... This might take a moment."):
        training_successful = train_and_save_prediction_model()
        if training_successful:
            st.session_state.ai_model, st.session_state.label_encoder = load_ai_model_from_drive()
            st.rerun()
        else:
            st.error("AI model training failed. See messages above.")

if st.session_state.ai_model and st.session_state.label_encoder:
    st.sidebar.success("AI Model Ready: ✅")
else:
    st.sidebar.warning("AI Model Not Ready: ❌ (Train it!)")


# --- Card Input Section ---
st.header("Enter Round Details")

available_cards_for_selection = [card for card in ALL_CARDS if card not in st.session_state.played_cards]

card1 = st.selectbox("Select Card 1", available_cards_for_selection, key="card1_select")
card2 = st.selectbox("Select Card 2", [c for c in available_cards_for_selection if c != card1], key="card2_select")
card3 = st.selectbox("Select Card 3", [c for c in available_cards_for_selection if c != card1 and c != card2], key="card3_select")

if card1 and card2 and card3:
    total = card_values[card1] + card_values[card2] + card_values[card3]
    st.write(f"**Calculated Total:** {total}")

    outcome = ""
    if total > 21:
        outcome = "Over 21"
        st.success("Result: Over 21")
    elif total < 21:
        outcome = "Under 21"
        st.info("Result: Under 21")
    else:
        outcome = "Exactly 21"
        st.warning("Result: Exactly 21")

    if st.button("Add Round"):
        timestamp = datetime.now()
        round_id = len(st.session_state.rounds) + 1
        new_round = {
            'Timestamp': timestamp,
            'Round_ID': round_id,
            'Card1': card1,
            'Card2': card2,
            'Card3': card3,
            'Sum': total,
            'Outcome': outcome,
            'Deck_ID': st.session_state.current_deck_id
        }
        st.session_state.rounds = pd.concat([st.session_state.rounds, pd.DataFrame([new_round])], ignore_index=True)

        st.session_state.played_cards.add(card1)
        st.session_state.played_cards.add(card2)
        st.session_state.played_cards.add(card3)

        save_rounds()
        st.rerun()
else:
    st.write("Please select all three cards to calculate the total and add the round.")

## Real-time Insights

### Current Streak

if not st.session_state.rounds.empty:
    current_deck_rounds = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id].copy()
    if not current_deck_rounds.empty:
        streak_outcome, streak_length = get_current_streak(current_deck_rounds)
        if streak_length >= STREAK_THRESHOLD:
            st.markdown(f"**Current Streak:** 🔥 {streak_length}x **{streak_outcome}** in a row! 🔥")
        elif streak_length > 0:
            st.write(f"**Current Streak:** {streak_length}x {streak_outcome}")
    else:
        st.write("No rounds played in the current deck yet to determine a streak.")
else:
    st.write("No rounds played yet.")

### Daily Tendency

if not st.session_state.rounds.empty:
    today_date = datetime.now().date()
    st.session_state.rounds['Timestamp'] = pd.to_datetime(st.session_state.rounds['Timestamp'], errors='coerce')
    daily_rounds = st.session_state.rounds[st.session_state.rounds['Timestamp'].dt.date == today_date]

    if not daily_rounds.empty:
        over_count = daily_rounds[daily_rounds['Outcome'] == 'Over 21'].shape[0]
        under_count = daily_rounds[daily_rounds['Outcome'] == 'Under 21'].shape[0]
        total_daily_outcomes = over_count + under_count

        if total_daily_outcomes > 0:
            over_percentage = over_count / total_daily_outcomes
            under_percentage = under_count / total_daily_outcomes

            st.write(f"**Today's Outcomes (Deck {st.session_state.current_deck_id}):**")
            st.write(f"- Over 21: {over_count} ({over_percentage:.1%})")
            st.write(f"- Under 21: {under_count} ({under_percentage:.1%})")
            st.write(f"- Exactly 21: {daily_rounds[daily_rounds['Outcome'] == 'Exactly 21'].shape[0]}")

            if over_percentage > OVER_UNDER_BIAS_THRESHOLD:
                st.markdown(f"📈 **Today's Trend:** Leaning towards **Over 21**!")
            elif under_percentage > OVER_UNDER_BIAS_THRESHOLD:
                st.markdown(f"📉 **Today's Trend:** Leaning towards **Under 21**!")
            else:
                st.write("📊 **Today's Trend:** Fairly balanced between Over and Under.")
        else:
            st.write("No 'Over 21' or 'Under 21' outcomes recorded for today yet.")
    else:
        st.write("No rounds recorded for today yet.")
else:
    st.write("No historical rounds to analyze daily tendency.")

st.header("Observed Patterns (Current Deck)")

if not st.session_state.rounds.empty:
    current_deck_rounds_for_patterns = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id].copy()
    if not current_deck_rounds_for_patterns.empty:
        pattern_counts = find_patterns(current_deck_rounds_for_patterns, PATTERNS_TO_WATCH)

        found_any_pattern = False
        for pattern_name, count in pattern_counts.items():
            if count > 0:
                st.write(f"- `{pattern_name}`: Found **{count}** time(s)")
                found_any_pattern = True

        if not found_any_pattern:
            st.write("No defined patterns observed in the current deck yet.")
    else:
        st.write("No rounds played in the current deck to find patterns.")
else:
    st.write("No historical rounds to find patterns.")


## Prediction Module

st.header("Next Round Prediction")

if not st.session_state.rounds.empty:
    # Get all outcomes from the current deck for pattern and AI sequence prediction
    current_deck_outcomes = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]['Outcome'].tolist()

    predicted_by_pattern = False
    pattern_prediction_outcome = None
    pattern_prediction_confidence = 0

    if len(current_deck_outcomes) >= 2: # Patterns need at least 2 outcomes to start forming
        sorted_patterns = sorted(PATTERNS_TO_WATCH.items(), key=lambda item: len(item[1]), reverse=True)

        for pattern_name, pattern_sequence in sorted_patterns:
            pattern_len = len(pattern_sequence)
            if len(current_deck_outcomes) >= pattern_len and \
               current_deck_outcomes[-pattern_len:] == pattern_sequence:

                outcome, confidence = predict_next_outcome_from_pattern(st.session_state.rounds, pattern_sequence)

                if outcome:
                    pattern_prediction_outcome = outcome
                    pattern_prediction_confidence = confidence
                    st.write(f"Based on pattern `{pattern_name}` (last {pattern_len} rounds):")
                    st.markdown(f"**Prediction:** ➡️ **{pattern_prediction_outcome}** (Confidence: {pattern_prediction_confidence:.1f}%)")
                    predicted_by_pattern = True
                    break

    ai_model_prediction_attempted = False
    ai_model_prediction_error_occurred = False

    if st.session_state.ai_model and st.session_state.label_encoder:
        # Get all outcomes from the current deck for pattern and AI sequence prediction
        current_deck_outcomes = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]['Outcome'].tolist()

        # We need enough outcomes to create the required lagged features for prediction
        if len(current_deck_outcomes) < PREDICTION_ROUNDS_CONSIDERED:
            st.info(f"AI Model needs at least {PREDICTION_ROUNDS_CONSIDERED} past rounds in the current deck to make a prediction (based on historical sequence). Only {len(current_deck_outcomes)} available.")
            ai_model_prediction_attempted = False
        else:
            ai_model_prediction_attempted = True
            st.markdown("---")
            st.subheader("🤖 AI Model's Prediction for the *Next Round* (based on recent outcomes)")

            try:
                # Get the last PREDICTION_ROUNDS_CONSIDERED outcomes from the current deck
                recent_outcomes_for_lags = current_deck_outcomes[-PREDICTION_ROUNDS_CONSIDERED:]

                # DEBUG PRINTS
                print(f"DEBUG (PREDICTION): PREDICTION_ROUNDS_CONSIDERED: {PREDICTION_ROUNDS_CONSIDERED}")
                print(f"DEBUG (PREDICTION): Length of current_deck_outcomes: {len(current_deck_outcomes)}")
                print(f"DEBUG (PREDICTION): recent_outcomes_for_lags: {recent_outcomes_for_lags}")

                # Encode these outcomes using the trained label encoder
                recent_outcomes_encoded = st.session_state.label_encoder.transform(recent_outcomes_for_lags)

                # DEBUG PRINTS
                print(f"DEBUG (PREDICTION): Length of recent_outcomes_encoded: {len(recent_outcomes_encoded)}")
                print(f"DEBUG (PREDICTION): Encoded recent outcomes: {recent_outcomes_encoded}")
                print(f"DEBUG (PREDICTION): LabelEncoder classes in session state: {st.session_state.label_encoder.classes_.tolist()}") # NEW DEBUG PRINT

                # Create a DataFrame for prediction matching the training features' structure
                prediction_features_dict = {}
                for i in range(PREDICTION_ROUNDS_CONSIDERED):
                    prediction_features_dict[f'Outcome_Lag{i+1}'] = [recent_outcomes_encoded[PREDICTION_ROUNDS_CONSIDERED - 1 - i]]

                X_predict = pd.DataFrame(prediction_features_dict)

                # **FIX 3: Ensure prediction features match trained model's features**
                if hasattr(st.session_state.ai_model, 'feature_columns_'):
                    expected_features = st.session_state.ai_model.feature_columns_
                    # Reorder columns of X_predict to match the order the model expects
                    X_predict = X_predict[expected_features]
                    print(f"DEBUG (PREDICTION): Reordered X_predict columns to match model's expected features.")
                else:
                    print(f"DEBUG (PREDICTION): Warning: Model does not have 'feature_columns_'. Relying on strict feature generation order.")

                # Final check on feature count
                if X_predict.shape[1] != PREDICTION_ROUNDS_CONSIDERED:
                    raise ValueError(f"Feature mismatch: Expected {PREDICTION_ROUNDS_CONSIDERED} features but got {X_predict.shape[1]}.")

                # DEBUG PRINTS
                print(f"DEBUG (PREDICTION): Shape of X_predict (features for prediction): {X_predict.shape}")
                print(f"DEBUG (PREDICTION): Columns of X_predict: {X_predict.columns.tolist()}")

                # --- NEW DEBUG PRINTS FOR PREDICTION INPUT ---
                print(f"DEBUG (PREDICTION): Shape of X_predict BEFORE prediction: {X_predict.shape}")
                print(f"DEBUG (PREDICTION): Columns of X_predict BEFORE prediction: {X_predict.columns.tolist()}")
                if hasattr(st.session_state.ai_model, 'feature_columns_'):
                    print(f"DEBUG (PREDICTION): Model's expected feature columns: {st.session_state.ai_model.feature_columns_}")
                if hasattr(st.session_state.ai_model, 'n_features_in_'):
                    print(f"DEBUG (PREDICTION): Model's expected number of features (n_features_in_): {st.session_state.ai_model.n_features_in_}")
                # --- END NEW DEBUG PRINTS ---

                predicted_encoded_outcome = st.session_state.ai_model.predict(X_predict)
                predicted_outcome_ai = st.session_state.label_encoder.inverse_transform(predicted_encoded_outcome)[0]

                # Get raw probabilities from the model
                proba_output = st.session_state.ai_model.predict_proba(X_predict)
                probabilities = proba_output[0] # Probabilities for the single prediction instance

                # Create a mapping from model's internal class encoding to its probability column index
                # The model's `classes_` attribute tells us the order of classes for `predict_proba` output
                model_classes_map = {cls: idx for idx, cls in enumerate(st.session_state.ai_model.classes_)}

                # Get the model's internal encoding for the predicted outcome string
                # This ensures we are looking up the correct probability based on what the model actually outputs
                if predicted_outcome_ai in model_classes_map:
                    model_internal_index = model_classes_map[predicted_outcome_ai]
                    confidence_ai = probabilities[model_internal_index] * 100
                else:
                    # If the predicted outcome was not one the model was trained to output directly
                    # (e.g., 'Exactly 21' if the training data never had it as a 'next_outcome')
                    confidence_ai = 0.0 # Assign 0% confidence if the model didn't learn to predict it
                    st.warning(f"AI Model did not output probability for '{predicted_outcome_ai}'. Setting confidence to 0%. This usually means the model was not trained on this specific outcome class in the historical data.")

                # DEBUG PRINTS (keep these updated ones)
                print(f"DEBUG (PREDICTION): Raw probabilities output shape: {proba_output.shape}")
                print(f"DEBUG (PREDICTION): Model's classes_ attribute: {st.session_state.ai_model.classes_.tolist()}")
                print(f"DEBUG (PREDICTION): Predicted outcome string: {predicted_outcome_ai}")
                print(f"DEBUG (PREDICTION): Model internal index for predicted outcome: {model_internal_index if 'model_internal_index' in locals() else 'N/A'}")
                print(f"DEBUG (PREDICTION): Confidence calculated: {confidence_ai:.1f}%")

                # Corrected indentation for the display elements
                st.markdown(f"➡️ **{predicted_outcome_ai}** (Confidence: {confidence_ai:.1f}%)")
                st.caption(f"Based on the last {PREDICTION_ROUNDS_CONSIDERED} outcomes in the current deck.")

                prob_df = pd.DataFrame({
                    'Outcome': st.session_state.label_encoder.classes_,
                    'Probability': probabilities
                }).sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

            except ValueError as e:
                st.error(f"AI Model prediction error: {e}. Ensure historical outcomes are consistent with model training and the LabelEncoder.")
                ai_model_prediction_error_occurred = True
            except Exception as e:
                st.error(f"An unexpected error occurred during AI model prediction: {e}")
                ai_model_prediction_error_occurred = True
    else:
        # This else block covers cases where the model is not ready
        st.info("AI Model is not loaded. Please train the model first to get predictions.")

else:
    st.info("No rounds played yet to provide any predictions.")

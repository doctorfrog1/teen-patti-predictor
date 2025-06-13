import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os # Keep this for local file operations like model saving
import joblib
MODEL_FOLDER_ID = "1CZepfjRZxWV_wfmEQuZLnbj9H2yAS9Ac"

# Machine Learning imports
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# Correct Google Authentication import for service accounts
from google.oauth2.service_account import Credentials # For gspread
from pydrive2.auth import GoogleAuth # For pydrive2
from pydrive2.drive import GoogleDrive # For pydrive2

# REMOVE these if they are present, as they are likely conflicting:
# from pydrive2.auth import GoogleAuth
# from pydrive2.drive import GoogleDrive
# from oauth2client.service_account import ServiceAccountCredentials # This is the old one



# --- Configuration ---
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

@st.cache_resource # Use st.cache_resource for objects like clients
def get_gspread_and_drive_clients():
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Google Cloud service account credentials not found in `st.secrets`. Please configure `secrets.toml`.")
            return None, None

        creds_dict = st.secrets["gcp_service_account"]
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive' # ESSENTIAL for pydrive2 Drive access
        ]

        # --- Authenticate for gspread (Sheets) ---
        gspread_credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(gspread_credentials)

        # --- Authenticate for pydrive2 (Drive) ---
        # pydrive2 typically expects a file path for service accounts,
        # so we'll write a temporary file.
        temp_creds_path = "gcp_service_account_creds_for_pydrive.json"
        with open(temp_creds_path, "w") as f:
            import json
            json.dump(dict(creds_dict), f)

        gauth = GoogleAuth()
        gauth.settings = {
            'oauth_scope': scopes,
            'client_config': {
                'service_account': {
                    'json_file': temp_creds_path
                }
            }
        }
        gauth.ServiceAuth() # Authenticate using the service account

        drive = GoogleDrive(gauth)

        # Clean up the temporary file immediately
        os.remove(temp_creds_path)

        return gc, drive # NOW returns both clients

    except Exception as e:
        st.error(f"Error loading Google Cloud credentials for Sheets/Drive: {e}. Please ensure st.secrets are configured correctly with service account details.")
        st.caption("For more info on Streamlit secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management")
        return None, None

# Function to delete model files from Drive (ensure this exists and works)
def delete_model_files_from_drive(drive_client, folder_id):
    model_filename = "prediction_model.joblib"
    encoder_filename = "label_encoder.joblib"
    try:
        st.info("Attempting to delete old AI model files from Google Drive...")
        existing_files = drive_client.ListFile({'q': f"'{folder_id}' in parents and (title='{model_filename}' or title='{encoder_filename}') and trashed=false"}).GetList()
        if existing_files:
            for file in existing_files:
                st.info(f"Deleting existing file: {file['title']}")
                file.Delete()
            st.info("Old AI model files deleted from Google Drive.")
        else:
            st.info("No old AI model files found on Google Drive to delete.")
    except Exception as e:
        st.error(f"Error deleting old AI model files from Google Drive: {e}")

# @st.cache_data # <-- UNCOMMENT THIS LINE if it's commented out in your file.
def load_all_historical_rounds_from_sheet():
    gc, _ = get_gspread_and_drive_clients()
    if gc is None:
        # Return empty DataFrame with all expected columns if client failed
        return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

    try:
        spreadsheet = gc.open("Casino Card Game Log")
        worksheet = spreadsheet.worksheet("Sheet1")
        data = worksheet.get_all_records()
        if data:
            df = pd.DataFrame(data)
            # Ensure column names are consistent
            df.columns = df.columns.str.replace(' ', '_')

            # --- AGGRESSIVE CLEANING FOR 'Outcome' COLUMN ---
            df['Outcome'] = df['Outcome'].astype(str).str.strip() # Convert to string, remove whitespace

            # Create a temporary column to hold the standardized short form (O, U, E)
            df['Standard_Outcome_Char'] = df['Outcome'].apply(lambda x: {
                "Over 21": "O",
                "Under 21": "U",
                "Exactly 21": "E"
            }.get(x, x[0] if isinstance(x, str) and x and x[0] in ['O', 'U', 'E'] else None))

            # Now, map these standardized chars back to the full strings
            df['Outcome'] = df['Standard_Outcome_Char'].map({
                "O": "Over 21",
                "U": "Under 21",
                "E": "Exactly 21"
            })
            # Drop the temporary column
            df = df.drop(columns=['Standard_Outcome_Char'])
            # --- END AGGRESSIVE CLEANING ---

            # Filter out any outcomes that are still not in our expected list after cleaning
            valid_outcomes = ['Over 21', 'Under 21', 'Exactly 21']
            df = df[df['Outcome'].isin(valid_outcomes)]

            # Ensure Deck_ID is handled correctly, even if it wasn't in original data or was problematic
            if 'Deck_ID' in df.columns:
                df['Deck_ID'] = pd.to_numeric(df['Deck_ID'], errors='coerce').fillna(1).astype(int)
            else:
                df['Deck_ID'] = 1 # Default to 1 if not found

            # Ensure 'Timestamp' is datetime type for consistency
            if 'Timestamp' in df.columns:
                # Coerce errors will turn invalid dates into NaT (Not a Time)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                # Fill NaT with a default timestamp if needed (e.g., yesterday)
                df['Timestamp'] = df['Timestamp'].fillna(datetime.now() - timedelta(days=1))
            else:
                # If timestamp column is missing, assign a default
                df['Timestamp'] = datetime.now() - timedelta(days=1)

            return df
        else:
            # Return empty DataFrame with all expected columns if no data found
            return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
    except SpreadsheetNotFound:
        st.error(f"Google Sheet 'Casino Card Game Log' not found. Please ensure the sheet exists and the service account has access.")
        return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
    except Exception as e:
        st.error(f"Error loading historical rounds from Google Sheet: {e}. Starting with empty history.")
        return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
        
# Ensure this is at the top with other imports: import os, joblib, sklearn, pandas, etc.
# MODEL_FOLDER_ID must be defined at the top level of your script, outside functions.

def train_and_save_prediction_model():
    # Get both gspread client and pydrive2 drive client
    gc, drive = get_gspread_and_drive_clients()

    if not (gc and drive): # Check if both clients were successfully obtained
        st.error("AI model training failed. Could not connect to Google Cloud (Sheets or Drive).")
        return False

    st.info("Preparing data for AI model training...")
    all_rounds_df = load_all_historical_rounds_from_sheet() # This function should have the strong cleaning logic

    if all_rounds_df.empty:
        st.warning("No historical data available to train the AI model.")
        # If no data, delete old models from Drive
        delete_model_files_from_drive(drive, MODEL_FOLDER_ID) # Pass drive client and folder ID
        return False

    # Filter for today's data (if you still want this specific filter)
    if 'Timestamp' in all_rounds_df.columns:
        all_rounds_df['Timestamp'] = pd.to_datetime(all_rounds_df['Timestamp'], errors='coerce')
        all_rounds_df.dropna(subset=['Timestamp'], inplace=True)
        today = datetime.now().strftime("%Y-%m-%d")
        recent_rounds_df = all_rounds_df[all_rounds_df['Timestamp'].dt.strftime("%Y-%m-%d") == today].copy()
    else:
        st.warning("No 'Timestamp' column found in historical data. Training on all available data.")
        recent_rounds_df = all_rounds_df.copy() # Train on all data if no timestamp column

    MIN_ROUNDS_FOR_TRAINING = 4 # You might want to adjust this threshold or remove for testing
    if len(recent_rounds_df) < MIN_ROUNDS_FOR_TRAINING:
        st.warning(f"Not enough recent rounds data to train the AI model. Need at least {MIN_ROUNDS_FOR_TRAINING} rounds from today. Found {len(recent_rounds_df)}. Training on ALL available data instead.")
        recent_rounds_df = all_rounds_df.copy()
        if len(recent_rounds_df) < MIN_ROUNDS_FOR_TRAINING:
            st.warning(f"Still not enough total historical data. Need at least {MIN_ROUNDS_FOR_TRAINING} rounds. Found {len(recent_rounds_df)}.")
            return False


    # Extract features (X) and labels (y)
    for col in ['Card1', 'Card2', 'Card3', 'Sum']:
        if col in recent_rounds_df.columns:
            recent_rounds_df[col] = pd.to_numeric(recent_rounds_df[col], errors='coerce').fillna(0)
        else:
            st.warning(f"Missing column '{col}'. Filling with zeros for training.")
            recent_rounds_df[col] = 0

    recent_rounds_df.dropna(subset=['Card1', 'Card2', 'Card3', 'Sum'], inplace=True)

    X = recent_rounds_df[['Card1', 'Card2', 'Card3', 'Sum']]
    y = recent_rounds_df['Outcome']

    valid_outcomes = ['Over 21', 'Under 21', 'Exactly 21']
    y = y[y.isin(valid_outcomes)]

    if X.empty or y.empty or len(X) != len(y):
        st.error("After data preparation, features (X) or outcomes (y) are empty or mismatched. AI model training failed.")
        delete_model_files_from_drive(drive, MODEL_FOLDER_ID) # Pass drive client and folder ID
        return False

    le = LabelEncoder()
    try:
        le.fit(['Over 21', 'Under 21', 'Exactly 21']) # Ensure these are the only expected outcomes
    except Exception as e:
        st.error(f"Error fitting LabelEncoder: {e}. Check expected outcomes.")
        return False

    try:
        y_encoded = le.transform(y)
    except ValueError as e:
        st.error(f"Error during encoding: {e}. This likely means an outcome appeared in your data that the LabelEncoder was not fitted on. Ensure your sheet data is clean and matches expected outcomes ('Over 21', 'Under 21', 'Exactly 21').")
        st.error("AI model training failed. See messages above.")
        delete_model_files_from_drive(drive, MODEL_FOLDER_ID)
        return False

    if len(pd.Series(y_encoded).unique()) < 2:
        st.error(f"Error during AI model training: This solver needs samples of at least 2 classes in the data, but the data contains only one class after encoding. Found: {le.inverse_transform(pd.Series(y_encoded).unique())}")
        st.error("AI model training failed. Please ensure your Google Sheet has rounds with different outcomes (e.g., 'Over 21' AND 'Under 21').")
        delete_model_files_from_drive(drive, MODEL_FOLDER_ID)
        return False

    st.info(f"Training AI model with {len(X)} samples.")

    model = LogisticRegression(max_iter=1000, random_state=42)
    try:
        model.fit(X, y_encoded)
    except Exception as e:
        st.error(f"Error during model fitting (Logistic Regression): {e}")
        st.error("AI model training failed.")
        delete_model_files_from_drive(drive, MODEL_FOLDER_ID)
        return False

    st.session_state.ai_model = model
    st.session_state.label_encoder = le

    # --- Call the Google Drive save function ---
    success = save_ai_model_to_drive(model, le, drive, MODEL_FOLDER_ID)

    if success:
        st.success("AI prediction model trained and loaded into session state!")
        return True
    else:
        st.error("Failed to save AI model to Google Drive. Training complete but model not persistently stored.")
        return False

def save_ai_model_to_drive(model, label_encoder, drive_client, folder_id):
    model_filename = "prediction_model.joblib"
    encoder_filename = "label_encoder.joblib"

    try:
        st.info("Saving AI model to Google Drive...")

        # Save locally first as temporary files
        joblib.dump(model, model_filename)
        joblib.dump(label_encoder, encoder_filename)

        # Check if files exist on Drive and delete before uploading new ones
        existing_files = drive_client.ListFile({'q': f"'{folder_id}' in parents and (title='{model_filename}' or title='{encoder_filename}') and trashed=false"}).GetList()
        for file in existing_files:
            st.info(f"Deleting existing file on Drive: {file['title']}")
            file.Delete()

        # Upload model file
        file_model = drive_client.CreateFile({'title': model_filename, 'parents': [{'id': folder_id}]})
        file_model.SetContentFile(model_filename)
        file_model.Upload()
        os.remove(model_filename) # Clean up local temp file

        # Upload LabelEncoder file
        file_encoder = drive_client.CreateFile({'title': encoder_filename, 'parents': [{'id': folder_id}]})
        file_encoder.SetContentFile(encoder_filename)
        file_encoder.Upload()
        os.remove(encoder_filename) # Clean up local temp file

        st.success("AI prediction model and LabelEncoder saved successfully to Google Drive!")
        return True
    except Exception as e:
        st.error(f"Error saving AI model to Google Drive: {e}")
        # Ensure cleanup if saving fails
        if os.path.exists(model_filename): os.remove(model_filename)
        if os.path.exists(encoder_filename): os.remove(encoder_filename)
        return False

@st.cache_resource(ttl=3600) # Cache the loaded model for 1 hour
def load_ai_model_from_drive():
    gc, drive = get_gspread_and_drive_clients() # Get clients for loading
    if not (gc and drive):
        st.sidebar.error("Could not connect to Google Drive to load AI model.")
        return None, None # Return None for both model and encoder

    model_filename = "prediction_model.joblib"
    encoder_filename = "label_encoder.joblib"
    folder_id = MODEL_FOLDER_ID # Use the same folder ID as for saving

    model = None
    encoder = None

    try:
        st.sidebar.info("Attempting to load AI model from Google Drive...")

        # Search for the model and encoder files in the specified folder
        model_file_list = drive.ListFile({'q': f"'{folder_id}' in parents and title='{model_filename}' and trashed=false"}).GetList()
        encoder_file_list = drive.ListFile({'q': f"'{folder_id}' in parents and title='{encoder_filename}' and trashed=false"}).GetList()

        if not model_file_list or not encoder_file_list:
            st.sidebar.warning(f"AI Prediction Model '{model_filename}' or '{encoder_filename}' not found on Google Drive in folder ID {folder_id}. Please train the model first.")
            return None, None

        # Download files to temporary local paths
        temp_model_path = "temp_prediction_model.joblib"
        temp_encoder_path = "temp_label_encoder.joblib"

        model_file_list[0].GetContentFile(temp_model_path)
        encoder_file_list[0].GetContentFile(temp_encoder_path)

        # Load models from temporary local paths
        model = joblib.load(temp_model_path)
        encoder = joblib.load(temp_encoder_path)

        st.sidebar.success("AI Prediction Model Loaded from Google Drive.")
        return model, encoder
    except Exception as e:
        st.sidebar.error(f"Error loading AI model from Google Drive: {e}")
        return None, None
    finally:
        # Clean up temporary local files
        if os.path.exists(temp_model_path): os.remove(temp_model_path)
        if os.path.exists(temp_encoder_path): os.remove(temp_encoder_path)



def load_rounds():
    gc, _ = get_gspread_and_drive_clients() # Corrected
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
    gc, _ = get_gspread_and_drive_clients() # Corrected
    if not gc:
        st.warning("Cannot save rounds: Google Sheets client not available.")
        return

    try:
        spreadsheet = gc.open("Casino Card Game Log")
        worksheet = spreadsheet.worksheet("Sheet1")

        data_to_write = [st.session_state.rounds.columns.tolist()] + st.session_state.rounds.astype(str).values.tolist()

        worksheet.clear()
        worksheet.update('A1', data_to_write)

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
# This loads the model once when the app starts from Google Drive
# This MUST be placed here, at the very top level of your script,
# before any st.session_state access or Streamlit UI elements are defined.
ai_model_initial_load, label_encoder_initial_load = load_ai_model_from_drive()


# --- Session State Initialization ---
if 'rounds' not in st.session_state:
    st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

if 'current_deck_id' not in st.session_state:
    # Use the corrected function call here
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
    all_historical_rounds = load_all_historical_rounds_from_sheet()
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
    current_deck_outcomes = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]['Outcome'].tolist()

    predicted_by_pattern = False
    pattern_prediction_outcome = None
    pattern_prediction_confidence = 0

    if len(current_deck_outcomes) >= 2:
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

    # The AI model predicts based on the *current hand* (Card1, Card2, Card3, Sum),
# not based on a sequence of previous outcomes.
# So, we need the inputs for the *current* selection
ai_model_prediction_attempted = False
ai_model_prediction_error_occurred = False

# Only attempt AI prediction if all 3 cards are selected AND the model is loaded
# The 'card1', 'card2', 'card3' variables are from your "Enter Round Details" section at the top.
# They hold the currently selected cards for the *next* round to be played.
if card1 and card2 and card3 and st.session_state.ai_model and st.session_state.label_encoder:
    ai_model_prediction_attempted = True
    st.markdown("---")
    st.subheader("AI Model's Prediction for the *current hand*") # Changed text for clarity
    try:
        # Re-calculate total here using selected cards to ensure it's up-to-date
        # These values (card_values[card1], current_total) are the FEATURES for the prediction
        current_total = card_values[card1] + card_values[card2] + card_values[card3]

        # Create a DataFrame with the current hand's features, matching training X
        # The model expects columns: 'Card1', 'Card2', 'Card3', 'Sum'
        current_hand_features = pd.DataFrame({
            'Card1': [card_values[card1]], # Value of the first selected card
            'Card2': [card_values[card2]], # Value of the second selected card
            'Card3': [card_values[card3]], # Value of the third selected card
            'Sum': [current_total]          # Sum of the selected cards
        })

        # Make the prediction using the current hand's features
        predicted_encoded_outcome = st.session_state.ai_model.predict(current_hand_features)
        predicted_outcome_ai = st.session_state.label_encoder.inverse_transform(predicted_encoded_outcome)[0]

        # Get probabilities
        probabilities = st.session_state.ai_model.predict_proba(current_hand_features)[0] # <--- Use current_hand_features here too!
        confidence_ai = probabilities[st.session_state.label_encoder.transform([predicted_outcome_ai])[0]] * 100

        st.markdown(f"🤖 **AI Model Prediction:** ➡️ **{predicted_outcome_ai}** (Confidence: {confidence_ai:.1f}%)")
        st.caption("Based on the currently selected cards for the next round.") # Changed caption for clarity

        prob_df = pd.DataFrame({
            'Outcome': st.session_state.label_encoder.classes_,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=False)
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

    except ValueError as e:
        st.error(f"AI Model prediction error: {e}. This means the input data for prediction might be inconsistent with training data (e.g., non-numeric values for cards/sum).")
        ai_model_prediction_error_occurred = True
    except Exception as e:
        st.error(f"An unexpected error occurred during AI model prediction: {e}")
        ai_model_prediction_error_occurred = True
else:
    # This else block covers cases where cards are not selected or model is not ready
    if not (card1 and card2 and card3):
        st.info("Select all three cards to see the AI Model's Prediction for this hand.")
    elif not (st.session_state.ai_model and st.session_state.label_encoder):
        st.info("AI Model is not loaded. Please train the model first to get predictions.")

# You can keep this part if you want a fallback message for no patterns/AI prediction, otherwise remove it
# This specifically checks if neither pattern prediction nor AI prediction was attempted/successful
# if not predicted_by_pattern and not ai_model_prediction_attempted:
#     st.write("No strong pattern observed, and AI model prediction not available for this hand.")

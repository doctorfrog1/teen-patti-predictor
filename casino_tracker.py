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
            streak_count += 1
        else:
            break
    return current_outcome, streak_count

def predict_next_outcome_from_pattern(df_all_rounds, pattern_sequence):
    """
    Analyzes historical data to predict the next outcome after a given pattern.
    Args:
        df_all_rounds (pd.DataFrame): The full DataFrame of all historical rounds.
        pattern_sequence (list): The sequence of outcomes to look for (e.g., ['Over 21', 'Over 21']).
    Returns:
        tuple: (most_likely_outcome, confidence_percentage) or (None, 0) if no data.
    """
    if df_all_rounds.empty or not pattern_sequence:
        return None, 0

    next_outcomes = []
    pattern_len = len(pattern_sequence)

    # Iterate through all rounds, checking in each deck context
    # Group by Deck_ID to prevent patterns from crossing deck boundaries
    for deck_id, deck_df in df_all_rounds.groupby('Deck_ID'):
        outcomes_in_deck = deck_df['Outcome'].tolist()

        for i in range(len(outcomes_in_deck) - pattern_len): # -pattern_len because we need a subsequent outcome
            if outcomes_in_deck[i : i + pattern_len] == pattern_sequence:
                if (i + pattern_len) < len(outcomes_in_deck): # Ensure there IS a next outcome
                    next_outcomes.append(outcomes_in_deck[i + pattern_len])

    if not next_outcomes:
        return None, 0

    outcome_counts = pd.Series(next_outcomes).value_counts()
    most_likely_outcome = outcome_counts.index[0]
    confidence_percentage = (outcome_counts.iloc[0] / len(next_outcomes)) * 100

    return most_likely_outcome, confidence_percentage

def find_patterns(df, patterns_to_watch):
    """
    Detects predefined sequences (patterns) in the outcomes of a DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with an 'Outcome' column.
        patterns_to_watch (dict): A dictionary where keys are pattern names
                                  and values are lists of outcomes.
    Returns:
        dict: Counts of how many times each pattern was found.
    """
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
    st.session_state.played_cards = set() # This clears played cards for the NEW deck
    update_deck_ids_sheet(st.session_state.current_deck_id) # Save new deck ID to sheet

    # Player A's fixed cards are re-added immediately for the new deck
    for card in PLAYER_A_FIXED_CARDS_STR:
        st.session_state.played_cards.add(card)

    st.success(f"Starting New Deck: Deck {st.session_state.current_deck_id}. Played cards reset for this deck.")

# --- Define PATTERNS_TO_WATCH at the top level ---
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


# --- AI Model Initialization (Call load_ai_model here, before session state or UI) ---
# This loads the model once when the app starts from Streamlit App Data
# This MUST be placed here, at the very top level of your script,
# before any st.session_state access or Streamlit UI elements are defined.
ai_model_initial_load, label_encoder_initial_load = load_ai_model()


# --- Session State Initialization ---
if 'rounds' not in st.session_state:
    st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

# Initialize current_deck_id only if it's not already in session state
if 'current_deck_id' not in st.session_state:
    gc_temp, _ = get_gspread_and_drive_clients() # Use the combined getter here
    temp_df = pd.DataFrame()
    if gc_temp:
        try:
            temp_spreadsheet = gc_temp.open(SPREADSHEET_NAME) # Use constant
            temp_worksheet = temp_spreadsheet.worksheet(SHEET_NAME) # Use constant
            temp_data = temp_worksheet.get_all_records()
            if temp_data:
                temp_df = pd.DataFrame(temp_data)
                if 'Deck_ID' in temp_df.columns and not temp_df.empty:
                    # Ensure Deck_ID is numeric before max()
                    temp_df['Deck_ID'] = pd.to_numeric(temp_df['Deck_ID'], errors='coerce').fillna(1).astype(int)
                    st.session_state.current_deck_id = temp_df['Deck_ID'].max()
                else:
                    st.session_state.current_deck_id = 1
            else:
                st.session_state.current_deck_id = 1
        except Exception as e:
            st.warning(f"Could not load Deck_ID from sheet on startup: {e}. Defaulting to Deck 1.")
            st.session_state.current_deck_id = 1
    else:
        st.session_state.current_deck_id = 1

if 'played_cards' not in st.session_state:
    st.session_state.played_cards = set()

# --- NEW: Ensure AI model and encoder are in session state for later updates ---
# These will be updated by the 'Train AI Model' button click
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = ai_model_initial_load
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = label_encoder_initial_load

if 'historical_patterns' not in st.session_state: # Ensure this is initialized if not already
    st.session_state.historical_patterns = pd.DataFrame(columns=['Timestamp', 'Deck_ID', 'Pattern_Name', 'Pattern_Sequence', 'Start_Round_ID', 'End_Round_ID'])


# --- Load data on app startup ---
# This will call the revised load_rounds
load_rounds()

st.title("Casino Card Game Tracker & Predictor")

# --- Streamlit Sidebar ---
st.sidebar.header(f"Current Deck: ID {st.session_state.current_deck_id}")
if st.sidebar.button("New Deck (Reset Learning)"):
    reset_deck()
    st.rerun() # Rerun to update the available cards list and clear display

# --- NEW: AI Model Training Button in Sidebar ---
st.sidebar.markdown("---") # Separator
st.sidebar.subheader("AI Model Management")

if st.sidebar.button("Train/Retrain AI Model"):
    # Load all historical rounds for training.
    # This will be filtered for today's data inside the training function.
    all_historical_rounds = load_all_historical_rounds_from_sheet()
    with st.spinner("Training AI model... This might take a moment."):
        # The training function now returns True/False based on success
        training_successful = train_and_save_prediction_model(all_historical_rounds)
        if training_successful:
            # If training is successful, reload the model into memory
            # This ensures the app uses the newly trained model immediately
            st.session_state.ai_model, st.session_state.label_encoder = load_ai_model()
            st.rerun() # Rerun to update prediction with new model
        else:
            st.error("AI model training failed. See messages above.")

# Display AI model status in sidebar (optional but helpful)
if st.session_state.ai_model and st.session_state.label_encoder:
    st.sidebar.success("AI Model Ready: ✅")
else:
    st.sidebar.warning("AI Model Not Ready: ❌ (Train it!)")


# --- Card Input Section ---
st.header("Enter Round Details")

# Filter available cards by removing played cards and Player A's fixed cards
available_cards_for_selection = [card for card in ALL_CARDS if card not in st.session_state.played_cards]


card1 = st.selectbox("Select Card 1", available_cards_for_selection, key="card1_select")
card2 = st.selectbox("Select Card 2", [c for c in available_cards_for_selection if c != card1], key="card2_select")
card3 = st.selectbox("Select Card 3", [c for c in available_cards_for_selection if c != card1 and c != card2], key="card3_select")

# Calculate total
if card1 and card2 and card3: # Ensure all cards are selected before calculating
    total = card_values[card1] + card_values[card2] + card_values[card3]
    st.write(f"**Calculated Total:** {total}")

    # Determine outcome
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

    # Add round button
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

        # Add cards to played_cards set for the current deck
        st.session_state.played_cards.add(card1)
        st.session_state.played_cards.add(card2)
        st.session_state.played_cards.add(card3)

        save_rounds()
        st.rerun() # Rerun to update displays and available cards
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

    # --- Pattern-Based Prediction Attempt ---
    predicted_by_pattern = False
    pattern_prediction_outcome = None
    pattern_prediction_confidence = 0

    if len(current_deck_outcomes) >= 2: # Need at least two outcomes to start matching patterns
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

    # --- NEW: AI Model Prediction Attempt ---
    ai_model_prediction_attempted = False
    ai_model_prediction_error_occurred = False # Flag to manage fallback logic

    if st.session_state.ai_model and st.session_state.label_encoder: # Check if model/encoder are loaded
        ai_model_prediction_attempted = True # AI prediction was attempted
        st.markdown("---") # Separator for AI Prediction
        st.subheader("AI Model's Prediction")

        # Get latest rounds for AI prediction, this needs to be numerical features
        all_historical_rounds = load_all_historical_rounds_from_sheet()
        current_features_df = get_latest_rounds_for_prediction(all_historical_rounds, SEQUENCE_LENGTH)

        if current_features_df is not None and not current_features_df.empty:
            try:
                model = st.session_state.ai_model
                le = st.session_state.label_encoder
                
                # Make prediction
                # predict_proba returns probabilities for each class
                prediction_proba = model.predict_proba(current_features_df)
                
                # Get the predicted class index
                predicted_class_index = model.predict(current_features_df)[0]
                
                # --- Debugging Prediction ---
                st.write("--- Debugging AI Model Prediction ---")
                st.write(f"Prediction input features: {current_features_df.values}")
                st.write(f"Raw prediction probabilities: {prediction_proba}")
                st.write(f"Predicted class index from model: {predicted_class_index}")
                st.write(f"LabelEncoder classes (at prediction time): {le.classes_}")
                st.write(f"Length of LabelEncoder classes: {len(le.classes_)}")
                
                # Ensure the predicted_class_index is within bounds of the LabelEncoder's classes
                if predicted_class_index < len(le.classes_):
                    predicted_outcome = le.inverse_transform([predicted_class_index])[0]
                    confidence = prediction_proba[0, predicted_class_index] * 100

                    st.info(f"AI Prediction: {predicted_outcome} (Confidence: {confidence:.1f}%)")
                else:
                    st.error(f"AI Model prediction error: Predicted class index {predicted_class_index} is out of bounds for LabelEncoder with classes of size {len(le.classes_)}. This often means the model was trained on more outcome classes than the encoder knows about, or vice-versa. Ensure consistency in training data.")
                    st.info("Try retraining the model, ensuring diverse outcomes (Under 21, Over 21, Exactly 21) are present in your training data.")
                    ai_model_prediction_error_occurred = True

                st.write("--- End Debugging AI Model Prediction ---")

            except Exception as e:
                st.error(f"AI Model prediction error: {str(e)}. Ensure model is trained and data is consistent.")
                st.info("Try retraining the model if this persists, ensuring diverse outcomes (Under 21, Over 21, Exactly 21) are present in your training data.")
                ai_model_prediction_error_occurred = True
        else:
            st.warning(f"AI model needs at least {SEQUENCE_LENGTH} recent rounds (sums) for prediction. Play more rounds!")
            ai_model_prediction_error_occurred = True # Treat as error for fallback purposes
    elif not st.session_state.ai_model:
        st.warning("AI Prediction Model not loaded. Please train it using the 'Train AI Model' button in the sidebar.")
        ai_model_prediction_attempted = True # Still attempted, but not ready


    # --- Existing Fallback to Simple Frequency-Based Prediction (Adjusted) ---
    # Only show this if no pattern prediction AND no successful AI prediction was made.
    # It will show if AI was not ready, or had an error, or insufficient rounds for AI.
    if not predicted_by_pattern and (not ai_model_prediction_attempted or ai_model_prediction_error_occurred):
        st.markdown("---") # Separator for Simple Prediction
        st.subheader("Simple Frequency Prediction")
        recent_rounds = st.session_state.rounds[
             st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id
            ].tail(PREDICTION_ROUNDS_CONSIDERED)

        if not recent_rounds.empty:
            outcome_counts = recent_rounds['Outcome'].value_counts()

            # Exclude 'Exactly 21' for 'Over'/'Under' bias
            outcome_counts_for_bias = outcome_counts.drop(labels='Exactly 21', errors='ignore')

            if not outcome_counts_for_bias.empty:
                predicted_outcome = outcome_counts_for_bias.idxmax()
                confidence = (outcome_counts_for_bias.max() / outcome_counts_for_bias.sum()) * 100
                st.info(f"Simple Prediction (last {len(recent_rounds)} rounds): ➡️ **{predicted_outcome}** (Confidence: {confidence:.1f}%)")
                st.caption("This prediction is based on the most frequent outcome in recent rounds, excluding 'Exactly 21'.")
            else:
                st.write("Not enough 'Over 21' or 'Under 21' outcomes in recent rounds for a simple frequency prediction.")
        else:
            st.write("No recent rounds in the current deck to make a simple frequency prediction.")
else:
    st.warning("No rounds played yet to make any predictions.")

st.markdown("---")
st.subheader("All Round History")
if not st.session_state.rounds.empty:
    st.dataframe(st.session_state.rounds.sort_values(by='Timestamp', ascending=False), use_container_width=True)
else:
    st.write("No rounds recorded yet.")

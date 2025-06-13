import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials # Ensure this is imported



# --- Configuration ---
PLAYER_A_FIXED_CARDS_STR = {'J♣', '10♠', '9♠'}
PREDICTION_ROUNDS_CONSIDERED = 10
STREAK_THRESHOLD = 3
OVER_UNDER_BIAS_THRESHOLD = 0.6

# --- AI Configuration ---
SEQUENCE_LENGTH = 3
MODEL_FILE = "prediction_model.joblib"
ENCODER_FILE = "label_encoder.joblib"
MODEL_DIR = ".streamlit/data"

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
    """Authenticates with Google Sheets and Google Drive using service account credentials."""
    try:
        # Get credentials as a standard dictionary from st.secrets
        creds_info = dict(st.secrets.gcp_service_account)

        # For gspread (no change here, this part is robust)
        gc = gspread.service_account_from_dict(creds_info)

        # For pydrive2: Authenticate using ServiceAccountCredentials directly
        # Define the scopes for Google Drive and Google Sheets
        scope = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]

        # Create credentials object from the service account info
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)

        # Initialize GoogleAuth with the credentials
        gauth = GoogleAuth()
        gauth.credentials = creds

        # Important: Prevent pydrive2 from trying to save or load credential files
        gauth.LoadCredentialsFile = lambda: None
        gauth.SaveCredentialsFile = lambda: None

        # No need to call gauth.Authenticate() explicitly if credentials are set directly like this
        # The GoogleDrive object will use the set credentials.

        drive = GoogleDrive(gauth)

        return gc, drive
    except Exception as e:
        st.error(f"Error loading Google Cloud credentials for Sheets/Drive: {e}. Please ensure st.secrets are configured correctly with service account details.")
        st.stop()
        return None, None # Ensure return None for both if error occurs

# Function to delete model files from Drive (ensure this exists and works)
def delete_model_files_from_drive():
    gc, drive = get_gspread_and_drive_clients()
    if not drive:
        st.error("Could not connect to Google Drive to delete old model files.")
        return

    # Replace with your actual folder ID
    folder_id = "1CZepfjRZxWV_wfmEQuZLnbj9H2yAS9Ac" # Your specific folder ID

    try:
        # Search for model files within the specified folder
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and (title='prediction_model.joblib' or title='label_encoder.joblib') and trashed=false"}).GetList()
        for file in file_list:
            st.info(f"Deleting old model file: {file['title']} (ID: {file['id']})")
            file.Delete()
    except Exception as e:
        st.error(f"Error deleting old model files from Google Drive: {e}")

@st.cache_data # Use st.cache_data for functions that return dataframes
def load_all_historical_rounds_from_sheet():
    gc, _ = get_gspread_and_drive_clients()
    if gc is None:
        return pd.DataFrame() # Return empty DataFrame if client failed

    try:
        spreadsheet = gc.open("Casino Card Game Log") # This needs to be the literal string
        worksheet = spreadsheet.worksheet("Sheet1") # Assuming your sheet is named "Sheet1"
        data = worksheet.get_all_records()
        if data:
            df = pd.DataFrame(data)
            # Ensure column names are consistent
            df.columns = df.columns.str.replace(' ', '_') # Replace spaces in column names with underscores
            # --- NEW CLEANING STEP HERE ---
            # Standardize the 'Outcome' column to expected values
            df['Outcome'] = df['Outcome'].astype(str).str.strip() # Convert to string, remove whitespace
            df['Outcome'] = df['Outcome'].replace({
                "Over 21": "Over 21",
                "Under 21": "Under 21",
                "Exactly 21": "Exactly 21",
                "Under 21_U": "Under 21", # Explicitly map the problematic value
                "Under 21_1": "Under 21", # Map any other known problematic values
                # Add more mappings if you discover other variations in your sheet
            })
            # Filter out any outcomes that are still not in our expected list after cleaning
            valid_outcomes = ['Over 21', 'Under 21', 'Exactly 21']
            df = df[df['Outcome'].isin(valid_outcomes)]
            # --- END NEW CLEANING STEP ---

            return df
        else:
            return pd.DataFrame()
    except SpreadsheetNotFound:
        st.error(f"Google Sheet 'Casino Card Game Log' not found. Please ensure the sheet exists and the service account has access.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading historical rounds from Google Sheet: {e}. Starting with empty history.")
        return pd.DataFrame()

def train_and_save_prediction_model():
    gc, drive = get_gspread_and_drive_clients()
    if not (gc and drive):
        st.error("AI model training failed. Could not connect to Google Cloud.")
        return False

    st.info("Preparing data for AI model training...")
    all_rounds_df = load_all_historical_rounds_from_sheet()

    if all_rounds_df.empty:
        st.warning("No historical data available to train the AI model.")
        delete_model_files_from_drive() # Clear any old models if data is empty
        return False

    # Filter for today's data (or whatever logic you prefer for training subset)
    today = datetime.now().strftime("%Y-%m-%d")
    # Ensure 'Timestamp' column is datetime type before filtering
    all_rounds_df['Timestamp'] = pd.to_datetime(all_rounds_df['Timestamp'])
    recent_rounds_df = all_rounds_df[all_rounds_df['Timestamp'].dt.strftime("%Y-%m-%d") == today].copy()

    if len(recent_rounds_df) < 4: # You might want to adjust this threshold or remove for testing
        st.warning(f"Not enough recent rounds data to train the AI model. Need at least 4 rounds from today. Found {len(recent_rounds_df)}.")
        st.info("Cleared old AI model files due to insufficient data.")
        delete_model_files_from_drive()
        return False

    # Extract features (X) and labels (y)
    # Ensure relevant columns are numeric, coerce errors if necessary
    for col in ['Card1', 'Card2', 'Card3', 'Sum']:
        if col in recent_rounds_df.columns:
            recent_rounds_df[col] = pd.to_numeric(recent_rounds_df[col], errors='coerce').fillna(0) # Handle non-numeric gracefully

    X = recent_rounds_df[['Card1', 'Card2', 'Card3', 'Sum']]
    y = recent_rounds_df['Outcome']

    # Handle cases where 'y' might still contain unexpected values despite sheet cleaning (e.g., if cleaning fails)
    # This is a fallback to ensure the LabelEncoder receives only expected labels
    valid_outcomes = ['Over 21', 'Under 21', 'Exactly 21']
    y = y[y.isin(valid_outcomes)] # Filter y to only valid outcomes BEFORE encoding

    if y.empty:
        st.error("No valid outcome data found for training after filtering. AI model training failed.")
        delete_model_files_from_drive()
        return False

    # Initialize and fit the LabelEncoder
    le = LabelEncoder()
    # Fit the encoder on all *possible* outcome classes to avoid 'unseen labels' errors.
    le.fit(['Over 21', 'Under 21', 'Exactly 21']) # THIS IS CRITICAL AND MUST BE EXACT.

    try:
        y_encoded = le.transform(y)
    except ValueError as e:
        st.error(f"Error during encoding: {e}. This likely means an outcome appeared in your data that the LabelEncoder was not fitted on. Ensure all historical outcomes are used to fit the encoder and data is clean.")
        st.error("AI model training failed. See messages above.")
        delete_model_files_from_drive()
        return False

    # Check if there are at least two unique classes for training
    if len(pd.Series(y_encoded).unique()) < 2: # Use pd.Series to handle potentially empty or single-value numpy array
        st.error(f"Error during AI model training or saving: This solver needs samples of at least 2 classes in the data, but the data contains only one class after encoding. Found: {le.inverse_transform(pd.Series(y_encoded).unique())}")
        st.error("AI model training failed. See messages above.")
        delete_model_files_from_drive() # Delete potentially corrupted models
        return False

    st.info(f"Training AI model with {len(recent_rounds_df)} samples from today's data...")

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
    model.fit(X, y_encoded)

    # Save the trained model and LabelEncoder to Google Drive
    model_filename = "prediction_model.joblib"
    encoder_filename = "label_encoder.joblib"
    folder_id = "1CZepfjRZxWV_wfmEQuZLnbj9H2yAS9Ac" # Your specific folder ID

    try:
        # Check if files exist and delete before uploading new ones (important for retraining)
        existing_files = drive.ListFile({'q': f"'{folder_id}' in parents and (title='{model_filename}' or title='{encoder_filename}') and trashed=false"}).GetList()
        for file in existing_files:
            st.info(f"Deleting existing file: {file['title']}")
            file.Delete()

        # Save model
        joblib.dump(model, model_filename)
        file_model = drive.CreateFile({'title': model_filename, 'parents': [{'id': folder_id}]})
        file_model.SetContentFile(model_filename)
        file_model.Upload()
        os.remove(model_filename) # Clean up local file

        # Save LabelEncoder
        joblib.dump(le, encoder_filename)
        file_encoder = drive.CreateFile({'title': encoder_filename, 'parents': [{'id': folder_id}]})
        file_encoder.SetContentFile(encoder_filename)
        file_encoder.Upload()
        os.remove(encoder_filename) # Clean up local file

        st.success("AI prediction model trained and saved successfully to Google Drive!")
        return True
    except Exception as e:
        st.error(f"Error during AI model training or saving: {e}")
        st.error("AI model training failed. See messages above.")
        # Ensure cleanup if saving fails
        if os.path.exists(model_filename): os.remove(model_filename)
        if os.path.exists(encoder_filename): os.remove(encoder_filename)
        return False

@st.cache_resource
def load_ai_model():
    model = None
    le = None

    # NO st.connection("my_data") HERE
    gc, drive = get_gspread_and_drive_clients()
    if not drive:
        st.sidebar.error("Google Drive client not available. Cannot load AI model.")
        return None, None

    download_model_path = MODEL_FILE
    download_encoder_path = ENCODER_FILE

    try:
        model_folder_id = st.secrets.google_drive.model_folder_id

        file_list = drive.ListFile({
            'q': f"'{model_folder_id}' in parents and title='{MODEL_FILE}' and trashed=false"
        }).GetList()
        if not file_list:
            st.sidebar.warning(f"AI Prediction Model '{MODEL_FILE}' not found on Google Drive in folder ID {model_folder_id}.")
            return None, None
        file = file_list[0]
        file.GetContentFile(download_model_path)

        file_list_encoder = drive.ListFile({
            'q': f"'{model_folder_id}' in parents and title='{ENCODER_FILE}' and trashed=false"
        }).GetList()
        if not file_list_encoder:
            st.sidebar.warning(f"Label Encoder '{ENCODER_FILE}' not found on Google Drive in folder ID {model_folder_id}.")
            if os.path.exists(download_model_path):
                os.remove(download_model_path)
            return None, None
        file_encoder = file_list_encoder[0]
        file_encoder.GetContentFile(download_encoder_path)

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
        if os.path.exists(download_model_path):
            os.remove(download_model_path)
        if os.path.exists(download_encoder_path):
            os.remove(download_encoder_path)

def load_all_historical_rounds_from_sheet():
    gc, _ = get_gspread_and_drive_clients() # Corrected
    if not gc:
        return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

    try:
        spreadsheet = gc.open("Casino Card Game Log")
        worksheet = spreadsheet.worksheet("Sheet1")
        data = worksheet.get_all_records()
        if data:
            df = pd.DataFrame(data)
            if 'Deck_ID' in df.columns:
                df['Deck_ID'] = pd.to_numeric(df['Deck_ID'], errors='coerce').fillna(1).astype(int)
            else:
                df['Deck_ID'] = 1
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

# --- AI Model Initialization (Call load_ai_model here, before session state or UI) ---
# This loads the model once when the app starts from Streamlit App Data
# This MUST be placed here, at the very top level of your script,
# before any st.session_state access or Streamlit UI elements are defined.
ai_model_initial_load, label_encoder_initial_load = load_ai_model()


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
        training_successful = train_and_save_prediction_model(all_historical_rounds)
        if training_successful:
            st.session_state.ai_model, st.session_state.label_encoder = load_ai_model()
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

    if st.session_state.ai_model and st.session_state.label_encoder and len(current_deck_outcomes) >= SEQUENCE_LENGTH:
        ai_model_prediction_attempted = True
        st.markdown("---")
        st.subheader("AI Model's Prediction")
        try:
            last_n_outcomes = current_deck_outcomes[-SEQUENCE_LENGTH:]

            # Ensure all outcomes in last_n_outcomes are known to the encoder
            known_outcomes = st.session_state.label_encoder.classes_
            if not all(outcome in known_outcomes for outcome in last_n_outcomes):
                st.warning("AI model cannot predict: Unknown outcomes in the recent sequence. Retrain model with more diverse data.")
                ai_model_prediction_error_occurred = True
            else:
                # Need to join the sequence for transformation as it was trained on joined strings
                encoded_last_n = st.session_state.label_encoder.transform(["_".join(last_n_outcomes)]).reshape(1, -1)

                predicted_encoded_outcome = st.session_state.ai_model.predict(encoded_last_n)[0]
                predicted_outcome_ai = st.session_state.label_encoder.inverse_transform([predicted_encoded_outcome])[0]

                probabilities = st.session_state.ai_model.predict_proba(encoded_last_n)[0]
                # Find the probability for the predicted outcome
                confidence_ai = probabilities[st.session_state.label_encoder.transform([predicted_outcome_ai])[0]] * 100

                st.markdown(f"🤖 **AI Model Prediction:** ➡️ **{predicted_outcome_ai}** (Confidence: {confidence_ai:.1f}%)")
                st.caption(f"Based on the last {SEQUENCE_LENGTH} outcomes: {', '.join(last_n_outcomes)}")

                prob_df = pd.DataFrame({
                    'Outcome': st.session_state.label_encoder.classes_,
                    'Probability': probabilities
                }).sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"AI Model prediction error: {e}. Ensure model is trained and data is consistent.")
            st.caption("Try retraining the model if this persists.")
            ai_model_prediction_error_occurred = True
    elif st.session_state.ai_model and st.session_state.label_encoder and len(current_deck_outcomes) < SEQUENCE_LENGTH:
        st.warning(f"AI model needs at least {SEQUENCE_LENGTH} recent outcomes to predict. Play more rounds!")
    else:
        st.info("AI Model is not loaded or not enough data for AI prediction. Train the model and play more rounds.")

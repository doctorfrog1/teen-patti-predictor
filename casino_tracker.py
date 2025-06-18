import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os # Keep this for local file operations like model saving
import joblib
import traceback # Import traceback for detailed error logging
import numpy as np # NEW: Added for robust NaN handling and Deck_ID generation
from sklearn.ensemble import RandomForestClassifier

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
PLAYER_A_FIXED_CARDS_STR = {'J♣', '10♠', '8♠'}
PREDICTION_ROUNDS_CONSIDERED = 1 # Number of previous rounds to consider for AI sequence prediction
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

# --- Existing Helper Functions (Example: save_rounds function, etc.) ---
# ... (your existing helper functions like `save_rounds`, `load_rounds`, etc.) ...


# NEW HELPER FUNCTION: Extracts base features common for training and prediction
def _get_base_features(df_input):
    """
    Extracts common base features like Hour_Of_Day, Day_Of_Week, Deck_Round_Number.
    Ensures Timestamp is datetime and Deck_ID is string.
    """
    df = df_input.copy()

    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Extract time-based features
    df['Hour_Of_Day'] = df['Timestamp'].dt.hour
    df['Day_Of_Week'] = df['Timestamp'].dt.dayofweek # Monday=0, Sunday=6

    # Calculate Deck_Round_Number (how many rounds into the current deck)
    # This assumes df is already sorted by Timestamp for correct cumulative count
    df['Deck_Round_Number'] = df.groupby('Deck_ID').cumcount() + 1

    # Ensure Deck_ID is treated as a string for one-hot encoding
    df['Deck_ID'] = df['Deck_ID'].astype(str)

    return df

# NEW HELPER FUNCTION: Prepares features for AI model prediction
def prepare_prediction_features(last_n_outcomes_str, current_deck_id, current_timestamp, current_deck_rounds_df, label_encoder, trained_model_feature_columns):
    """
    Prepares the feature DataFrame for prediction, matching the training features.

    Args:
        last_n_outcomes_str (list): List of the last PREDICTION_ROUNDS_CONSIDERED outcome strings
                                     (e.g., ['Over 21', 'Under 21', ...]). These are actual outcome strings.
        current_deck_id (int): The ID of the current deck.
        current_timestamp (datetime): The timestamp for the round being predicted (current time).
        current_deck_rounds_df (pd.DataFrame): DataFrame of all rounds in the current deck so far.
        label_encoder (LabelEncoder): The fitted LabelEncoder from training.
        trained_model_feature_columns (list): The list of column names the model was trained on.

    Returns:
        pd.DataFrame: A DataFrame with one row, containing features ready for prediction.
    """
    # Create a temporary DataFrame with a single row representing the 'next' round's context
    predict_data = pd.DataFrame([{
        'Timestamp': current_timestamp,
        'Deck_ID': current_deck_id,
        'Outcome': 'PLACEHOLDER' # This is a dummy for _get_base_features, won't be used for actual outcome
    }])

    # Apply the base feature extraction
    df_predict_processed = _get_base_features(predict_data)

    # Encode the last N outcomes using the provided label_encoder
    # Ensure they are in the correct order for lags (Lag1 is most recent)
    encoded_lags = [label_encoder.transform([outcome])[0] for outcome in last_n_outcomes_str[::-1]]

    # Create lagged feature columns (Outcome_Lag1, Outcome_Lag2, etc.)
    for i in range(1, PREDICTION_ROUNDS_CONSIDERED + 1):
        if i <= len(encoded_lags):
            df_predict_processed[f'Outcome_Lag{i}'] = str(encoded_lags[i-1]) # Convert encoded int to string for get_dummies
        else:
            # This case should ideally not happen if len(last_n_outcomes_str) matches PREDICTION_ROUNDS_CONSIDERED.
            # If it could, you'd need a strategy like a 'missing' category, but for now we expect full lags.
            df_predict_processed[f'Outcome_Lag{i}'] = 'MISSING_LAG' # Fallback for debugging if it happens

    # Calculate Deck_Round_Number for the *next* round
    df_predict_processed['Deck_Round_Number'] = len(current_deck_rounds_df) + 1

    # Define categorical features (must match `train_ai_model` exactly)
    categorical_features_for_dummies = [f'Outcome_Lag{i}' for i in range(1, PREDICTION_ROUNDS_CONSIDERED + 1)] + \
                                     ['Hour_Of_Day', 'Day_Of_Week', 'Deck_ID']

    # Convert relevant columns to string type for `pd.get_dummies`
    for col in categorical_features_for_dummies:
        if col in df_predict_processed.columns:
            df_predict_processed[col] = df_predict_processed[col].astype(str)

    # Apply one-hot encoding to the single prediction row
    X_categorical_encoded_predict = pd.get_dummies(df_predict_processed[categorical_features_for_dummies],
                                                   columns=categorical_features_for_dummies,
                                                   drop_first=False)

    # Select numerical features (must match `train_ai_model` exactly)
    numerical_features = ['Deck_Round_Number']
    X_numerical_predict = df_predict_processed[numerical_features]

    # Combine all features for the single row
    X_predict_raw = pd.concat([X_categorical_encoded_predict, X_numerical_predict], axis=1)

    # Ensure the prediction DataFrame has ALL the columns that the model was trained on.
    # Fill any missing columns (e.g., if a specific 'Hour_Of_Day_10' column existed in training but not in this single prediction row) with 0.
    X_predict_final = X_predict_raw.reindex(columns=trained_model_feature_columns, fill_value=0)

    # Ensure column order matches the training order
    X_predict_final = X_predict_final[trained_model_feature_columns]

    return X_predict_final # Return as a DataFrame with one row

@st.cache_resource(ttl="1h")
def train_ai_model(df_all_rounds):
    st.sidebar.info("Attempting to train AI model...")

    # Check for sufficient data
    if df_all_rounds.empty or len(df_all_rounds['Outcome'].unique()) < 2:
        st.sidebar.warning("Insufficient historical data to train the AI model. Need at least two different outcomes.")
        return None, None

    # Sort data to ensure correct lag and round number calculation
    df_all_rounds_sorted = df_all_rounds.sort_values(by=['Deck_ID', 'Timestamp']).reset_index(drop=True)

    # --- Start Feature Engineering ---
    # Use the new helper function to get base time and deck features
    df_processed = _get_base_features(df_all_rounds_sorted)

    # Fit label encoder on all possible outcomes to ensure consistency for prediction
    # This helps even if a specific outcome hasn't appeared in the training data yet.
    label_encoder = LabelEncoder()
    all_possible_outcomes = ['Over 21', 'Under 21', 'Exactly 21']
    label_encoder.fit(all_possible_outcomes)
    df_processed['Outcome_Encoded'] = label_encoder.transform(df_processed['Outcome'])

    # Create lagged features for outcomes (using the encoded outcomes)
    for i in range(1, PREDICTION_ROUNDS_CONSIDERED + 1):
        # Shift based on deck to ensure lags don't cross deck boundaries
        df_processed[f'Outcome_Lag{i}'] = df_processed.groupby('Deck_ID')['Outcome_Encoded'].shift(i)

    # Filter out rows with NaN created by shifting (these are at the start of decks)
    df_filtered = df_processed.dropna(subset=[f'Outcome_Lag{PREDICTION_ROUNDS_CONSIDERED}']).copy()

    # Align the target variable (y) with the filtered features (X)
    y_aligned = df_filtered['Outcome_Encoded']

    # Define categorical features that need one-hot encoding
    # Outcome_LagX are encoded numbers, treat them as categories for get_dummies
    categorical_features_for_dummies = [f'Outcome_Lag{i}' for i in range(1, PREDICTION_ROUNDS_CONSIDERED + 1)] + \
                                     ['Hour_Of_Day', 'Day_Of_Week', 'Deck_ID']

    # Convert values in these columns to string type for `pd.get_dummies`
    for col in categorical_features_for_dummies:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].astype(str)

    # Apply one-hot encoding to the defined categorical features
    X_categorical_encoded = pd.get_dummies(df_filtered[categorical_features_for_dummies],
                                           columns=categorical_features_for_dummies,
                                           drop_first=False) # Keep all dummies for consistent columns

    # Define numerical features (Deck_Round_Number will be directly used as a number)
    numerical_features = ['Deck_Round_Number']
    X_numerical = df_filtered[numerical_features]

    # Combine all features (one-hot encoded and numerical)
    X_combined = pd.concat([X_categorical_encoded, X_numerical], axis=1)

    # Fill any remaining NaNs in features (should be rare if dropna was effective, but good practice)
    X_combined = X_combined.fillna(0)
    # --- End Feature Engineering ---

    # Train the Logistic Regression model
    model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    try:
        model.fit(X_combined, y_aligned)

        # Store the exact feature column names the model was trained on
        # This is CRUCIAL for preparing prediction input correctly later
        model.feature_columns_ = X_combined.columns.tolist()

        st.sidebar.success("AI Prediction Model Trained Successfully!")
        return model, label_encoder
    except ValueError as e:
        st.sidebar.error(f"Error during model training: {e}. This often means your data is not diverse enough, or there's a problem with feature scaling or sparse data. Ensure enough data with all outcome types.")
        return None, None
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during AI model training: {e}")
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

# --- REVISED AI MODEL INITIALIZATION BLOCK ---
# This combined block handles initializing session state, attempting to load from Drive,
# and if necessary, triggering a re-train.

# Always ensure these are defined in session_state, as they're checked repeatedly.
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# Flag to determine if we need to try training
model_needs_training = False

# Check if model is not loaded OR if it's an "old" model lacking feature_columns_
if st.session_state.ai_model is None or not hasattr(st.session_state.ai_model, 'feature_columns_'):
    st.sidebar.info("AI Model not found or needs update. Attempting to load from Google Drive...")
    
    # Attempt to load from Drive first
    loaded_model, loaded_encoder = load_ai_model_from_drive()
    
    if loaded_model is not None and loaded_encoder is not None and hasattr(loaded_model, 'feature_columns_'):
        st.session_state.ai_model = loaded_model
        st.session_state.label_encoder = loaded_encoder
        st.sidebar.success("AI Model loaded from Drive and ready!")
    else:
        # If loading failed or it was an old model, flag for training
        st.sidebar.warning("Model not found on Drive or outdated. Will attempt to train.")
        model_needs_training = True

if model_needs_training:
    # Fetch all historical data needed for training.
    all_rounds_df_for_initial_training = load_all_historical_rounds_from_sheet()

    if not all_rounds_df_for_initial_training.empty:
        st.info("AI Model not found or needs training. Attempting to train from historical data...")
        with st.spinner("Training AI model... This might take a moment."):
            # `train_and_save_prediction_model()` should use your *modified* `train_ai_model` internally.
            # It also handles saving locally and uploading to Google Drive.
            training_successful = train_and_save_prediction_model() # This function should now use your updated train_ai_model
            
            if training_successful:
                # After successful training and upload, re-load the model into session state
                # to ensure the app uses the newly trained model immediately.
                # This will *also* show a success message from `load_ai_model_from_drive` if it works.
                st.session_state.ai_model, st.session_state.label_encoder = load_ai_model_from_drive()
                if st.session_state.ai_model and hasattr(st.session_state.ai_model, 'feature_columns_'):
                    st.success("AI Model trained from historical data and ready for use!")
                else:
                    st.warning("AI Model trained but failed to load into session state (or missing features). Check Drive permissions.")
            else:
                st.warning("AI Model could not be trained initially. Please ensure sufficient and diverse historical data (at least 2 different outcomes) in your Google Sheet.")
    else:
        st.info("No historical data available to train the AI model on app startup. Please add some rounds!")

# --- END OF REVISED AI MODEL INITIALIZATION BLOCK ---


# --- Session State Initialization (Keep these, they are standard Streamlit practice) ---
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

# Note: The AI model and label encoder are now handled by the block above,
# so you don't need the redundant `if 'ai_model' not in st.session_state:` checks here.
# The previous block ensures they are set in session state.

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
       st.success("Round saved successfully!") # Optional: confirmation message
      
       # --- START OF CONTINUOUS LEARNING LOGIC (UPDATED) ---
       st.info("New round added. Triggering AI model re-training...")
       with st.spinner("Re-training AI model with latest data..."):
           training_successful = train_and_save_prediction_model() # This function handles fetching, training, and saving
           if training_successful:
               # Reload the model into session state after successful training/saving to Drive
               st.session_state.ai_model, st.session_state.label_encoder = load_ai_model_from_drive()
               st.success("AI Model re-trained and updated with the latest data!")
           else:
               st.warning("AI Model could not be re-trained with the latest data. See logs/messages above.")
       # --- END OF CONTINUOUS LEARNING LOGIC ---

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

   if st.session_state.ai_model and st.session_state.label_encoder and hasattr(st.session_state.ai_model, 'feature_columns_'):
       current_deck_rounds_df = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id].copy()
       current_deck_outcomes = current_deck_rounds_df['Outcome'].tolist()

       if len(current_deck_outcomes) < PREDICTION_ROUNDS_CONSIDERED:
           st.info(f"AI Model needs at least {PREDICTION_ROUNDS_CONSIDERED} past outcomes in the current deck to make a prediction (based on historical sequence, time, and deck features). Only {len(current_deck_outcomes)} available.")
           ai_model_prediction_attempted = False
       else:
           ai_model_prediction_attempted = True
           st.markdown("---")
           st.subheader("🤖 AI Model's Prediction for the *Next Round*")

           try:
               # Get the last PREDICTION_ROUNDS_CONSIDERED outcomes from the current deck for lagged features
               recent_outcomes_for_lags = current_deck_outcomes[-PREDICTION_ROUNDS_CONSIDERED:]

               # Pass all necessary context to the feature preparation function
               X_predict = prepare_prediction_features(
                   last_n_outcomes_str=recent_outcomes_for_lags,
                   current_deck_id=st.session_state.current_deck_id,
                   current_timestamp=datetime.now(), # Use current time for prediction context
                   current_deck_rounds_df=current_deck_rounds_df, # Pass current deck rounds for round number calc
                   label_encoder=st.session_state.label_encoder,
                   trained_model_feature_columns=st.session_state.ai_model.feature_columns_
               )
               # --- DEBUGGING AID (You can remove these print statements after successful testing) ---
               # print(f"DEBUG (PREDICTION): Shape of X_predict (features for prediction): {X_predict.shape}")
               # print(f"DEBUG (PREDICTION): Columns of X_predict: {X_predict.columns.tolist()}")
               # print(f"DEBUG (PREDICTION): X_predict data: \n{X_predict.head()}")
               # print(f"DEBUG (TRAINING): Stored model features: {st.session_state.ai_model.feature_columns_}")
               # --- END DEBUGGING AID ---


               predicted_encoded_outcome = st.session_state.ai_model.predict(X_predict)
               predicted_outcome_ai = st.session_state.label_encoder.inverse_transform(predicted_encoded_outcome)[0]

               proba_output = st.session_state.ai_model.predict_proba(X_predict)
               probabilities = proba_output[0]

               model_string_classes = st.session_state.label_encoder.inverse_transform(st.session_state.ai_model.classes_)

               # Ensure the predicted outcome is one of the known classes
               if predicted_outcome_ai in model_string_classes:
                   model_internal_index = list(model_string_classes).index(predicted_outcome_ai)
                   confidence_ai = probabilities[model_internal_index] * 100
               else:
                   confidence_ai = 0.0
                   st.warning(f"AI Model predicted an unknown outcome '{predicted_outcome_ai}'. Setting confidence to 0%.")

               st.markdown(f"➡️ **{predicted_outcome_ai}** (Confidence: {confidence_ai:.1f}%)")
               st.caption(f"Based on the last {PREDICTION_ROUNDS_CONSIDERED} outcomes, **time of day, day of week, and current deck progress.**")
               prob_df = pd.DataFrame({
                   'Outcome': model_string_classes,
                   'Probability': probabilities
               }).sort_values(by='Probability', ascending=False)
               st.dataframe(prob_df, hide_index=True, use_container_width=True)

           except ValueError as e:
               st.error(f"AI Model prediction error: {e}. This often means your historical data for the current deck is insufficient or inconsistent for the chosen PREDICTION_ROUNDS_CONSIDERED.")
               ai_model_prediction_error_occurred = True
           except Exception as e:
               st.error(f"An unexpected error occurred during AI model prediction: {e}")
               ai_model_prediction_error_occurred = True
   else:
       st.info("AI Model is not loaded or not fully trained (requires feature_columns_). Please ensure sufficient data and initial training.")

   # --- End AI Prediction Section ---

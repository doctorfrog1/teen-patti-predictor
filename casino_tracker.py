import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os

# --- NEW IMPORTS FOR AI ---
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib # For saving/loading the model

# --- Configuration ---
PLAYER_A_FIXED_CARDS_STR = {'J♣', '10♠', '9♠'} # Player A's fixed cards (assuming they are always out of play)
PREDICTION_ROUNDS_CONSIDERED = 10 # Number of previous rounds to consider for simple prediction
STREAK_THRESHOLD = 3 # Minimum streak length to highlight
OVER_UNDER_BIAS_THRESHOLD = 0.6 # If Over/Under > 60% of rounds, show bias

# --- AI Configuration ---
SEQUENCE_LENGTH = 3 # You can adjust this based on how many past outcomes you think matter
MODEL_FILE = "prediction_model.joblib"
ENCODER_FILE = "label_encoder.joblib"


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
def get_gspread_client():
    try:
        gc = gspread.service_account_from_dict(st.secrets.gcp_service_account)
        return gc
    except Exception as e:
        st.error(f"Error loading Google Sheets credentials: {e}. Please ensure st.secrets are configured.")
        st.stop() # Stop the app if credentials are not loaded
        return None

# --- NEW: AI Model Training Function (Modified for Daily Learning & Streamlit App Data) ---
def train_and_save_prediction_model(all_rounds_df, sequence_length=SEQUENCE_LENGTH):
    """
    Trains a Logistic Regression model on recent (today's) outcomes and saves it using Streamlit App Data.
    """
    conn = st.connection("my_data")

    # Filter data for today's outcomes
    today = datetime.now().date()
    if 'Timestamp' in all_rounds_df.columns:
        all_rounds_df['Timestamp'] = pd.to_datetime(all_rounds_df['Timestamp'])
        daily_rounds_df = all_rounds_df[all_rounds_df['Timestamp'].dt.date == today].copy()
    else:
        daily_rounds_df = pd.DataFrame() # No timestamp, cannot filter

    st.info(f"Preparing data for AI model training from {len(daily_rounds_df)} rounds played today ({today})...")
    if daily_rounds_df.empty or len(daily_rounds_df) < sequence_length + 1:
        st.warning(f"Not enough recent rounds data to train the AI model. Need at least {sequence_length + 1} rounds from today.")
        try:
            if conn.exists(MODEL_FILE): conn.delete(MODEL_FILE)
            if conn.exists(ENCODER_FILE): conn.delete(ENCODER_FILE)
            st.info("Cleared old AI model files due to insufficient data.")
        except Exception:
            pass # Ignore if deletion fails or files don't exist
        return False # Indicate training failed

    # 1. Prepare data: Create sequences (features) and target outcomes (labels)
    features = []
    labels = []
    outcomes = daily_rounds_df['Outcome'].tolist()

    for i in range(len(outcomes) - sequence_length):
        features.append(outcomes[i : i + sequence_length])
        labels.append(outcomes[i + sequence_length])

    if not features:
        st.warning(f"Not enough sequences generated from today's data ({len(outcomes)} outcomes, sequence length {sequence_length}).")
        return False

    # 2. Encoding categorical data
    le = LabelEncoder()
    all_possible_outcomes = list(set(outcomes + ['Over 21', 'Under 21', 'Exactly 21']))
    le.fit(all_possible_outcomes)

    encoded_features = []
    for seq in features:
        encoded_features.append(le.transform(seq))
    encoded_labels = le.transform(labels)

    X = pd.DataFrame(encoded_features)
    y = encoded_labels

    st.info(f"Training AI model with {len(X)} samples from today's data...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    try:
        model.fit(X, y)

        # Save the trained model and encoder using Streamlit App Data connection
        with conn.open(MODEL_FILE, "wb") as f:
            joblib.dump(model, f)
        with conn.open(ENCODER_FILE, "wb") as f:
            joblib.dump(le, f)

        st.success("AI prediction model trained and saved successfully to Streamlit App Data!")
        return True
    except Exception as e:
        st.error(f"Error during AI model training or saving: {e}")
        return False

# --- NEW: AI Model Loading Function (Modified for Streamlit App Data) ---
@st.cache_resource
def load_ai_model():
    st.write("--- Debugging st.secrets ---")
    st.write(f"st.secrets content: {st.secrets}")

    if "connections" in st.secrets:
        st.write(f"st.secrets['connections'] content: {st.secrets['connections']}")
        if "my_data" in st.secrets["connections"]:
            st.write(f"st.secrets['connections']['my_data'] content: {st.secrets['connections']['my_data']}")
            if "type" in st.secrets["connections"]["my_data"]:
                st.write("Found 'type' key in my_data connection!")
            else:
                st.write("ERROR: 'type' key NOT found in my_data connection despite being in secrets.toml!")
        else:
            st.write("ERROR: 'my_data' connection NOT found in st.secrets['connections']!")
    else:
        st.write("ERROR: 'connections' section NOT found in st.secrets!")

    st.write("--- End Debugging st.secrets ---")
    conn = st.connection("my_data")

    model = None
    le = None

    try:
        if conn.exists(MODEL_FILE) and conn.exists(ENCODER_FILE):
            with conn.open(MODEL_FILE, "rb") as f:
                model = joblib.load(f)
            with conn.open(ENCODER_FILE, "rb") as f:
                le = joblib.load(f)
            st.sidebar.success("AI Prediction Model Loaded.")
        else:
            st.sidebar.warning("AI Prediction Model files not found. Please train the model first.")
    except Exception as e:
        st.sidebar.error(f"Error loading AI model from Streamlit App Data: {e}")
    return model, le

# --- NEW: Function to load all rounds from sheet for training ---
def load_all_historical_rounds_from_sheet():
    gc = get_gspread_client()
    if not gc:
        return pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID']) # Return empty DF

    try:
        spreadsheet = gc.open("Casino Card Game Log")
        worksheet = spreadsheet.worksheet("Sheet1")
        data = worksheet.get_all_records()
        if data:
            df = pd.DataFrame(data)
            # Ensure Deck_ID is numeric
            if 'Deck_ID' in df.columns:
                df['Deck_ID'] = pd.to_numeric(df['Deck_ID'], errors='coerce').fillna(1).astype(int) # Default to 1 if NaN after coerce
            else:
                df['Deck_ID'] = 1 # Add Deck_ID if missing
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
    gc = get_gspread_client()
    if not gc:
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
        st.session_state.played_cards = set(PLAYER_A_FIXED_CARDS_STR) # Only Player A's cards initially if data load fails.
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

    # --- Crucial REVISED played_cards Initialization Logic ---
    st.session_state.played_cards = set() # Always start fresh for the current deck's played cards here

    # Add cards played in the current deck from the history
    if not st.session_state.rounds.empty:
        current_deck_rounds = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]
        for _, row in current_deck_rounds.iterrows():
            st.session_state.played_cards.add(row['Card1'])
            st.session_state.played_cards.add(row['Card2'])
            st.session_state.played_cards.add(row['Card3'])

    # Always add Player A's fixed cards (they are never available in any deck)
    for card in PLAYER_A_FIXED_CARDS_STR:
        st.session_state.played_cards.add(card)


def save_rounds():
    gc = get_gspread_client()
    if not gc:
        st.warning("Cannot save rounds: Google Sheets client not available.")
        return

    try:
        spreadsheet = gc.open("Casino Card Game Log") # <--- ENSURE THIS MATCHES YOUR SHEET NAME
        worksheet = spreadsheet.worksheet("Sheet1") # <--- ENSURE THIS MATCHES YOUR SHEET TAB NAME

        data_to_write = [st.session_state.rounds.columns.tolist()] + st.session_state.rounds.astype(str).values.tolist()

        worksheet.clear()
        worksheet.update('A1', data_to_write)

    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Cannot save: Google Sheet 'Casino Card Game Log' not found. Please create the sheet and share it correctly.")
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

    # Player A's fixed cards are re-added immediately for the new deck
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

# Initialize current_deck_id only if it's not already in session state
if 'current_deck_id' not in st.session_state:
    temp_gc = get_gspread_client()
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

# --- NEW: Ensure AI model and encoder are in session state for later updates ---
# These will be updated by the 'Train AI Model' button click
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = ai_model_initial_load
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = label_encoder_initial_load

if 'historical_patterns' not in st.session_state: # Ensure this is initialized if not already
    st.session_state.historical_patterns = pd.DataFrame(columns=['Timestamp', 'Deck_ID', 'Pattern_Name', 'Pattern_Sequence', 'Start_Round_ID', 'End_Round_ID'])


# --- Load data on app startup ---
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

    # --- NEW: AI Model Prediction Attempt ---
    ai_model_prediction_attempted = False
    ai_model_prediction_error_occurred = False # Flag to manage fallback logic

    if st.session_state.ai_model and st.session_state.label_encoder and len(current_deck_outcomes) >= SEQUENCE_LENGTH:
        ai_model_prediction_attempted = True
        st.markdown("---") # Separator for AI Prediction
        st.subheader("AI Model's Prediction")
        try:
            last_n_outcomes = current_deck_outcomes[-SEQUENCE_LENGTH:]

            known_outcomes = st.session_state.label_encoder.classes_
            if not all(outcome in known_outcomes for outcome in last_n_outcomes):
                st.warning("AI model cannot predict: Unknown outcomes in the recent sequence. Retrain model with more diverse data.")
                ai_model_prediction_error_occurred = True
            else:
                encoded_last_n = st.session_state.label_encoder.transform(last_n_outcomes).reshape(1, -1)

                predicted_encoded_outcome = st.session_state.ai_model.predict(encoded_last_n)[0]
                predicted_outcome_ai = st.session_state.label_encoder.inverse_transform([predicted_encoded_outcome])[0]

                probabilities = st.session_state.ai_model.predict_proba(encoded_last_n)[0]
                confidence_ai = probabilities[predicted_encoded_outcome] * 100

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
        ai_model_prediction_attempted = True
    elif not st.session_state.ai_model:
        st.warning("AI Prediction Model not loaded. Please train it using the 'Train AI Model' button in the sidebar.")
        ai_model_prediction_attempted = True


    # --- Existing Fallback to Simple Frequency-Based Prediction (Adjusted) ---
    # Only show this if no pattern prediction AND no successful AI prediction was made.
    # It will show if AI was not ready, or had an error, or insufficient rounds.
    if not predicted_by_pattern and (not ai_model_prediction_attempted or ai_model_prediction_error_occurred):
         recent_rounds = st.session_state.rounds[
            st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id
         ].tail(PREDICTION_ROUNDS_CONSIDERED)

         if not recent_rounds.empty:
            outcome_counts = recent_rounds['Outcome'].value_counts()

            if 'Exactly 21' in outcome_counts.index:
                outcome_counts = outcome_counts.drop(labels='Exactly 21', errors='ignore')

            if not outcome_counts.empty:
                predicted_outcome = outcome_counts.index[0]
                st.write(f"Based on last {len(recent_rounds)} rounds (current deck):")
                st.markdown(f"**Prediction:** ➡️ **{predicted_outcome}**")
            else:
                st.write("Not enough 'Over 21' or 'Under 21' outcomes in recent rounds for a simple prediction.")
         else:
             st.write("Not enough rounds played in current deck for a prediction.")
elif st.session_state.rounds.empty: # Original check for no rounds at all
    st.write("No historical rounds available for prediction.")


### Full Round History

st.header("Round History (Current Deck)")
if not st.session_state.rounds.empty:
    current_deck_history = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]
    if not current_deck_history.empty:
        st.dataframe(current_deck_history.set_index('Round_ID'))
    else:
        st.write("No rounds played in the current deck yet.")
else:
    st.write("No rounds recorded yet.")

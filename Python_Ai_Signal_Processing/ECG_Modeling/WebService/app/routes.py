from scipy.signal import find_peaks
from flask import abort, flash, redirect
import matplotlib.pyplot as plt
from flask import Blueprint, logging, request, jsonify, render_template, redirect, send_from_directory, url_for, flash, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import io
import base64
from pymongo import MongoClient
from functools import wraps
from dotenv import load_dotenv
from bson.objectid import ObjectId
import traceback
import logging
import traceback
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure, ConfigurationError
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


# Suppress GPU warnings if running on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load environment variables
load_dotenv()

db_username = os.getenv("DB_USERNAME")
db_password = os.getenv("DB_PASSWORD")
db_cluster = os.getenv("DB_CLUSTER")
db_name = os.getenv("DB_NAME")
print("ENV VARIABLES", db_username, db_password, db_cluster, db_name)

# MongoDB setup
# MongoDB connection setup
# Initialize MongoDB variables
client, db, users_collection = None, None, None


def connect_to_mongo():
    """
    Connect to MongoDB and return the client, database, and users collection.
    """
    global client, db, users_collection  # Ensure these variables are available globally

    try:
        # Build the connection string dynamically
        mongo_uri = (
            f"mongodb+srv://{db_username}:{db_password}@final.yzzh9ig.mongodb.net/?retryWrites=true&w=majority&appName=Final")

        # Establish connection with a 5-second timeout
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        users_collection = db["Users"]
        print("Connected to MongoDB successfully.")
        print(users_collection)

        # Test the connection by running a simple command
        client.admin.command('ping')
        logging.info("Connected to MongoDB successfully.")
        return client, db, users_collection

    except ServerSelectionTimeoutError:
        logging.error(
            "Failed to connect to MongoDB: Server selection timeout.", exc_info=True)
    except OperationFailure:
        logging.error(
            "Failed to connect to MongoDB: Operation failure.", exc_info=True)
    except ConfigurationError:
        logging.error(
            "Failed to connect to MongoDB: Configuration error.", exc_info=True)
    except Exception as e:
        logging.error(
            f"Unexpected error connecting to MongoDB: {e}", exc_info=True)

    # Return None on failure
    client, db, users_collection = None, None, None
    return None, None, None

# Initialize the connection at the start


# Load the pre-trained model
try:
    model = load_model("./Model/noisy_signal_classifier.h5",
                       custom_objects={"Input": lambda *args, **kwargs: None})
except Exception as e:
    print("Error loading model:", e)
    traceback.print_exc()

# Blueprint setup
main = Blueprint('main', __name__)

# Login required decorator


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash("Please log in to access this page.", "error")
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated_function

# Enforce redirect to /login if not authenticated and check database connectivity


@main.before_app_request
def check_app_state():
    global client, db, users_collection

    # If the request is for the error page or static files, skip further checks
    if request.endpoint in ['main.error', 'static']:
        return

    # Check MongoDB connection
    if users_collection is None:
        logging.warning("MongoDB connection unavailable. Reinitializing...")
        client, db, users_collection = connect_to_mongo()
        if users_collection is None:
            flash("Database connection failed. Please try again later.", "error")
            # Redirect to the error page only once
            if request.endpoint != 'main.error':
                return redirect(url_for('main.error'))
            return  # Avoid further redirection in the error handler

    # Check user login state
    if not session.get('logged_in') and request.endpoint not in ['main.login', 'main.register', 'static']:
        flash("Please log in to access this page.", "error")
        return redirect(url_for('main.login'))


@main.route('/error')
def error():
    return render_template(
        'error.html',
        message="Database connection is currently unavailable. Please try again later."
    )

# Routes


@main.route('/')
@login_required
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering index page: {e}", exc_info=True)
        flash("An unexpected error occurred while loading the page.", "error")
        return redirect(url_for('main.error'))


@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            user = users_collection.find_one(
                {'username': username, 'password': password})
        except Exception as e:
            print("Error querying database:", e)
            flash("An error occurred while logging in. Please try again later.", "error")
            return redirect(url_for('main.login'))

        if user:
            session['logged_in'] = True
            session['username'] = username
            session['is_admin'] = user.get('isAdmin', False)
            flash("Login successful", "info")
            return redirect(url_for('main.index'))
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for('main.login'))

    return render_template('login.html')


@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        is_admin = request.form.get('is_admin') == 'on'

        try:
            if users_collection.find_one({'username': username}):
                flash("User already exists", "error")
                return redirect(url_for('main.register'))

            users_collection.insert_one({
                'username': username,
                'password': password,
                'isAdmin': is_admin
            })
            flash("Account created successfully", "info")
            return redirect(url_for('main.index'))
        except Exception as e:
            print("Error inserting into database:", e)
            flash("An error occurred while registering. Please try again later.", "error")
            return redirect(url_for('main.register'))

    return render_template('register.html')


@main.route('/logout')
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('main.login'))


@main.route('/static/docs/<filename>')
def serve_pdf(filename):
    # Resolve the absolute path to the `static/docs` directory
    docs_path = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), '../static/docs')

    # Debugging: Log paths for troubleshooting
    print(f"Resolved docs_path: {docs_path}")
    print(f"Filename requested: {filename}")

    # Sanitize the file name to prevent directory traversal attacks
    if '..' in filename or filename.startswith('/'):
        print("Invalid file path detected:", filename)
        return "Invalid file path", 400

    # Check if the file exists
    full_path = os.path.join(docs_path, filename)
    if not os.path.isfile(full_path):
        print("File not found at:", full_path)
        return abort(404)  # Flask's built-in 404 error

    # Serve the file
    return send_from_directory(docs_path, filename)


@main.route('/information')
def report():
    # Use Flask's `url_for` to dynamically generate the PDF URL
    pdf_url = '/static/docs/AI-EMG-FilteringSpasticity-TFG-JMF.pdf'
    return render_template('report.html', pdf_url=pdf_url)


@main.route('/detect', methods=['POST'])
@login_required
def detect_noise():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            flash('No file uploaded. Please upload a CSV file.', 'error')
            return redirect(url_for('main.index'))

        file = request.files['file']

        # Validate file type
        if not file.filename.endswith('.csv'):
            flash('Invalid file type. Please upload a CSV file.', 'error')
            return redirect(url_for('main.index'))

        # Read CSV file
        try:
            data = pd.read_csv(file)
        except pd.errors.ParserError:
            flash('The uploaded file is not a valid CSV format.', 'error')
            return redirect(url_for('main.index'))

        # Check for the required column
        if 'Original Signal' not in data.columns:
            flash("'Original Signal' column not found in the uploaded file.", 'error')
            return redirect(url_for('main.index'))

        # Normalize the signal data
        signal_data = data['Original Signal'].values
        scaler = MinMaxScaler()
        normalized_signal = scaler.fit_transform(signal_data.reshape(-1, 1)).flatten()

        # Predict noise using the loaded model
        predictions = model.predict(normalized_signal.reshape(-1, 1))
        classifications = (predictions > 0.5).astype(int).flatten()

        # Calculate the number of spastic signals
        num_spastic_signals = np.sum(classifications == 0)  # Spastic signals are classified as `0`

        # Identify absolute maxima and minima for extrema visualization
        peaks, _ = find_peaks(normalized_signal)
        troughs, _ = find_peaks(-normalized_signal)
        abs_maxima_indices = [i for i in peaks if normalized_signal[i] > 0.65]
        abs_minima_indices = [i for i in troughs if normalized_signal[i] < 0.2]
        abs_extrema_indices = sorted(abs_maxima_indices + abs_minima_indices)
        abs_extrema_values = normalized_signal[abs_extrema_indices]

        # Generate diagrams

        # 1. Signal Plot
        signal_plot = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.plot(data.index, normalized_signal, label='Normalized Signal', color='blue')
        plt.title('Normalized Signal Plot')
        plt.xlabel('Index')
        plt.ylabel('Normalized Signal Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(signal_plot, format='png')
        plt.close()
        signal_plot.seek(0)
        signal_plot_url = base64.b64encode(signal_plot.getvalue()).decode('utf8')

        # 2. Summary Plot
        summary = {
            "Not Spastic": int(classifications.sum()),
            "Spastic": int(len(classifications) - classifications.sum())
        }
        summary_plot = io.BytesIO()
        labels = list(summary.keys())
        values = list(summary.values())
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=['green', 'orange'])
        plt.title('Summary of Signal Classification')
        plt.xlabel('Classification')
        plt.ylabel('Count')
        plt.savefig(summary_plot, format='png')
        plt.close()
        summary_plot.seek(0)
        summary_plot_url = base64.b64encode(summary_plot.getvalue()).decode('utf8')

        # 3. Peak Connection Plot
        peak_connection_plot = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.plot(data.index[abs_extrema_indices], abs_extrema_values, marker='o',
                 linestyle='-', color='purple', label='Absolute Extrema Connected')
        plt.title('Connected Absolute Maxima and Minima')
        plt.xlabel('Index')
        plt.ylabel('Normalized Signal Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(peak_connection_plot, format='png')
        plt.close()
        peak_connection_plot.seek(0)
        peak_connection_plot_url = base64.b64encode(peak_connection_plot.getvalue()).decode('utf8')

        # 4. Spastic Regions Plot
        spastic_plot = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, normalized_signal, color='black', label='Normalized Signal')
        spastic_indices = np.where(classifications == 0)[0]
        non_spastic_indices = np.where(classifications == 1)[0]

        # Highlight spastic and non-spastic regions
        if len(spastic_indices) > 0:
            plt.scatter(spastic_indices, normalized_signal[spastic_indices],
                        color='red', label='Spastic Regions')
        if len(non_spastic_indices) > 0:
            plt.scatter(non_spastic_indices, normalized_signal[non_spastic_indices],
                        color='green', label='Non-Spastic Regions')

        plt.title('Spastic and Non-Spastic Regions Highlighted')
        plt.xlabel('Index')
        plt.ylabel('Normalized Signal Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(spastic_plot, format='png')
        plt.close()
        spastic_plot.seek(0)
        spastic_plot_url = base64.b64encode(spastic_plot.getvalue()).decode('utf8')

        # Render the results page with diagrams and spastic signal count
        return render_template(
            'results.html',
            signal_plot_url=signal_plot_url,
            summary_plot_url=summary_plot_url,
            peak_connection_plot_url=peak_connection_plot_url,
            spastic_plot_url=spastic_plot_url,
            num_spastic_signals=num_spastic_signals
        )

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        flash('An internal error occurred. Please try again later.', 'error')
        return redirect(url_for('main.index'))
    finally:
        plt.close('all')  # Close all Matplotlib figures


@main.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled Exception: {e}")
    traceback.print_exc()
    return jsonify({'error': 'Unhandled server error occurred.'}), 500

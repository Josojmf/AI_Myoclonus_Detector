import os
from flask import Flask
from app.routes import main  # Import the Blueprint
import matplotlib.pyplot as plt
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
# Global variable for MongoDB client
client = None

def cleanup(exception=None):
    """
    Cleanup resources when the app context is torn down.
    """
    global client
    if client:
        client.close()
        logging.info("Closed MongoDB connection.")
    plt.close('all')  # Close all Matplotlib figures

def create_app():
    # Dynamically resolve the base directory
    base_dir = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(base_dir, '../templates')
    static_dir = os.path.join(base_dir, '../static')

    # Log resolved paths for debugging
    logging.info(f"Base Directory: {base_dir}")
    logging.info(f"Templates Directory: {template_dir}")
    logging.info(f"Static Directory: {static_dir}")

    # Create the Flask app with dynamically resolved paths
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config['SECRET_KEY'] = 'supersecretkey'

    # Register the Blueprint
    app.register_blueprint(main, url_prefix="/")

    # Register the teardown function with the app
    app.teardown_appcontext(cleanup)

    return app

from flask import Flask
from app.routes import main  # Import the Blueprint
import matplotlib.pyplot as plt
import logging
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
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config['SECRET_KEY'] = 'supersecretkey'

    # Register the Blueprint
    app.register_blueprint(main, url_prefix="/")

    # Register the teardown function with the app
    app.teardown_appcontext(cleanup)

    return app

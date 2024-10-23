import os
from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_moment import Moment
from flask_migrate import Migrate
import logging
from logging.handlers import RotatingFileHandler
from flask_caching import Cache
from config import Config  # Import your Config class

# Initialize Flask-Caching
cache = Cache()

app = Flask(__name__, static_folder='static')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure the Flask app using the Config class
app.config.from_object(Config)
app.static_folder = 'static'

bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login = LoginManager(app)
login.login_view = 'login'
moment = Moment(app)
migrate = Migrate(app, db, render_as_batch=True)

# Configure logging
if not app.debug:
    if app.config['LOG_TO_STDOUT']:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)
    else:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler(
            app.config['LOG_FILENAME'],
            maxBytes=10240,
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(app.config['LOG_LEVEL'])
        app.logger.addHandler(file_handler)

    app.logger.setLevel(app.config['LOG_LEVEL'])
    app.logger.info('audrey')

# Initialize Flask-Caching with the app instance
cache.init_app(app)

from home import routes, errors, models, session_tracker
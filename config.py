import os
import logging
import tensorflow as tf
import numpy as np
import pickle

# Define basedir outside of the Config class
basedir = os.path.abspath(os.path.dirname(__file__))

# Path to the .env file
env_path = os.path.join(basedir, '.env')

# Open and read the .env file
with open(env_path) as f:
    for line in f:
        # Ignore lines starting with '#'
        if not line.startswith('#'):
            # Split each line by '=' to separate key and value
            key, value = line.strip().split('=', 1)
            # Set the environment variable
            os.environ[key] = value


# Now environment variables are loaded, you can access them as usual
class Config(object):
    # Form security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', '').replace(
        'postgres://', 'postgresql://') or \
                              'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Pagination
    HISTORY_PER_PAGE = int(os.environ.get('HISTORY_PER_PAGE') or 10)
    GEOCODING_API_KEY = os.environ.get('GEOCODING_API_KEY')
    GOOGLE_MAPS_API_KEY=os.environ.get('GOOGLE_MAPS_API_KEY')

    # Heroku logs
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')

    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'home/model')

    # MODEL = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'efficientnetv2s.h5'))
    # REC_MODEL = pickle.load(open(os.path.join(MODEL_DIR, 'RF.pkl'), 'rb'))
    # Upload folder and allowed extensions
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp' 'JPG'}

    # Logging setup
    LOG_FILENAME = 'audrey.log'
    LOG_LEVEL = logging.ERROR

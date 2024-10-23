from home import db, login, app
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, Numeric, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from datetime import datetime
from hashlib import md5
from sqlalchemy.exc import SQLAlchemyError
from flask import (
    jsonify, flash
)
from sqlalchemy import ForeignKey
import random
import string
import logging


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


@app.errorhandler(SQLAlchemyError)
def handle_database_error(e):
    # Log error
    app.logger.error(f'Database error: {str(e)}')
    # user-friendly response
    return jsonify({'error': f'A database error occurred {str(e)}'}), 500


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(64), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    about_me = db.Column(db.String(120))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='author', lazy='dynamic')
    def __repr__(self):
        return f'User: {self.username}'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return f'https://www.gravatar.com/avatar/{digest}?d=identicon&s={size}'

class Prediction(db.Model):  # Change db.model to db.Model
    __tablename__ = 'prediction'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    plant_class = db.Column(db.String(100))
    disease_status = db.Column(db.String(100))
    first_50_words = db.Column(db.String(200))
    disease_explanation = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), index=True)
    image_id = db.Column(db.Integer, db.ForeignKey('uploaded_images.id'), index=True)
    image = db.relationship('UploadedImage', backref='prediction')

class UploadedImage(db.Model):
    __tablename__ = 'uploaded_images'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    predictions = db.relationship('Prediction', backref='uploaded_image')

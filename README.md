# AgroHealth

AgroHealth is a web application that helps farmers to monitor their crops disease and pests It provides a platform for farmers to track the health and growth of their crops crops disease and pests, as well as access to information on best practices for farming and livestock care.

The system is built using Flask, SQLAlchemy, PostgreSQL, HTML/CSS, JavaScript, Bootstrap, OpenCV, TensorFlow, Keras, and Google Maps API. The application is designed to be user-friendly and easy to use, with a focus on providing accurate and reliable information to farmers. The object detection and tracking feature uses OpenCV and TensorFlow to identify and track objects in real-time is trained on coffee dataset of only 3 deficiencies. The Google Maps API is used to provide geolocation and mapping functionality to help users track the and get directions to the nearest agro-vet clinics. The application also includes a history feature that allows users to track their progress over time and view their history of crop disease and pests. The classification model is trained on a dataset of 27000 images of 4 classes crops: 'Cashew anthracnose', 'Cashew healthy', 'Cashew leaf miner', 'Cashew gumosis', 'Cashew red rust',  # Cashew'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic', # Cassava 'Maize fall armyworm', 'Maize grasshopper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight', 'Maize leaf spot', 'Maize streak virus',  # Maize 'Tomato healthy', 'Tomato leaf blight', 'Tomato leaf curl', 'Tomato septoria leaf spot', 'Tomato verticulium wilt'  # Tomato. The model is trained using the TensorFlow library and the Keras API. The model is trained using the Adam optimizer and the categorical cross-entropy loss function. The model is trained for 10 epochs and achieves an accuracy of 95%. The model is then saved as a .h5 file and loaded into the Flask application. The user can then use the model to predict the class of a new image by uploading the image to the application. Also The user can download disease treatment recommendations from the application.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Contributing](#contributing)

## Installation
1. Clone the repository:

```
git clone https://github.com/AlukweJonesTerah/agrohealth.git
```

2. Install the model and create .env:

## .env file has the following variables:

SECRET_KEY=
HISTORY_PER_PAGE=
GOOGLE_MAPS_API_KEY=
GEOCODING_API_KEY=


To get .env file, run the following command:

```
cp .env.example .env or send a request to get conetnts of .env file

https://docs.google.com/document/d/1NvE2N64o0Ykby1OPspBqgn_zL99vMO0esOWrY3unwEA/edit?usp=sharing

```
To get models file, for running the project:

```
https://drive.google.com/drive/folders/1PXnJ-1ZtH1BrfcZ5c0ag_oPpXGfWC-F9?usp=sharing
```

3. Install the dependencies:
```
cd agrohealth
pnpm install
```
```
pip inst
```
4. Set up the database:
```
python app.py db init
python app.py
python app.py db migrate -m "Initial migration."
python app.py db upgrade
```
5. Start the server:
```
export FLASK_APP=app.py  # for linux/macOS
set FLASK_APP=app.py  # for windows

```

```
python app.py runserver 

```
6. Open your web browser and navigate to http://localhost:5000 to access the application.


## Usage
Once the application is running, you can use it to track the health and growth of your crops, as well as access information on best practices for farming and livestock care.

## Contributing
Contributions are welcome! If you find a bug or have a feature request, please open an issue on the GitHub repository. If you would like to contribute code, please fork the repository and submit a pull request.  
## License
This project is licensed under the MIT License. See the LICENSE file for details.


## Technologies Used
- Python
- Flask
- SQLAlchemy
- PostgreSQL
- HTML/CSS
- JavaScript
- Bootstrap
- OpenCV
- TensorFlow
- Keras
- Google Maps API
- Google


## Features
- User authentication and authorization
- Crop disease and pest tracking, including disease identification and treatment recommendations
- Crop growth monitoring and prediction, 
- Geolocation and mapping, mapping of users location and nearest agro-vet clinics
- Object detection and tracking


## Database
The database is set up using SQLAlchemy and PostgreSQL. The database schema is defined in the app/models.py file.

python app.py db init
python app.py db migrate -m "Initial migration."
flask db init
flask db migrate -m "Create users table"

flask db upgrade

python app.py db upgrade



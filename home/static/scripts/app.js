{% extends "base.html" %}
{% block content %}
    <h1>Welcome to the Recommendation App</h1>
    <form id="uploadForm" class='upload_form' enctype="multipart/form-data" method="POST">
        {{ form.csrf_token }}
        {{ form.hidden_tag() }}
        {{ form.image(accept="image/*", id="fileInput", multiple="multiple") }}
        <p>Drag your files here or click in this area.</p>
        {{ form.submit() }}
        <button type="submit">Upload</button>
        <!-- Image element to display selected image -->
        <img id="preview-image" src="#" alt="Preview Image" style="display: none;">
        <span id="file-text">No file chosen</span>
        {{ form.image.errors }}
    </form>
    <div id="result"></div>
    <div id="prediction-result">{{ disease_status }}</div>
{% endblock %}

{% block scripts %}
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Access the form element and attach the submit event listener
            document.getElementById('uploadForm').addEventListener('submit', function(event) {
                event.preventDefault();

                var formData = new FormData();
                // Access the file input element
                var fileInput = document.getElementById('fileInput');
                // Check if a file is selected
                if (fileInput.files.length > 0) {
                    // Append the selected file to the FormData object
                    formData.append('image', fileInput.files[0]);

                    var xhr = new XMLHttpRequest();
                    xhr.open('POST', '/predict', true);
                    xhr.onload = function() {
                        if (xhr.status === 200) {
                            var response = JSON.parse(xhr.responseText);
                            // Process the prediction response here
                            document.getElementById('predictedPlantClass').innerText = response.predicted_plant_class;
                            document.getElementById('predictedDiseaseStatus').innerText = response.predicted_disease_status;
                        } else {
                            // Handle error response
                            console.error('Error:', xhr.responseText);
                        }
                    };
                    xhr.send(formData);
                } else {
                    console.error('No file selected');
                }
            });
        });
    </script>
{% endblock %}


<script>
        document.addEventListener('DOMContentLoaded', function() {
            // Access the form element and attach the submit event listener
            document.getElementById('uploadForm').addEventListener('submit', function(event) {
                event.preventDefault();

                var formData = new FormData();
                // Access the file input element
                var fileInput = document.getElementById('fileInput');
                // Check if a file is selected
                if (fileInput.files.length > 0) {
                    // Append the selected file to the FormData object
                    formData.append('image', fileInput.files[0]);

                    var xhr = new XMLHttpRequest();
                    xhr.open('POST', '/predict', true);
                    xhr.onload = function() {
                        if (xhr.status === 200) {
                            var response = JSON.parse(xhr.responseText);
                            // Process the prediction response here
                            document.getElementById('predictedPlantClass').innerText = response.predicted_plant_class;
                            document.getElementById('predictedDiseaseStatus').innerText = response.predicted_disease_status;
                        } else {
                            // Handle error response
                            console.error('Error:', xhr.responseText);
                        }
                    };
                    xhr.send(formData);
                } else {
                    console.error('No file selected');
                }
            });
        });
    </script>
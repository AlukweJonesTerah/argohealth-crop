//$(document).ready(function(){
//  $('form input').change(function () {
//    $('form p').text(this.files.length + " file(s) selected");
//  });
//});

$(document).ready(function(){
    // When a file is selected
    $('#image').change(function () {
      // Get the selected file
      var file = this.files[0];
      // Update the paragraph text to show the file name
      $('p').text(file.name + " selected");

      // Create a FileReader object
      var reader = new FileReader();
      // When the FileReader has loaded the file
      reader.onload = function (e) {
        // Update the image source to display the selected image
        $('#selected-image').attr('src', e.target.result);
      }
      // Read the selected file as a data URL
      reader.readAsDataURL(file);
    });
  });

  $(document).ready(function() {
  $('#image').change(function () {
    var files = $(this)[0].files;
    if (files.length > 0) {
      var fileName = files[0].name;
      $('#file-text').text(fileName);
      $('#preview-image').attr('src', URL.createObjectURL(files[0]));
      $('#preview-image').css('display', 'block');
    } else {
      $('#file-text').text('No file chosen');
      $('#preview-image').css('display', 'none');
    }
  });
});
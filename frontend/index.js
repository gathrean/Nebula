document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('audioFile');
    const status = document.getElementById('status');

    const file = fileInput.files[0];
    if (!file) {
        status.innerText = 'Please select a file.';
        return;
    }

    const formData = new FormData();
    formData.append('audio', file);

    // Assuming you'll send this data to a Python backend
    // You can use fetch or XMLHttpRequest to send the file data to your backend
    // Example using fetch:
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            status.innerText = 'File uploaded successfully!';
        } else {
            status.innerText = 'File upload failed.';
        }
    })
    .catch(error => {
        status.innerText = 'An error occurred while uploading the file.';
        console.error('Error:', error);
    });
});

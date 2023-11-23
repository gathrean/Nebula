// Get DOM elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('audioFile');
const status = document.getElementById('status');

// Function to upload file data using fetch
function handleFileUpload(formData) {
    return fetch('/upload', {
        method: 'POST',
        body: formData
    });
}

// Function to display upload status based on response
function displayUploadStatus(response) {
    if (response.ok) {
        status.innerText = 'File uploaded successfully!';
    } else {
        status.innerText = 'File upload failed.';
    }
}

// Function to handle a selected file
function handleSelectedFile(file) {
    if (!file) {
        status.innerText = 'Please select a file.';
        return;
    }

    const formData = new FormData();
    formData.append('audio', file);

    // Upload the file
    handleFileUpload(formData)
        .then(response => displayUploadStatus(response))
        .catch(error => {
            status.innerText = 'An error occurred while uploading the file.';
            console.error('Error:', error);
        });
}

// Function to handle dropped files
function handleDroppedFiles(files) {
    if (files.length === 0) {
        status.innerText = 'No files dropped.';
        return;
    }

    const file = files[0];
    handleSelectedFile(file);
}

// Function to handle file input change
function handleFileInputChange(event) {
    const selectedFile = event.target.files[0];
    handleSelectedFile(selectedFile);
}

// Event listeners for drag and drop functionality
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

// Event listener for dropping files into the drop zone
dropZone.addEventListener('drop', function(event) {
    preventDefaults(event);
    unhighlight();

    const dt = event.dataTransfer;
    const files = dt.files;

    handleDroppedFiles(files);
}, false);

// Function to prevent default behavior
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Functions to highlight and unhighlight the drop zone
function highlight() {
    dropZone.classList.add('highlight');
}

function unhighlight() {
    dropZone.classList.remove('highlight');
}

// Event listener for file input change
fileInput.addEventListener('change', handleFileInputChange);

// Event listener for click on the drop zone to trigger file input
dropZone.addEventListener('click', () => fileInput.click());

// Function to handle the drop anywhere on the webpage
document.addEventListener('dragenter', function(event) {
    event.preventDefault();
    highlight();
});

document.addEventListener('dragover', function(event) {
    event.preventDefault();
});

document.addEventListener('dragleave', function(event) {
    event.preventDefault();
    unhighlight();
});

document.addEventListener('drop', function(event) {
    event.preventDefault();
    unhighlight();

    const dt = event.dataTransfer;
    const files = dt.files;

    handleDroppedFiles(files);
}, false);
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('audioFile');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (event) => {
    event.stopPropagation();
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
});

dropZone.addEventListener('drop', (event) => {
    event.stopPropagation();
    event.preventDefault();
    const files = event.dataTransfer.files;
    handleFileUpload(files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFileUpload(fileInput.files[0]);
    }
});

function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('audio', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayPredictedLabels(data.predicted);
        } else {
            document.getElementById('status').textContent = 'Error: ' + data.error;
        }
    })
    .catch(error => {
        document.getElementById('status').textContent = 'Error: ' + error;
    });
}

function displayPredictedLabels(labels) {
    const labelsContainer = document.getElementById('predictedLabels');
    labelsContainer.innerHTML = '';
    labels.forEach(label => {
        const labelElement = document.createElement('div');
        labelElement.textContent = label;
        labelsContainer.appendChild(labelElement);
    });
}

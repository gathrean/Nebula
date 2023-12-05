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

function displayPredictedLabels(labelsData) {
    const labelsContainer = document.getElementById('predictedLabels');
    labelsContainer.innerHTML = '';

    let labelsString = '';
    if (Array.isArray(labelsData)) {
        labelsString = labelsData.join(',');
    } else if (typeof labelsData === 'string') {
        labelsString = labelsData;
    } else {
        console.error('Invalid labels data type:', labelsData);
        return;
    }

    const labelParts = labelsString.split(',').filter(part => part.includes(':'));
    
    // Loop through all elements in labelParts
    for (let i = 0; i < labelParts.length; i++) {
        const part = labelParts[i];
        const [instrument, probability] = part.split(':').map(s => s.trim());
        
        // Check if instrument and probability are valid
        if (instrument && probability) {
            const probabilityPercentage = (parseFloat(probability) * 100).toFixed(2) + '%';
            const labelElement = document.createElement('div');
            labelElement.textContent = `${instrument}`;
            labelsContainer.appendChild(labelElement);
        }
    }
}






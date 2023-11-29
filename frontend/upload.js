function handleFileUpload(formData) {
    return fetch('http://localhost:3000/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('Server Response:', response);
        return response;
    });
}


function displayPredictionResult(data) {
    console.log('Received data from server:', data);

    const predictionResult = document.getElementById('predictionResult');
    if (data.success) {
        const predictedLabels = data.predicted.join(', ');
        predictionResult.innerHTML = `<p>Prediction Result: ${predictedLabels}</p>`;
    } else {
        predictionResult.innerHTML = `<p>Error: ${data.error}</p>`;
    }
}

function handleSelectedFile(file) {
    const formData = new FormData();
    formData.append('audio', file);

    handleFileUpload(formData)
    .then(response => {
        if (!response.ok) {
            throw new Error('File upload failed.');
        }
        return response.json();
    })
    .then(data => {
        displayPredictionResult(data);
    })
    .catch(error => {
        status.innerText = 'An error occurred while uploading the file.';
        console.error('Error:', error);
    });
}

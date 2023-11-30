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

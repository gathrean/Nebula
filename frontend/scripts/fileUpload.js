let animationTimer;

function handleFileUpload(file) {
    const nebulaNone = document.querySelector('.character-container:nth-child(1)');
    const nebulaLoading = document.querySelector('.character-container:nth-child(2)');
    const nebulaDone = document.querySelector('.character-container:nth-child(3)');
    const predictedLabels = document.getElementById('predictedLabels');
    const videoElement = document.querySelector('.character-container:nth-child(2) video');

    // Hide 'None' and 'Done' images, show 'Loading' video
    nebulaNone.style.display = 'none';
    nebulaDone.style.display = 'none';
    nebulaLoading.style.display = 'block';
    predictedLabels.style.display = 'none'; // Hide predicted labels initially
    
    // Resetting the video
    if (!videoElement.paused || videoElement.currentTime > 0) {
        videoElement.pause();
        videoElement.currentTime = 0; // Reset video to start
        videoElement.play(); // Start playing the video
    }

    const formData = new FormData();
    formData.append('audio', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                clearInterval(animationTimer); // Clear the timer
                // Show 'Done' image after a delay (6 seconds minimum)
                animationTimer = setTimeout(() => {
                    nebulaLoading.style.display = 'none';
                    nebulaDone.style.display = 'block';
                    predictedLabels.style.display = 'block'; // Show predicted labels when Done is displayed
                    displayPredictedLabels(data.predicted);
                }, 6000);
            } else {
                document.getElementById('status').textContent = 'Error: ' + data.error;
                // Show 'None' image in case of error
                nebulaLoading.style.display = 'none';
                nebulaNone.style.display = 'block';
            }
        })
        .catch(error => {
            document.getElementById('status').textContent = 'Error: ' + error;
            // Show 'None' image in case of error
            nebulaLoading.style.display = 'none';
            nebulaNone.style.display = 'block';
        });
}